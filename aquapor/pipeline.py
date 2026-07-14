"""High-level pipeline steps.

Each function corresponds to one stage of the workflow and can be run on its
own (e.g. from a REPL or a notebook), as long as it is given the outputs of the
preceding steps. `main.py` chains them together for the CLI.

The climate inputs (IMERG, CHIRPS, AETI) are the same regardless of the chosen
spatial aggregation, so they are downloaded and merged once; only the aggregation
polygons (see `aggregations.py`) change between country / basin / subbasin.
"""

import os

import flox.xarray
import geopandas as gpd
import xarray as xr
from dask.diagnostics import ProgressBar
from pyproj import Geod

from aggregations import AGGREGATIONS, AGGREGATION_CHOICES
from download import (
    download_shapefile,
    download_urls,
    imerg_missing_months,
    is_valid,
    make_urls,
)
from raster import merge_tifs, preprocess, preprocess_imerg, rasterize

# Geodesic area on the WGS84 spheroid, matching the method behind the legacy
# UN_AREAS.gpkg `area` column (PostGIS ST_Area(geography) / pyproj Geod).
GEOD = Geod(ellps="WGS84")


def geodesic_area_km2(gdf):
    """Return the geodesic (WGS84) area of each feature in `gdf`, in km2."""
    lonlat = gdf.to_crs("EPSG:4326").geometry
    return lonlat.apply(lambda geom: abs(GEOD.geometry_area_perimeter(geom)[0]) / 1e6)


def results_path(workdir, aggregation):
    """Path of the output CSV for a given aggregation."""
    return os.path.join(workdir, f"results_{aggregation}.csv")


def setup_folders(workdir, aggregations):
    """Create (if needed) and return the shared climate folders plus one vector
    folder per selected aggregation."""
    folders = {
        "chirps": os.path.join(workdir, "CHIRPS_v3"),
        "imerg": os.path.join(workdir, "IMERG_v7"),
        "aeti": os.path.join(workdir, "L1-AETI-A"),
    }
    for aggregation in aggregations:
        folders[aggregation] = os.path.join(workdir, aggregation)

    for folder in folders.values():
        if not os.path.isdir(folder):
            os.makedirs(folder)

    return folders


def download_climate_data(years, auth, folders):
    """Download (and preprocess) the IMERG, CHIRPS and AETI rasters."""
    missing = imerg_missing_months(years, folders["imerg"], auth=auth)
    if missing:
        details = "; ".join(
            f"{year}: {12 - len(gaps)}/12 months published"
            for year, gaps in sorted(missing.items())
        )
        raise RuntimeError(
            f"IMERG monthly data is not yet complete for the requested year(s) "
            f"({details}). The GPM 3IMERGM Final product lags real time by several "
            f"months, and a yearly precipitation total needs all 12 months. Rerun "
            f"once the missing months are published, or drop the incomplete year(s)."
        )

    imerg_fhs = preprocess_imerg(
        download_urls(
            make_urls("IMERG_v7", years), folders["imerg"], is_valid=is_valid, auth=auth
        )
    )
    chirps_fhs = download_urls(
        make_urls("CHIRPS_v3", years), folders["chirps"], is_valid=is_valid
    )
    aeti_fhs = download_urls(
        make_urls("L1-AETI-A", years), folders["aeti"], is_valid=is_valid
    )

    assert len(imerg_fhs) == len(aeti_fhs) == len(chirps_fhs)

    return imerg_fhs, chirps_fhs, aeti_fhs


def download_aggregation_vector(aggregation, folder):
    """Download the zipped shapefile for `aggregation` and return the `.shp` path."""
    return download_shapefile(AGGREGATIONS[aggregation]["url"], folder)


def merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, years):
    """Merge the IMERG and CHIRPS precipitation products per year."""
    chirps = {int(fh.split(".")[-2]): fh for fh in chirps_fhs}
    imerg = {int(fh.split(".")[-2]): fh for fh in imerg_fhs}
    pcp_fhs = [merge_tifs([imerg[year], chirps[year]], aeti_fhs[0]) for year in years]
    return pcp_fhs


def rasterize_vectors(shp, example_raster, agg_cfg):
    """Rasterize each grouping of an aggregation onto the grid of `example_raster`.

    Returns a mapping of grouping name -> rasterized tif.
    """
    tifs = {}
    for grouping, spec in agg_cfg["groupings"].items():
        tifs[grouping] = rasterize(
            shp,
            example_raster,
            agg_cfg["id_field"],
            where=spec["where"],
            label=grouping,
        )
    return tifs


def open_climate_data(aeti_fhs, pcp_fhs):
    """Open the AETI/PCP rasters into one dataset and add the derived IRWR var."""
    ds = xr.open_mfdataset(
        aeti_fhs + pcp_fhs,
        chunks={"x": 2**12, "y": 2**12},
        preprocess=preprocess,
        data_vars="all",
        compat="no_conflicts",
    )
    ds["IRWR"] = (ds["PCP"] - ds["AETI"]).clip(min=0)
    return ds


def attach_groupings(ds, grouping_tifs):
    """Add each rasterized grouping to a copy of `ds` as a coordinate."""
    ds = ds.copy()
    for grouping, tif in grouping_tifs.items():
        ds[grouping] = (
            xr.open_dataarray(tif, chunks="auto")
            .isel(band=0)
            .drop_vars("band")
            .drop_attrs()
        )
    ds = ds.set_coords(list(grouping_tifs)).drop_vars("spatial_ref")
    return ds


def expected_groups(gdf, agg_cfg):
    """Return the integer id values expected in each grouping's raster."""
    id_field = agg_cfg["id_field"]
    ids = gdf[id_field].astype(int)
    groups = {}
    for grouping, spec in agg_cfg["groupings"].items():
        mask = spec["mask"]
        groups[grouping] = (ids if mask is None else ids[mask(gdf)]).values
    return groups


def _assign_names(out_ds, gdf, agg_cfg):
    """Attach the human-readable name of each area as a coordinate on `id`."""
    id_field, name_field = agg_cfg["id_field"], agg_cfg["name_field"]
    lookup = gdf.dropna(subset=[id_field]).copy()
    lookup[id_field] = lookup[id_field].astype(int)
    lookup = lookup.set_index(id_field)[name_field]
    return out_ds.assign_coords(
        name=("id", [lookup.loc[idx] for idx in out_ds["id"].values])
    )


def calculate_statistics(ds, gdf, agg_cfg, groups, output_file):
    """Reduce the dataset to per-area statistics, writing one netCDF per variable.

    `groups` is the mapping returned by `expected_groups`. Alongside the AETI/PCP/
    IRWR statistics a `total-count` (total pixels per area) is written, used later
    as the denominator of the valid fraction.
    """
    stats = {"AETI": ["mean"], "PCP": ["mean"], "IRWR": ["mean", "count"]}

    files = list()
    for var, stats_ in stats.items():
        out_ = output_file.replace(".csv", f"_{var}.nc")
        if not os.path.isfile(out_):
            print(f"Calculating statistics for {var} {stats_}.")
            out_ds = xr.Dataset()
            for stat in stats_:
                out_ds[f"{var.lower()}-{stat}"] = xr.concat(
                    [
                        flox.xarray.xarray_reduce(
                            ds[var],
                            ds[grouping],
                            func=stat,
                            expected_groups=group_ids,
                        ).rename({grouping: "id"})
                        for grouping, group_ids in groups.items()
                    ],
                    dim="id",
                )
            out_ds = _assign_names(out_ds, gdf, agg_cfg)
            _write_netcdf(out_ds, out_)
        files.append(out_)

    # Total number of (assigned) pixels per area, counting each grouping's own
    # coordinate so it works both for countries and the overlying clusters.
    out_ = output_file.replace(".csv", "_TOTAL.nc")
    if not os.path.isfile(out_):
        print("Calculating total pixel count per area.")
        out_ds = xr.Dataset()
        out_ds["total-count"] = xr.concat(
            [
                flox.xarray.xarray_reduce(
                    ds[grouping],
                    ds[grouping],
                    func="count",
                    expected_groups=group_ids,
                ).rename({grouping: "id"})
                for grouping, group_ids in groups.items()
            ],
            dim="id",
        )
        out_ds = _assign_names(out_ds, gdf, agg_cfg)
        _write_netcdf(out_ds, out_)
    files.append(out_)

    return files


def _write_netcdf(out_ds, out_):
    """Write `out_ds` to `out_` (a `.nc` path) with compression, atomically."""
    encoding = {var: {"zlib": True} for var in out_ds.data_vars}
    with ProgressBar():
        _ = out_ds.to_netcdf(out_.replace(".nc", ""), encoding=encoding)
    _ = os.rename(out_.replace(".nc", ""), out_)


def build_dataframe(files, gdf, agg_cfg, output_file):
    """Combine the statistics, compute areas/volumes and write the results CSV."""
    id_field = agg_cfg["id_field"]

    ds = xr.open_mfdataset(files, data_vars="all", compat="no_conflicts")
    df = ds.to_dataframe()
    df = df.reset_index().set_index("id")

    # None of the shapefiles carry an area attribute, so derive km2 from geometry
    # using geodesic area (see GEOD) — the method behind the legacy gpkg `area`
    # column, which it reproduces exactly on the same geometry. Round to int before
    # the volume calc, as the old pipeline did (it multiplied by the integer gpkg area).
    areas = gdf.dropna(subset=[id_field]).copy()
    areas[id_field] = areas[id_field].astype(int)
    area_km2 = geodesic_area_km2(areas.set_index(id_field))
    df["area"] = area_km2.round().astype("Int64")

    for var in ["aeti", "pcp", "irwr"]:
        df[f"{var}-volume"] = (df[f"{var}-mean"] / 1000000) * df["area"]
    df["valid-fraction"] = df["irwr-count"] / df["total-count"] * 100
    df = df.drop(["total-count", "irwr-count"], axis=1)
    df = df.round({"valid-fraction": 0, "irwr-mean": 1, "aeti-mean": 1, "pcp-mean": 1})
    df.loc[
        (df["irwr-mean"].isna()) & (df["valid-fraction"].isna()), "valid-fraction"
    ] = 0
    df = df.rename(
        {"aeti-mean": "aeti-depth", "pcp-mean": "pcp-depth", "irwr-mean": "irwr-depth"},
        axis=1,
    )
    df = df.reset_index().rename(columns={"id": id_field})
    df = df[
        [
            id_field,
            "name",
            "year",
            "valid-fraction",
            "area",
            "aeti-depth",
            "aeti-volume",
            "pcp-depth",
            "pcp-volume",
            "irwr-depth",
            "irwr-volume",
        ]
    ]

    df.to_csv(output_file)

    return df


def run_aggregation(aggregation, ds_base, aeti_fhs, folder, output_file):
    """Run the vector-dependent part of the pipeline for a single aggregation."""
    agg_cfg = AGGREGATIONS[aggregation]

    shp = download_aggregation_vector(aggregation, folder)
    grouping_tifs = rasterize_vectors(shp, aeti_fhs[0], agg_cfg)
    ds = attach_groupings(ds_base, grouping_tifs)

    gdf = gpd.read_file(shp)
    groups = expected_groups(gdf, agg_cfg)
    files = calculate_statistics(ds, gdf, agg_cfg, groups, output_file)
    return build_dataframe(files, gdf, agg_cfg, output_file)


def run(years, auth, aggregations, workdir=None):
    """Run the full pipeline for the given `years`, `auth` and `aggregations`.

    Returns a mapping of aggregation name -> resulting DataFrame. The climate data
    is downloaded and merged once and reused across all aggregations.
    """
    if workdir is None:
        workdir = os.getcwd()

    unknown = [a for a in aggregations if a not in AGGREGATIONS]
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s) {unknown}; choose from "
            f"{', '.join(AGGREGATION_CHOICES)}."
        )

    folders = setup_folders(workdir, aggregations)

    imerg_fhs, chirps_fhs, aeti_fhs = download_climate_data(years, auth, folders)
    pcp_fhs = merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, years)
    ds_base = open_climate_data(aeti_fhs, pcp_fhs)

    results = {}
    for aggregation in aggregations:
        output_file = results_path(workdir, aggregation)
        if os.path.isfile(output_file):
            print(f"Output file `{output_file}` already exists and will be overwritten.")
        results[aggregation] = run_aggregation(
            aggregation, ds_base, aeti_fhs, folders[aggregation], output_file
        )

    return results
