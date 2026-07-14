"""High-level pipeline steps.

Each function corresponds to one stage of the workflow and can be run on its
own (e.g. from a REPL or a notebook), as long as it is given the outputs of the
preceding steps. `main.py` chains them together for the CLI.
"""

import os

import flox.xarray
import geopandas as gpd
import xarray as xr
from dask.diagnostics import ProgressBar

from download import download_urls, is_valid, make_urls
from raster import merge_tifs, preprocess, preprocess_imerg, rasterize

COUNTRIES_URL = "https://storage.googleapis.com/fao-cog-data/aquapor/UN_AREAS.gpkg"


def setup_folders(workdir):
    """Create (if needed) and return the working folders and output path."""
    output_file = os.path.join(workdir, "results.csv")
    if os.path.isfile(output_file):
        print(f"Output file `{output_file}` already exists and will be overwritten.")

    folders = {
        "chirps": os.path.join(workdir, "CHIRPS_v3"),
        "imerg": os.path.join(workdir, "IMERG_v7"),
        "aeti": os.path.join(workdir, "L1-AETI-A"),
        "country": os.path.join(workdir, "countries"),
    }
    for folder in folders.values():
        if not os.path.isdir(folder):
            os.makedirs(folder)

    return folders, output_file


def download_input_data(years, auth, folders):
    """Download the country vectors and the IMERG, CHIRPS and AETI rasters."""
    countries_gpkg = download_urls([COUNTRIES_URL], folders["country"])[0]

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

    return countries_gpkg, imerg_fhs, chirps_fhs, aeti_fhs


def merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, years):
    """Merge the IMERG and CHIRPS precipitation products per year."""
    chirps = {int(fh.split(".")[-2]): fh for fh in chirps_fhs}
    imerg = {int(fh.split(".")[-2]): fh for fh in imerg_fhs}
    pcp_fhs = [merge_tifs([imerg[year], chirps[year]], aeti_fhs[0]) for year in years]
    return pcp_fhs


def rasterize_vectors(countries_gpkg, example_raster):
    """Rasterize the countries and clusters onto the grid of `example_raster`."""
    countries_fh = rasterize(countries_gpkg, example_raster, "m49", where="is_cluster=0")
    clusters_fh = rasterize(countries_gpkg, example_raster, "m49", where="is_cluster=1")
    return countries_fh, clusters_fh


def open_data(aeti_fhs, pcp_fhs, countries_fh, clusters_fh, countries_gpkg):
    """Open the rasters into one dataset and add country/cluster coordinates."""
    ds = xr.open_mfdataset(
        aeti_fhs + pcp_fhs,
        chunks={"x": 2**12, "y": 2**12},
        preprocess=preprocess,
        data_vars="all",
        compat="no_conflicts",
    )
    ds["country"] = (
        xr.open_dataarray(countries_fh, chunks="auto")
        .isel(band=0)
        .drop_vars("band")
        .drop_attrs()
    )
    ds["cluster"] = (
        xr.open_dataarray(clusters_fh, chunks="auto")
        .isel(band=0)
        .drop_vars("band")
        .drop_attrs()
    )
    ds = ds.set_coords(["country", "cluster"]).drop_vars("spatial_ref")
    ds["IRWR"] = (ds["PCP"] - ds["AETI"]).clip(min=0)

    gdf = gpd.read_file(countries_gpkg)
    country_coords = gdf.loc[gdf["is_cluster"] == 0]["m49"].values
    cluster_coords = gdf.loc[gdf["is_cluster"] == 1]["m49"].values

    return ds, gdf, country_coords, cluster_coords


def calculate_statistics(ds, gdf, country_coords, cluster_coords, output_file):
    """Reduce the dataset to per-area statistics, writing one netCDF per var."""
    groupers = {"country": country_coords, "cluster": cluster_coords}
    stats = {
        "AETI": ["mean"],
        "PCP": ["mean"],
        "IRWR": ["mean", "count"],
        "country": ["count"],
    }

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
                            ds[coord],
                            func=stat,
                            expected_groups=coord_groups,
                        ).rename({coord: "m49"})
                        for coord, coord_groups in groupers.items()
                    ],
                    dim="m49",
                )
            out_ds = out_ds.assign_coords(
                name=(
                    "m49",
                    [
                        gdf.set_index("m49").loc[idx]["disp_en"]
                        for idx in out_ds["m49"].values
                    ],
                )
            )
            encoding = {var: {"zlib": True} for var in out_ds.data_vars}
            with ProgressBar():
                _ = out_ds.to_netcdf(out_.replace(".nc", ""), encoding=encoding)
            _ = os.rename(out_.replace(".nc", ""), out_)
        files.append(out_)

    return files


def build_dataframe(files, gdf, output_file):
    """Combine the statistics, compute volumes and write the results CSV."""
    ds = xr.open_mfdataset(files, data_vars="all", compat="no_conflicts")
    df = ds.to_dataframe()
    df = df.reset_index().set_index("m49")

    df["area"] = gdf.set_index("m49")["area"]  # area is in km2
    for var in ["aeti", "pcp", "irwr"]:
        df[f"{var}-volume"] = (df[f"{var}-mean"] / 1000000) * df["area"]
    df["valid-fraction"] = df["irwr-count"] / df["country-count"] * 100
    df = df.drop(["country-count", "irwr-count"], axis=1)
    df = df.round({"valid-fraction": 0, "irwr-mean": 1, "aeti-mean": 1, "pcp-mean": 1})
    df.loc[
        (df["irwr-mean"].isna()) & (df["valid-fraction"].isna()), "valid-fraction"
    ] = 0
    df = df.rename(
        {"aeti-mean": "aeti-depth", "pcp-mean": "pcp-depth", "irwr-mean": "irwr-depth"},
        axis=1,
    )
    df = df.reset_index()[
        [
            "m49",
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


def run(years, auth, workdir=None):
    """Run the full pipeline end-to-end for the given `years` and `auth`."""
    if workdir is None:
        workdir = os.getcwd()

    folders, output_file = setup_folders(workdir)

    countries_gpkg, imerg_fhs, chirps_fhs, aeti_fhs = download_input_data(
        years, auth, folders
    )
    pcp_fhs = merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, years)
    countries_fh, clusters_fh = rasterize_vectors(countries_gpkg, aeti_fhs[0])
    ds, gdf, country_coords, cluster_coords = open_data(
        aeti_fhs, pcp_fhs, countries_fh, clusters_fh, countries_gpkg
    )
    files = calculate_statistics(
        ds, gdf, country_coords, cluster_coords, output_file
    )
    df = build_dataframe(files, gdf, output_file)

    return df
