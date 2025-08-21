import argparse
import itertools
import os
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import List

import flox.xarray
import geopandas as gpd
import numpy as np
import requests
import tqdm
import xarray as xr
from dask.diagnostics import ProgressBar
from osgeo import gdal, gdalconst

gdal.UseExceptions()


def is_valid(fp):
    engine = {".cog": "rasterio"}.get(os.path.splitext(fp)[-1], None)
    try:
        if os.path.isfile(fp):
            ds = xr.open_dataset(fp, chunks="auto", engine=engine)
            ds = ds.close()
            is_valid = True
        else:
            is_valid = False
    except Exception:
        is_valid = False
    return is_valid


def download_url(
    url,
    fp: str,
    session=None,
    waitbar=True,
    headers=None,
    verify=True,
    auth=None,
):
    if os.path.isfile(fp):
        os.remove(fp)

    folder, fn = os.path.split(fp)
    if not os.path.exists(folder):
        os.makedirs(folder)

    ext = os.path.splitext(fp)[-1]
    temp_fp = fp.replace(ext, "_temp")

    if isinstance(session, type(None)):
        session = requests.Session()

    redirecting = True
    while redirecting:
        resp = session.get(
            url,
            stream=True,
            allow_redirects=False,
            headers=headers,
            verify=verify,
            auth=auth,
        )
        resp.raise_for_status()
        if resp.status_code == 302:
            url = resp.headers["location"]
        else:
            redirecting = False

    if "Content-Length" in resp.headers.keys():
        tot_size = int(resp.headers["Content-Length"])
    else:
        tot_size = None

    if waitbar:
        wb = tqdm.tqdm(
            unit="Bytes", unit_scale=True, leave=False, total=tot_size, desc=fn
        )

    with open(temp_fp, "wb") as z:
        for data in resp.iter_content(chunk_size=1024):
            size = z.write(data)
            if waitbar:
                wb.update(size)

    os.rename(temp_fp, fp)

    return fp


def download_urls(
    urls, folder, is_valid=lambda x: os.path.isfile(x), auth=None
) -> List[str]:
    fns = [os.path.split(x) for x in urls]

    to_download = True
    attempt = 1
    max_attempts = 10

    while to_download and attempt <= max_attempts:
        to_download = [
            os.path.join(base_url, fn)
            for base_url, fn in fns
            if not is_valid(os.path.join(folder, fn))
        ]
        print(
            f"Still {len(to_download)} files left to download into `{os.path.split(folder)[-1]}`."
        )

        for url in to_download:
            fn = os.path.split(url)[-1]
            print(f"Downloading `{fn}`")
            fp = os.path.join(folder, fn)
            try:
                _ = download_url(url, fp, auth=auth)
            except Exception as e:
                print(f"Download attempt {attempt}/{max_attempts} of `{url}` failed.")
                attempt += 1
                if attempt == max_attempts:
                    raise e

    out_fhs = [os.path.join(folder, fn[1]) for fn in fns]

    return out_fhs


def make_urls(product, years):
    if product == "IMERG_v7":
        urls = list(
            itertools.chain.from_iterable(
                [
                    [
                        f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/{year}/3B-MO.MS.MRG.3IMERG.{year}{month:>02}01-S000000-E235959.{month:>02}.V07B.HDF5"
                        for month in range(1, 13)
                    ]
                    for year in years
                ]
            )
        )
    elif product == "CHIRPS_v3":
        urls = [
            f"https://data.chc.ucsb.edu/products/CHIRPS/v3.0/annual/global/tifs/chirps-v3.0.{year}.tif"
            for year in years
        ]
    elif product == "L1-AETI-A":
        urls = [
            f"https://storage.googleapis.com/fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-A/WAPOR-3.L1-AETI-A.{year}.tif"
            for year in years
        ]
    else:
        raise ValueError

    return urls


def rasterize(
    vector: str,
    example_raster: str,
    attribute: str,
    dtype=gdalconst.GDT_Int32,
    ndv=-9999,
    where=None,
):
    vector_ext = os.path.splitext(vector)[-1]
    new_ext = ".tif" if where is None else f"{where}.tif"
    out_fh = os.path.join(
        os.getcwd(), os.path.split(vector)[-1].replace(vector_ext, new_ext)
    )

    if not os.path.isfile(out_fh):
        info = gdal.Info(example_raster, format="json")

        width, height = info["size"]
        bounds = (
            info["cornerCoordinates"]["lowerLeft"]
            + info["cornerCoordinates"]["upperRight"]
        )

        vector_info = gdal.VectorInfo(Path(vector), format="json")
        fields = [
            x["type"]
            for x in vector_info["layers"][0]["fields"]
            if x["name"] == attribute
        ]
        if not fields:
            raise ValueError(f"Attribute `{attribute}` not found in vector fields.")

        waitbar = tqdm.tqdm(
            desc=f"Rasterizing {os.path.split(out_fh)[-1]}",
            leave=False,
            total=100,
            bar_format="{l_bar}{bar}|",
        )

        def _callback_func(info, *args):
            waitbar.update(info * 100 - waitbar.n)

        options = gdal.RasterizeOptions(
            attribute=attribute,
            outputType=dtype,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
            noData=ndv,
            outputBounds=bounds,
            width=width,
            height=height,
            where=where,
            callback=_callback_func,
        )

        x: gdal.Dataset = gdal.Rasterize(Path(out_fh), Path(vector), options=options)
        x.FlushCache()
        x = None

    return out_fh


def merge_tifs(
    rasters: List[str],
    example_raster: str,
    dstNodata=-9999,
    srcNodata=-9999,
    outputType=gdalconst.GDT_Int16,
):
    fns = [os.path.splitext(os.path.split(fh)[-1])[0] for fh in rasters]
    out_fh = os.path.join(
        os.path.split(rasters[0])[0], "__mergedwith__".join(fns) + ".tif"
    )

    if not os.path.isfile(out_fh):
        waitbar = tqdm.tqdm(
            desc="Merging files.",
            leave=False,
            total=100,
            bar_format="{l_bar}{bar}|",
        )

        def _callback_func(info, *args):
            waitbar.update(info * 100 - waitbar.n)

        info = gdal.Info(example_raster, format="json")

        width, height = info["size"]
        bounds = (
            info["cornerCoordinates"]["lowerLeft"]
            + info["cornerCoordinates"]["upperRight"]
        )

        options = gdal.WarpOptions(
            height=height,
            width=width,
            outputType=outputType,
            outputBounds=bounds,
            dstNodata=dstNodata,
            srcNodata=srcNodata,
            dstSRS="epsg:4326",
            creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
            callback=_callback_func,
            multithread=True,
            warpMemoryLimit=500,
            resampleAlg="bilinear",
        )

        x: gdal.Dataset = gdal.Warp(out_fh, rasters, options=options)
        x.FlushCache()
        x = None

    return out_fh


def preprocess_imerg(imerg_fhs: List[str]) -> List[str]:
    fhs = list()
    ds = xr.open_mfdataset(imerg_fhs, group="Grid")
    ds = (
        ds[["precipitation"]]
        .transpose("lat", "lon", ...)
        .rename({"lat": "y", "lon": "x"})
        .drop_attrs()
    )
    hours_in_month = ds.time.dt.days_in_month * 24
    ds["precipitation"] *= hours_in_month
    ds["precipitation"] = ds["precipitation"].groupby(ds.time.dt.year).sum(min_count=12)
    ds["precipitation"] = ds["precipitation"].where((ds["y"] >= 60.0 - 0.1), np.nan)
    ds["precipitation"] = ds["precipitation"].where(
        ~ds["precipitation"].isnull(), -9999
    )
    ds["precipitation"] = ds["precipitation"].rio.write_nodata(-9999)
    ds["precipitation"].attrs.update({"grid_mapping": "spatial_ref"})
    ds = ds.drop_dims("time")
    folder = os.path.split(imerg_fhs[0])[0]
    for year in ds["year"].values:
        imerg_yearly = os.path.join(folder, f"IMERG_v7.PCP-A.{year}.tif")
        fhs.append(imerg_yearly)
        ds["precipitation"].sel(year=year).rio.to_raster(imerg_yearly)
    return fhs


def preprocess(ds: xr.Dataset):
    fn = os.path.split(ds.encoding["source"])[-1]
    if "AETI" in fn:
        var = "AETI"
    else:
        var = "PCP"
    year = int(fn.split(".")[-2])
    ds = (
        ds.rename({"band_data": var, "band": "year"})
        .assign_coords({"year": [year]})
        .drop_attrs()
    )
    return ds


if __name__ == "__main__":
    imerg_url = "https://urs.earthdata.nasa.gov/"
    current_year = datetime.today().year

    #######
    ## Parse arguments from CLI.
    #######
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        help=f"Define years between 2018 and {current_year - 1}",
    )
    _ = parser.add_argument(
        "-a", "--authenticate", nargs=2, help=f"Create an account at {imerg_url}"
    )
    args = parser.parse_args()

    years = []
    if args.years is None:
        years = [current_year - 1]
        print(f"No --years defined, continueing with {years}.")
    else:
        for year_str in args.years:
            year = int(year_str)
            if year < 2018:
                print(f"Skipping {year}, earliest allowed year is 2018.")
                continue
            if year >= current_year:
                print(f"Skipping {year}, latest allowed year is {current_year - 1}.")
                continue
            years.append(year)

    if args.authenticate is None:
        print(
            f"Please specify a username and password (create an account here `{imerg_url}`) to download IMERG data. You can also specify it using --authenticate."
        )
        un = input("Username: ")
        pw = getpass("Password: ")
        auth = (un, pw)
    else:
        auth = tuple([str(x) for x in args.authenticate])

    # auth = ("", "")
    # years = [2018]
    
    #######
    ## Define paths
    #######
    workdir = os.getcwd()
    output_file = os.path.join(workdir, "results.csv")
    if os.path.isfile(output_file):
        print(f"Output file `{output_file}` already exists and will be overwritten.")

    chirps_folder = os.path.join(workdir, "CHIRPS_v3")
    imerg_folder = os.path.join(workdir, "IMERG_v7")
    aeti_folder = os.path.join(workdir, "L1-AETI-A")
    country_folder = os.path.join(workdir, "countries")
    for folder in [chirps_folder, imerg_folder, aeti_folder, country_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    #######
    ## Download input data
    #######
    countries_gpkg = download_urls(["https://storage.googleapis.com/fao-cog-data/aquapor/UN_AREAS.gpkg"], country_folder)[0]

    imerg_fhs = preprocess_imerg(
        download_urls(
            make_urls("IMERG_v7", years), imerg_folder, is_valid=is_valid, auth=auth
        )
    )
    chirps_fhs = download_urls(
        make_urls("CHIRPS_v3", years), chirps_folder, is_valid=is_valid
    )
    aeti_fhs = download_urls(
        make_urls("L1-AETI-A", years), aeti_folder, is_valid=is_valid
    )

    assert len(imerg_fhs) == len(aeti_fhs) == len(chirps_fhs)

    #######
    ## Merge two precipitation products
    #######
    chirps = {int(fh.split(".")[-2]): fh for fh in chirps_fhs}
    imerg = {int(fh.split(".")[-2]): fh for fh in imerg_fhs}
    pcp_fhs = [merge_tifs([imerg[year], chirps[year]], aeti_fhs[0]) for year in years]

    #######
    ## Rasterize vector data
    #######
    countries_fh = rasterize(
        countries_gpkg, aeti_fhs[0], "m49", where="is_cluster=0"
    )
    clusters_fh = rasterize(
        countries_gpkg, aeti_fhs[0], "m49", where="is_cluster=1"
    )

    #######
    ## Open data
    #######
    ds = xr.open_mfdataset(
        aeti_fhs + pcp_fhs, chunks={"x": 2**12, "y": 2**12}, preprocess=preprocess
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

    #######
    ## Calculate
    #######
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
                output = out_ds.to_netcdf(out_.replace(".nc", ""), encoding=encoding)
            _ = os.rename(out_.replace(".nc", ""), out_)
        files.append(out_)

    #######
    ## Convert to DataFrame
    #######
    ds = xr.open_mfdataset(files)
    df = ds.to_dataframe()
    df = df.reset_index().set_index("m49")

    #######
    ## Calculate Volumes
    #######
    df["area"] = gdf.set_index("m49")["area"] # area is in km2
    for var in ["aeti", "pcp", "irwr"]:
        df[f"{var}-volume"] = (df[f"{var}-mean"] / 1000000) * df["area"]
    df["valid-fraction"] = df["irwr-count"] / df["country-count"] * 100
    df = df.drop(["country-count", "irwr-count"], axis=1)
    df = df.round({"valid-fraction": 0, "irwr-mean": 1, "aeti-mean": 1, "pcp-mean": 1})
    df.loc[
        (df["irwr-mean"].isna()) & (df["valid-fraction"].isna()), "valid-fraction"
    ] = 0
    df = df.rename({"aeti-mean": "aeti-depth", "pcp-mean": "pcp-depth", "irwr-mean": "irwr-depth"}, axis=1)
    df = df.reset_index()[
        ["m49", "name", "year", "valid-fraction", "area", "aeti-depth", "aeti-volume", "pcp-depth", "pcp-volume", "irwr-depth", "irwr-volume"]
    ]

    df.to_csv(output_file)
