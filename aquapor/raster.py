"""Raster operations: rasterizing vectors, merging tifs and preprocessing."""

import os
from pathlib import Path
from typing import List

import numpy as np
import rioxarray  # noqa: F401  (registers the `.rio` xarray accessor)
import tqdm
import xarray as xr
from osgeo import gdal, gdalconst

gdal.UseExceptions()


def rasterize(
    vector: str,
    example_raster: str,
    attribute: str,
    dtype=gdalconst.GDT_Int32,
    ndv=-9999,
    where=None,
    label=None,
):
    """Burn `attribute` from `vector` onto the grid of `example_raster`.

    `label`, when given, is used as the output filename suffix instead of the
    (possibly messy) `where` clause, e.g. `label="country"` -> `<name>_country.tif`.
    """
    vector_ext = os.path.splitext(vector)[-1]
    if label is not None:
        new_ext = f"_{label}.tif"
    else:
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
    """Warp/merge `rasters` onto the grid of `example_raster`."""
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
    """Aggregate monthly IMERG HDF5 files into yearly precipitation tifs."""
    fhs = list()
    ds = xr.open_mfdataset(
        imerg_fhs, group="Grid", data_vars="all", compat="no_conflicts"
    )
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
    """`preprocess` hook for `xr.open_mfdataset` to name vars/years from filenames."""
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
