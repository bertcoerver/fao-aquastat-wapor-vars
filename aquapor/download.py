"""Utilities for building download URLs and fetching input files."""

import itertools
import os
from typing import List

import requests
import tqdm
import xarray as xr


def is_valid(fp):
    """Return True if `fp` exists and can be opened as a dataset."""
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
    """Download a single `url` to `fp`, following redirects."""
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
    """Download all `urls` into `folder`, retrying invalid/missing files."""
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
    """Build the list of download URLs for a given `product` and `years`."""
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
