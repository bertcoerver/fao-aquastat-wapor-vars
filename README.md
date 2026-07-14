## How to run

`cd "folder/in/which/to/output"`

`git clone https://github.com/bertcoerver/fao-aquastat-wapor-vars`

`conda env create --file=fao-aquastat-wapor-vars/environment.yml`

`conda activate aquapor-env`

Create an account at `https://urs.earthdata.nasa.gov/`, then run for example (omit `--years` to only run most recent year):

`python fao-aquastat-wapor-vars/aquapor/main.py --years 2023 2024 --authenticate username password`

## Code structure

The code is split over a few modules in [`aquapor/`](aquapor/):

- `main.py` — command line entry point; parses arguments and calls `pipeline.run`.
- `pipeline.py` — one function per workflow stage, plus a `run` orchestrator.
- `download.py` — building download URLs and fetching input files.
- `raster.py` — GDAL-based raster operations (rasterizing, merging, preprocessing).

## Running steps manually

Each stage of the workflow is a function in `pipeline.py` that takes the outputs
of the preceding stages as arguments, so steps can be run one at a time (e.g.
from a REPL). Run this from inside the `aquapor/` folder:

```python
import pipeline

folders, output_file = pipeline.setup_folders("/path/to/workdir")

# Download input data
countries_gpkg, imerg_fhs, chirps_fhs, aeti_fhs = pipeline.download_input_data(
    [2023, 2024], ("username", "password"), folders
)

# Merge two precipitation products
pcp_fhs = pipeline.merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, [2023, 2024])

# Rasterize vector data
countries_fh, clusters_fh = pipeline.rasterize_vectors(countries_gpkg, aeti_fhs[0])

# Open data
ds, gdf, country_coords, cluster_coords = pipeline.open_data(
    aeti_fhs, pcp_fhs, countries_fh, clusters_fh, countries_gpkg
)

# Calculate statistics and write the results CSV
files = pipeline.calculate_statistics(
    ds, gdf, country_coords, cluster_coords, output_file
)
df = pipeline.build_dataframe(files, gdf, output_file)
```
