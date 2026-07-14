## How to run

### 1. Install

Move to the folder in which you want the outputs to be written, then clone the
repository and create the conda environment:

```bash
cd "folder/in/which/to/output"
git clone https://github.com/bertcoerver/fao-aquastat-wapor-vars
conda env create --file=fao-aquastat-wapor-vars/environment.yml
conda activate aquapor-env
```

### 2. Get NASA Earthdata credentials

The script downloads IMERG precipitation data, which requires a (free) NASA
Earthdata account. Create one at <https://urs.earthdata.nasa.gov/> and pass the
username and password via `--authenticate` (if omitted, the script will prompt
for them interactively).

### 3. Run the script

```bash
python fao-aquastat-wapor-vars/aquapor/main.py --years 2023 2024 --authenticate username password
```

The available options are:

| option | description |
|---|---|
| `-y`, `--years` | One or more years between 2018 and last year. Omit to run only the most recent completed year. |
| `-a`, `--authenticate` | Earthdata username and password. Omit to be prompted interactively. |
| `-g`, `--aggregations` | One or more of `country`, `basin`, `subbasin` (default: `country`). |

For example, to compute the statistics for 2024 aggregated per country **and**
per basin in one run:

```bash
python fao-aquastat-wapor-vars/aquapor/main.py --years 2024 --authenticate username password --aggregations country basin
```

### Aggregation levels

By default statistics are computed per **country**. Use `--aggregations` (`-g`) to
select one or more of `country`, `basin` and `subbasin`; each level writes its own
`results_<level>.csv`. The climate inputs are downloaded only once and reused
across the selected levels.

| level | polygons | id field | writes |
|---|---|---|---|
| `country` | UN areas (incl. EU & China clusters) | `M49` | `results_country.csv` |
| `basin` | FAO Major Hydrological Basins | `FAO_MB_ID` | `results_basin.csv` |
| `subbasin` | FAO Sub-Basins | `FAO_SB_ID` | `results_subbasin.csv` |

## Code structure

The code is split over a few modules in [`aquapor/`](aquapor/):

- `main.py` — command line entry point; parses arguments and calls `pipeline.run`.
- `pipeline.py` — one function per workflow stage, plus a `run` orchestrator.
- `download.py` — building download URLs and fetching input files.
- `raster.py` — GDAL-based raster operations (rasterizing, merging, preprocessing).
- `aggregations.py` — registry describing each aggregation level (shapefile URL, id/name fields, groupings).

## Running steps manually

Each stage of the workflow is a function in `pipeline.py` that takes the outputs
of the preceding stages as arguments, so steps can be run one at a time (e.g.
from a REPL). The climate steps are shared; the vector steps take an aggregation
config from `aggregations.py`. Run this from inside the `aquapor/` folder:

```python
import geopandas as gpd

import pipeline
from aggregations import AGGREGATIONS

workdir, years = "/path/to/workdir", [2023, 2024]
aggregation = "country"  # or "basin" / "subbasin"

folders = pipeline.setup_folders(workdir, [aggregation])

# Download & merge the (shared) climate inputs
imerg_fhs, chirps_fhs, aeti_fhs = pipeline.download_climate_data(
    years, ("username", "password"), folders
)
pcp_fhs = pipeline.merge_precipitation(imerg_fhs, chirps_fhs, aeti_fhs, years)
ds_base = pipeline.open_climate_data(aeti_fhs, pcp_fhs)

# Vector steps for one aggregation
agg_cfg = AGGREGATIONS[aggregation]
shp = pipeline.download_aggregation_vector(aggregation, folders[aggregation])
grouping_tifs = pipeline.rasterize_vectors(shp, aeti_fhs[0], agg_cfg)
ds = pipeline.attach_groupings(ds_base, grouping_tifs)

gdf = gpd.read_file(shp)
groups = pipeline.expected_groups(gdf, agg_cfg)
output_file = pipeline.results_path(workdir, aggregation)
files = pipeline.calculate_statistics(ds, gdf, agg_cfg, groups, output_file)
df = pipeline.build_dataframe(files, gdf, agg_cfg, output_file)
```

`pipeline.run(years, auth, aggregations, workdir=...)` chains all of the above and
loops over the selected aggregations.
