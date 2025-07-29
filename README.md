## Setup environment 

`cd "folder/in/which/to/output"`

`git clone https://github.com/bertcoerver/fao-aquastat-wapor-vars`

`conda env create --file=fao-aquastat-wapor-vars/environment.yml`

`conda activate aquapor-env`

Create an account at `https://urs.earthdata.nasa.gov/`, then run:

`python fao-aquastat-wapor-vars/aquapor/main.py --years 2023 2024 --authenticate username password`
