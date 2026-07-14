"""Command line entry point for the AQUAPOR workflow.

The actual work lives in `pipeline.py` (the pipeline steps), `download.py`
(fetching input data) and `raster.py` (raster operations). This module only
parses the CLI arguments and calls `pipeline.run`.
"""

import argparse
from datetime import datetime
from getpass import getpass

import pipeline

IMERG_URL = "https://urs.earthdata.nasa.gov/"


def parse_args(argv=None):
    """Parse the command line arguments."""
    current_year = datetime.today().year
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        help=f"Define years between 2018 and {current_year - 1}",
    )
    _ = parser.add_argument(
        "-a", "--authenticate", nargs=2, help=f"Create an account at {IMERG_URL}"
    )
    return parser.parse_args(argv)


def resolve_years(years_arg):
    """Turn the raw --years argument into a validated list of years."""
    current_year = datetime.today().year

    if years_arg is None:
        years = [current_year - 1]
        print(f"No --years defined, continueing with {years}.")
        return years

    years = []
    for year_str in years_arg:
        year = int(year_str)
        if year < 2018:
            print(f"Skipping {year}, earliest allowed year is 2018.")
            continue
        if year >= current_year:
            print(f"Skipping {year}, latest allowed year is {current_year - 1}.")
            continue
        years.append(year)
    return years


def resolve_auth(authenticate_arg):
    """Turn the raw --authenticate argument into a (username, password) tuple."""
    if authenticate_arg is None:
        print(
            f"Please specify a username and password (create an account here "
            f"`{IMERG_URL}`) to download IMERG data. You can also specify it using "
            f"--authenticate."
        )
        un = input("Username: ")
        pw = getpass("Password: ")
        return (un, pw)
    return tuple([str(x) for x in authenticate_arg])


def main(argv=None):
    args = parse_args(argv)
    years = resolve_years(args.years)
    auth = resolve_auth(args.authenticate)
    pipeline.run(years, auth)


if __name__ == "__main__":
    main()
