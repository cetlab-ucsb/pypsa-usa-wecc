import os
import json
import time
import geopandas as gpd
import pandas as pd

from pvlib.iotools import get_nsrdb_psm4_conus


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_KEY = "FbUa1nQ0brYkkxawMHmgUMNaOCQerC1GQDHgTsLO" #sanzaypoudel4@gmail.com 

# API_KEY = "gzOXM04IDCz4NowUfGKJwbenwlHqpCQAwQM9pgog" #sanjaypoudel@ucsb.edu

EMAIL   = "sanzaypoudel4@gmail.com"

YEAR = 2024

INPUT_FILE = ("../data/breakthrough_network/base_grid/sub.csv")

OUTPUT_DIR = "../data/weather_file_for_distributed_pv_2024"

INTERVAL = 60          # Download hourly AMY data
LEAP_DAY = True

REQUEST_DELAY = 1     # seconds between API requests


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load census tract shapefile
# ---------------------------------------------------------

geom_df = pd.read_csv(INPUT_FILE)

geom_df = geom_df.loc[geom_df['interconnect'] == 'Western']

# geom_df has 'lat' and 'lon' columns.

# ---------------------------------------------------------
# Loop through each row
# ---------------------------------------------------------

for idx, row in geom_df.iterrows():

    geom_id = int(row['sub_id'])

    # -----------------------------------------------------
    # Get centroid latitude and longitude
    # -----------------------------------------------------

    lat = row['lat']
    lon = row['lon']


    print(
        f"Fetching PSM4 for geometry id: {geom_id} "
        f"@ ({lat:.4f}, {lon:.4f})"
    )


    # -----------------------------------------------------
    # Output filenames
    # -----------------------------------------------------

    weather_fp = os.path.join(
        OUTPUT_DIR,
        f"{geom_id}_weather_{YEAR}.csv"
    )

    meta_fp = os.path.join(
        OUTPUT_DIR,
        f"{geom_id}_meta_{YEAR}.json"
    )


    # -----------------------------------------------------
    # Download NSRDB PSM4 AMY data
    # -----------------------------------------------------

    try:

        weather, meta = get_nsrdb_psm4_conus(
            latitude=lat,
            longitude=lon,
            api_key=API_KEY,
            email=EMAIL,
            year=YEAR,
            time_step=INTERVAL,
            leap_day=LEAP_DAY,
        )


        # -------------------------------------------------
        # Save weather data
        # -------------------------------------------------

        weather.to_csv(weather_fp)

        print(
            f"Saved weather data: {weather_fp}"
        )


        # -------------------------------------------------
        # Save metadata
        # -------------------------------------------------

        with open(meta_fp, "w") as f:
            json.dump(
                meta,
                f,
                indent=2,
                default=str
            )

        print(
            f"Saved metadata: {meta_fp}"
        )


    except Exception as e:

        print(
            f"Error for geometry id {geom_id}: {e}"
        )


    # -----------------------------------------------------
    # Rate limit API requests
    # -----------------------------------------------------

    print(
        f"Sleeping {REQUEST_DELAY} seconds..."
    )

    time.sleep(REQUEST_DELAY)


print("All done!")
