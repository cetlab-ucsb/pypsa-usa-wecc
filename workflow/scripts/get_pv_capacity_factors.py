import json
from pathlib import Path
import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as pvwatts

WEATHER_DIR = Path('../data/weather_file_for_distributed_pv_2024')
OUTPUT_DIR = Path('../data/capacity_factors_for_distributed_pv_2024')

def load_puma_weather(geom_id: int):
    weather_path = WEATHER_DIR / f"{geom_id}_weather_2024.csv"
    meta_path    = WEATHER_DIR / f"{geom_id}_meta_2024.json"

    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    leap_day = (weather.index.month == 2) & (weather.index.day == 29)
    weather = weather.loc[~leap_day].copy()

    if len(weather) != 8760:
        raise ValueError(f"Expected 8760 weather records, got {len(weather)}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    sr = {
        'solar_resource_data': {
            'year':   weather.index.year.tolist(),
            'month':  weather.index.month.tolist(),
            'day':    weather.index.day.tolist(),
            'hour':   weather.index.hour.tolist(),
            'minute': weather.index.minute.tolist(),
            'lat':    meta['latitude'],
            'lon':    meta['longitude'],
            'tz':     int(meta['Time Zone']),
            'elev':   float(meta['altitude']),
            'dn':     weather['dni'].tolist(),
            'df':     weather['dhi'].tolist(),
            'gh':     weather['ghi'].tolist(),
            'tdry':   weather['temp_air'].tolist(),
            'wspd':   weather['wind_speed'].tolist(),
        }
    }
    return weather, sr


def simulate_pv(puma_id: int):
    weather, sr = load_puma_weather(puma_id)

    system = pvwatts.default("PVWattsSingleOwner")
    system.SolarResource.assign(sr)

    sd = system.SystemDesign
    sd.system_capacity = 1.0   # kW
    sd.tilt            = 20
    sd.azimuth         = 180
    sd.array_type      = 1
    sd.module_type     = 0
    sd.dc_ac_ratio     = 1.24
    sd.inv_eff         = 96.0
    sd.losses          = 14.0

    system.execute()

    pv_cf_hourly_dc = np.array(system.Outputs.dc)/1000
    pv_cf_hourly_ac = np.array(system.Outputs.ac)/1000

    pv_cf_hourly_dc = pd.Series(pv_cf_hourly_dc, index=weather.index, name='dc_power_kw')
    pv_cf_hourly_ac = pd.Series(pv_cf_hourly_ac, index=weather.index, name='ac_power_kw')

    return pv_cf_hourly_dc, pv_cf_hourly_ac

OUTPUT_DIR.mkdir(exist_ok=True)
mapping = pd.read_csv("../data/breakthrough_network/base_grid/sub.csv")
mapping = mapping.loc[mapping['interconnect'] == 'Western']

geom_ids = mapping['sub_id'].to_list()

for geom_id in geom_ids:
    dc_series, ac_series = simulate_pv(geom_id)
    df = pd.concat([dc_series, ac_series], axis=1)
    out_file = OUTPUT_DIR / f"{geom_id}_capacity_factors.csv"
    df.to_csv(out_file)
    print(f"Saved capacity factors for Substation ID: {geom_id} to {out_file}")
    
