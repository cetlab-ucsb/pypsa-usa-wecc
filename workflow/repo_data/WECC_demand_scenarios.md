# WECC custom demand scenarios (dataset 2026-11-06)

How the WECC team's demand datasets plug into this fork, which scenarios exist,
and where the numbers come from. The datasets themselves are built in the
`wecc/` analysis workspace (scripts named below), not in this repo; only the
configs and this note live here.

## Running a scenario

1. Pick a scenario from the table below and copy its config into place:
   `cp workflow/repo_data/config/config.wecc.<scenario>.yaml workflow/config/`
   (`workflow/config/` is git-ignored, so the templates live in `repo_data/config/`).
   The configs set `planning_horizons: [2024, 2032]`.
2. Put that scenario's `bus_hourly_2024.parquet` and `bus_hourly_2032.parquet`
   where the config points (`custom_demand_dir: ../../demand/scenarios/<scenario>`,
   relative to `workflow/`) **or** into `workflow/data/custom_demand/` — the
   fallback `build_custom_demand.py` uses when the configured folder is missing.
3. Run the workflow as usual with `--configfile workflow/config/config.wecc.<scenario>.yaml`.

Each parquet is 8760 rows × 4,772 Breakthrough-Energy substations, MW, float32,
`hour_utc` column, 2024 calendar with Feb 29 dropped, ~210 MB. They are too
large for git; get them from the shared drive (folder `demand/scenarios/`,
dataset date 2026-11-06). `bus_hourly_2024.parquet` is identical in every
scenario: it is the EIA-930-anchored empirical base year with no increment layers.

## Scenarios

Every scenario-year = buildings + industrial + Phase-1 industrial baseline +
road/HDV/rail + data-center increment + industrial process-heat electrification
layer. In this set the first four are the same in every scenario (EFS Medium
for buildings and transport, EFS Reference for industrial by design); the
scenarios vary the process-heat adoption tier and the data-center treatment:

| scenario | process-heat electrification (e-boiler average case) | data-center increment (EPRI Medium) |
|---|---|---|
| `medium_elec_low` | low adoption: cheapest 50% of facilities by 2050 | full |
| `medium_elec_low_btmnet` | low adoption | state BTM share removed |
| `medium_elec_medium` | medium adoption: 100% by 2050 | full |
| `medium_elec_medium_btmnet` | medium adoption | state BTM share removed |
| `medium_elec_high` | high adoption: 100% by 2040 | full |
| `medium_elec_high_btmnet` | high adoption | state BTM share removed |

"State BTM share removed" = each state's EPRI increment multiplied by
(1 − BTM generator nameplate ÷ full data-center capacity, from the lab + BTM
datasets, 2026-10-29): AZ 9%, CA 12%, NV 15%, UT 36%, WY 41%, MT 41%, NM 45%,
CO/ID/OR/WA 0%. The un-suffixed scenarios carry the full increment, grid-served
and behind-the-meter alike.

Annual energy by sector, TWh, and WECC coincident peak, GW (from
`demand/scenarios/scenario_annual_summary.csv`, built 2026-11-06):

| scenario | year | total | bldg | ind | phase1 | road | hdv | rail | dc | heat | peak GW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all six | 2024 | 759.1 | 579.3 | 157.9 | 10.6 | 10.3 | 0 | 1.0 | 0 | 0 | 140.8 |
| `medium_elec_low` | 2032 | 953.3 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 82.0 | 0.003 | 180.9 |
| `medium_elec_low_btmnet` | 2032 | 944.2 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 72.8 | 0.003 | 179.8 |
| `medium_elec_medium` | 2032 | 954.4 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 82.0 | 1.04 | 181.1 |
| `medium_elec_medium_btmnet` | 2032 | 945.2 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 72.8 | 1.04 | 179.9 |
| `medium_elec_high` | 2032 | 958.6 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 82.0 | 5.24 | 181.6 |
| `medium_elec_high_btmnet` | 2032 | 949.4 | 615.3 | 168.4 | 10.6 | 72.5 | 3.3 | 1.3 | 72.8 | 5.24 | 180.4 |

The heat layer is small in 2032 because every adoption S-curve starts in 2027
and is stretched to 2040 or 2050; it is 22.8 TWh (medium) by 2050.

## How each 2032 component was made

| component | 2032 source |
|---|---|
| bldg, ind, transport (road/hdv/rail) | linear interpolation of the 2030 and 2040 component files, hour by hour and substation by substation (0.8 × 2030 + 0.2 × 2040) — `wecc/scripts/interpolate_components_2032.py` |
| phase1 (Phase-1 facility baseline) | fixed at the 2024 level in every year |
| dc | native 2032: EPRI 2026 state annual energy extended from 2030 to 2032 at each state's own 2029→2030 growth rate, minus 2024, eastern-interconnect share carved out, allocated to substations by the facility database; hourly shape from five technology buckets (conventional/air, AI training/air, training/liquid, inference/air, inference/liquid) = LBNL Shape Maker IT profiles × UCSB lab climate-zone PUE profiles, weighted by the LBNL 2025 Update / Xiao et al. 2025 composition of post-2024 growth — `wecc/scripts/dc_composition/build_dc_load_profiles.py`, `wecc/scripts/build_dc_components_profiles.py` |
| heat | exact at 2032 from the facility adoption dates (a facility contributes its full simulated 2024-weather-year load once its adoption date ≤ 31 Dec 2032) — `wecc/scripts/build_adoption_scenarios.py`, `build_phase1_electrification_adoption_scenarios.py` |

All components sit on the 2024 UTC calendar (Feb 29 dropped). Conservation
`sum(bus_hourly) == sum(bus_demand)` is checked on every scenario-year by
`wecc/scripts/combine_demand_scenarios.py`, which sums the components.

## Retired

The 2026-08-19 scenario set (`baseline`, `<tier>_elec_best/_worst`, years
2024/2030/2040/2050, single 2027–2035 heat S-curve) and last night's
`_btmnet` twins of it are retired; their folders are in
`wecc/demand/scenarios/_retired_2026-11-06/` and their config templates were
removed from `repo_data/config/`.

Open items on the data-center layer (to be settled with the PI): whether the
lab PUE profiles are full PUE or cooling-only; whether to reconcile the lab's
PUE with EPRI's fleet PUE; the 2031–32 extrapolation rule.
Also open: the BTM netting basis — the state-level shares remove ~11% of the
EPRI Medium increment WECC-wide, versus a 27% capacity-weighted WECC share,
because the high-share states (UT/WY/NM/MT) carry little EPRI energy.

## ReEDS-zone deliverable (2026-11-06, later)

For zone-level runs the demand is also delivered as ReEDS-zone CSVs
(`wecc/demand/reeds_zone_profiles/`, README there): per scenario and year,
`nondc`, `dc_total` and `dc_inference` / `dc_training` / `dc_conventional`,
8760 × 32 zones (p30 merged into p28), MW, 2024 UTC calendar; plus a BTM
estimate per zone. Scenario folders follow the colleague's 2032 table:
`indelec_{low,medium,high}` (EFS Medium + DC Medium, heat adoption varies)
and `growth_{low,medium,high}` (aligned tiers). The substation-level
`medium_elec_*` parquets above remain available but are not needed for
zone-level runs.
