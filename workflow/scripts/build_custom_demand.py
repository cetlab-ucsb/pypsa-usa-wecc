"""Build zonal demand from precomputed hourly non-data-center ReEDS zone profiles."""

import logging
from pathlib import Path

import pandas as pd
from _helpers import configure_logging, get_multiindex_snapshots

logger = logging.getLogger(__name__)

# Data location relative to workflow/ (Snakemake's working directory)
DEFAULT_DEMAND_DIR = Path("data/custom_demand/reeds_zone_profiles_2026-11-06")


def resolve_demand_dir(scenario) -> Path:
    """Find the selected scenario under data/custom_demand, without fallback."""
    if not isinstance(scenario, str) or not scenario or scenario in (".", "..") or "/" in scenario or "\\" in scenario:
        raise ValueError("custom_demand_scenario must be a scenario name, not a path")
    demand_dir = DEFAULT_DEMAND_DIR / scenario
    if not demand_dir.is_dir():
        raise FileNotFoundError(f"Custom demand scenario folder not found: {demand_dir}")
    logger.info("Using custom demand folder: %s", demand_dir.resolve())
    return demand_dir


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_electrical_demand",
            interconnect="western",
            end_use="power",
        )
    configure_logging(snakemake)

    planning_horizons = snakemake.params.planning_horizons
    sns = get_multiindex_snapshots(snakemake.params.snapshots, planning_horizons)

    demand_dir = resolve_demand_dir(snakemake.params.custom_demand_scenario)

    blocks = []
    for year in planning_horizons:
        f = demand_dir / f"nondc_{year}.csv"
        df = pd.read_csv(f, index_col="hour_utc", parse_dates=["datetime_utc"]).sort_index()
        if not df.index.equals(pd.RangeIndex(8760)):
            raise ValueError(f"{f}: expected each hour_utc from 0 to 8759 exactly once")
        timestamps = pd.DatetimeIndex(df.pop("datetime_utc")).rename(None)
        df.index = timestamps.map(lambda stamp: stamp.replace(year=year))
        blocks.append(df)
        logger.info(f"{year}: {blocks[-1].sum().sum() / 1e6:.1f} TWh over {blocks[-1].shape[1]} zones")

    demand = pd.concat(blocks)
    assert len(demand) == len(sns), (
        f"Demand rows ({len(demand)}) != network snapshots ({len(sns)}). "
        "Check snapshots config covers full non-leap years."
    )
    assert not demand.isna().any().any()

    demand.to_csv(snakemake.output.elec_demand, index=True)
    logger.info(f"Custom demand written to {snakemake.output.elec_demand}")
