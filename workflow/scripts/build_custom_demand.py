"""Build electrical demand from a user-supplied substation-level dataset.

WECC team addition (not upstream PyPSA-USA). Activated when
`electricity: demand: profile: custom` is set in the config. Instead of
building demand from EFS/EIA, this reads pre-built substation-level hourly
demand (one parquet per planning horizon, columns = Breakthrough Energy
sub_ids, 8760 rows of MW indexed by hour_utc 0-8759) and disaggregates each
substation's load to that substation's buses in proportion to bus Pd —
mirroring the convention of WritePopulation in build_demand.py.

Inputs (via snakemake):
    network: elec_base_network.nc  (for bus <-> sub_id mapping and Pd)
    params.custom_demand_dir: folder containing bus_hourly_{year}.parquet
    params.planning_horizons, params.snapshots

Output:
    power_electricity.csv — index: datetime (8760 h per horizon, no Feb 29),
    columns: bus names, values: MW. Identical contract to the native
    build_electrical_demand output consumed by add_demand.py.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging, get_multiindex_snapshots

logger = logging.getLogger(__name__)


def snapshot_template() -> pd.DatetimeIndex:
    """8760 hourly stamps of a non-leap year (matches get_snapshots template)."""
    return pd.date_range("2019-01-01", "2020-01-01", freq="h", inclusive="left")


def sub_to_bus_weights(n: pypsa.Network) -> pd.DataFrame:
    """Return DataFrame(bus, sub_id, weight): Pd share of each bus within its sub.

    Buses at a sub with zero total Pd share the load equally.
    """
    b = n.buses[["sub_id", "Pd"]].copy()
    b["Pd"] = b.Pd.fillna(0)
    b["sub_id"] = b.sub_id.astype(int).astype(str)
    tot = b.groupby("sub_id").Pd.transform("sum")
    cnt = b.groupby("sub_id").Pd.transform("size")
    b["weight"] = (b.Pd / tot).where(tot > 0, 1.0 / cnt)
    return b.reset_index().rename(columns={"Bus": "bus", "index": "bus"})


def disaggregate(sub_demand: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """sub_id-column demand (MW) -> bus-column demand (MW)."""
    known = weights[weights.sub_id.isin(sub_demand.columns)]
    missing = sorted(set(sub_demand.columns) - set(weights.sub_id))
    if missing:
        lost = sub_demand[missing].sum().sum() / 1e6
        logger.warning(
            f"{len(missing)} sub_ids not in network (dropping {lost:.3f} TWh): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}",
        )
    bus_demand = sub_demand[known.sub_id].to_numpy() * known.weight.to_numpy()
    return pd.DataFrame(bus_demand, index=sub_demand.index, columns=known.bus)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_electrical_demand",
            interconnect="western",
            end_use="power",
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.params.planning_horizons
    sns = get_multiindex_snapshots(snakemake.params.snapshots, planning_horizons)

    demand_dir = Path(snakemake.params.custom_demand_dir)
    weights = sub_to_bus_weights(n)

    blocks = []
    for year in planning_horizons:
        f = demand_dir / f"bus_hourly_{year}.parquet"
        df = pd.read_parquet(f)
        df = df.set_index("hour_utc").sort_index()
        assert len(df) == 8760, f"{f}: expected 8760 rows, got {len(df)}"
        df.index = snapshot_template().map(lambda x: x.replace(year=year))
        blocks.append(disaggregate(df, weights))
        logger.info(f"{year}: {blocks[-1].sum().sum() / 1e6:.1f} TWh over {blocks[-1].shape[1]} buses")

    demand = pd.concat(blocks)
    assert len(demand) == len(sns), (
        f"Demand rows ({len(demand)}) != network snapshots ({len(sns)}). "
        "Check snapshots config covers full non-leap years."
    )
    assert not demand.isna().any().any()

    demand.round(4).to_csv(snakemake.output.elec_demand, index=True)
    logger.info(f"Custom demand written to {snakemake.output.elec_demand}")
