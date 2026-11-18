"""Add Senga-style data center demand after add_sectors and before optimization.

Each clustered zone receives its hourly nonnegative dc_total profile in MW.

Senga et al. (2026), DOI 10.1016/j.isci.2026.116497, Eqs. 1-5:
Y[t]=Y[t-1]+D[t]-S[t]; future h-hour S >= Y; future h-hour D >= -Y;
D <= s*L; S <= C-L+D. Store e=-Y; discharger=D and charger=S.
Capacity C=peak demand/utilization is fixed across modeled snapshots. No flexibility cost.
Full-period solves use cyclic shifting balances and deadlines.
This is not a reproduction of the full GenX study.

The build section remains linear. The two reusable helpers handle demand mapping
and solve-time constraints. solve_network.extra_functionality calls the latter
on every model creation; NetCDF alone cannot retain custom constraints.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

logger = logging.getLogger(__name__)


def read_data_center_demand(n, paths, scenario):
    """Align each planning year's hourly zone profiles to its network snapshots."""
    if not isinstance(n.snapshots, pd.MultiIndex) or n.snapshots.nlevels != 2:
        raise ValueError("Data centers require (investment period, timestamp) snapshots")
    if n.snapshots.empty or n.snapshots.has_duplicates:
        raise ValueError("Require nonempty, unique snapshots")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = {Path(path).name: path for path in paths}
    ac_buses = n.buses.index[n.buses.carrier.eq("AC")]
    periods = n.snapshots.get_level_values(0)
    blocks = []
    for year in periods.unique():
        path = files[f"dc_total_{year}.csv"]
        frame = pd.read_csv(path, index_col="hour_utc").sort_index()
        if not frame.index.equals(pd.RangeIndex(8760)):
            raise ValueError(f"{path}: expected each hour_utc from 0 to 8759 once")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("datetime_utc")))
        frame.index = dates.map(lambda stamp: stamp.replace(year=int(year)))
        if frame.index.has_duplicates or not np.isfinite(frame.to_numpy()).all() or (frame < 0).any().any():
            raise ValueError(f"{path}: require unique timestamps and finite nonnegative demand")
        frame.columns = [zone if zone in ac_buses else f"{zone} AC" for zone in frame]
        missing = frame.columns.difference(ac_buses)
        if len(missing):
            raise ValueError(f"{path}: no matching AC buses for {missing.tolist()}")
        snapshots = n.snapshots[periods == year]
        block = frame.loc[snapshots.get_level_values(1)].copy()
        block.index = snapshots
        blocks.append(block)
    demand = pd.concat(blocks).reindex(n.snapshots)
    if demand.isna().any().any():
        raise ValueError(f"Inconsistent zones across data center profiles for {scenario}")
    return demand.loc[:, demand.max() > 0]


def validate_hourly_snapshots(snapshots):
    """Validate contiguous hourly snapshots within each investment period.

    Partial days and non-midnight starts are allowed. Periods must occupy
    contiguous blocks, but need not be temporally adjacent to one another.
    """
    if not isinstance(snapshots, pd.MultiIndex) or snapshots.nlevels != 2 or snapshots.empty or snapshots.has_duplicates:
        raise ValueError("Data center flexibility requires nonempty unique (period, timestamp) snapshots")
    periods = snapshots.get_level_values(0)
    times = pd.DatetimeIndex(snapshots.get_level_values(1))
    if periods.isna().any() or times.isna().any():
        raise ValueError("Data center flexibility requires nonmissing periods and timestamps")
    for year in periods.unique():
        positions = np.flatnonzero(periods == year)
        dates = times[positions]
        if (not (np.diff(positions) == 1).all()
                or not ((dates[1:] == dates[:-1] + pd.Timedelta(hours=1))
                        | ((dates[1:] == dates[:-1] + pd.Timedelta(hours=25))
                           & dates[1:].is_leap_year & (dates[1:].month == 3)
                           & (dates[1:].day == 1) & (dates[1:].hour == 0))).all()):
            raise ValueError("Data center flexibility requires contiguous hourly periods")


def add_data_center_constraints(n, snapshots):
    """Enforce processing capacity and cyclic deadlines in full-period solves."""
    settings = n.meta.get("data_center_flexibility", {})
    if not settings.get("add_dr", False):
        return
    h = int(settings["shifting_horizon_hours"])
    snapshots = pd.MultiIndex.from_tuples(list(snapshots), names=n.snapshots.names)
    demand = n.get_switchable_as_dense("Load", "p_set")
    for number, shift_bus in enumerate(settings["shift_buses"]):
        charger = f"{shift_bus}-charger"
        sector_bus = n.links.at[charger, "bus0"]
        loads = n.loads.index[n.loads.bus.eq(sector_bus)]
        L = demand[loads].sum(axis=1)
        capacity = float(L.max()) / float(settings["utilization"])
        S = n.model["Link-p"].sel(Link=charger)
        D = n.model["Link-p"].sel(Link=f"{shift_bus}-discharger")
        headroom = xr.DataArray(
            (capacity - L.loc[snapshots]).to_numpy(),
            dims=["snapshot"], coords={"snapshot": S.coords["snapshot"]},
        )
        n.model.add_constraints(S - D <= headroom, name=f"dc-processing-capacity-{number}")
    periods = snapshots.get_level_values(0)
    groups = [
        np.flatnonzero(periods == year) for year in periods.unique()
    ]
    for year in periods.unique():
        if not snapshots[periods == year].equals(n.snapshots[n.snapshots.get_level_values(0) == year]):
            raise ValueError("Cyclic data center flexibility requires complete investment periods")
    for node_number, shift_bus in enumerate(settings["shift_buses"]):
        S = n.model["Link-p"].sel(Link=f"{shift_bus}-charger")
        D = n.model["Link-p"].sel(Link=f"{shift_bus}-discharger")
        e = n.model["Store-e"].sel(Store=shift_bus)
        for period_number, positions in enumerate(groups):
            for offset, position in enumerate(positions):
                future = positions[(offset + np.arange(1, h + 1)) % len(positions)]
                label = f"{node_number}-{period_number}-{offset}"
                n.model.add_constraints(
                    S.isel(snapshot=future).sum("snapshot") + e.isel(snapshot=position) >= 0,
                    name=f"dc-delay-{label}",
                )
                n.model.add_constraints(
                    D.isel(snapshot=future).sum("snapshot") - e.isel(snapshot=position) >= 0,
                    name=f"dc-advance-{label}",
                )


def add_data_center_flexibility(n, cfg, demand_path, scenario):
    """Update the network in place before its existing export."""
    if n.meta.get("data_center_flexibility") or n.buses.carrier.eq("dc-elec").any():
        raise ValueError("Data centers already exist; use the pre-data-center network")
    demand = read_data_center_demand(n, demand_path, scenario)
    utilization = float(cfg.get("utilization", 1))
    fraction = float(cfg.get("flexible_fraction", 0.1))
    h = cfg.get("shifting_horizon_hours", 4)
    if not np.isfinite([utilization, fraction]).all() or not 0 < utilization <= 1 or not 0 <= fraction <= 1:
        raise ValueError("Require 0 < utilization <= 1 and 0 <= flexible_fraction <= 1")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("shifting_horizon_hours must be a positive integer")
    add_dr = bool(cfg.get("add_dr", False)) and fraction > 0
    if add_dr:
        validate_hourly_snapshots(n.snapshots)
        if not np.allclose(n.snapshot_weightings.stores, 1):
            raise ValueError("Data center flexibility requires one-hour Store weights")
    n.add("Carrier", "dc-elec", nice_name="Data center electricity")
    shift_buses = []
    for bus in demand.columns:
        sector_bus = f"{bus} dc-elec"
        L = demand[bus]
        C = L.max() / utilization
        n.add("Bus", sector_bus, carrier="dc-elec", v_nom=1,
              x=n.buses.at[bus, "x"], y=n.buses.at[bus, "y"])
        for field in ("reeds_zone", "reeds_ba", "country", "STATE", "STATE_NAME", "interconnect"):
            if field in n.buses:
                n.buses.at[sector_bus, field] = n.buses.at[bus, field]
        n.add("Link", f"{sector_bus} supply", bus0=bus, bus1=sector_bus,
              carrier="dc-elec", efficiency=1, p_nom=3 * C,
              p_nom_extendable=False, p_min_pu=0, p_max_pu=1)
        n.add("Load", sector_bus, bus=sector_bus, carrier="dc-elec", p_set=L)

        # PART 2: signed Store and directional shifting links. e=-Y.
        if not add_dr:
            continue
        shift_bus = f"{sector_bus}-shift"
        shift_buses.append(shift_bus)
        D_max = fraction * L
        S_max = C - L + D_max
        energy = float(h * max(D_max.max(), S_max.max()))
        lower = -h * S_max / energy
        upper = h * D_max / energy
        n.add("Bus", shift_bus, carrier="dc-elec", v_nom=1,
              x=n.buses.at[bus, "x"], y=n.buses.at[bus, "y"])
        n.add("Link", f"{shift_bus}-charger", bus0=sector_bus, bus1=shift_bus,
              carrier="dc-elec", efficiency=1, p_nom=float(S_max.max()),
              p_nom_extendable=False, p_min_pu=0, p_max_pu=S_max / S_max.max())
        n.add("Link", f"{shift_bus}-discharger", bus0=shift_bus, bus1=sector_bus,
              carrier="dc-elec", efficiency=1, p_nom=float(D_max.max()),
              p_nom_extendable=False, p_min_pu=0, p_max_pu=D_max / D_max.max())
        n.add("Store", shift_bus, bus=shift_bus, carrier="dc-elec",
              e_nom=energy, e_nom_extendable=False, e_min_pu=lower, e_max_pu=upper,
              e_initial=0, e_initial_per_period=False, e_cyclic=True,
              e_cyclic_per_period=True, standing_loss=0, marginal_cost_storage=0)
    # PART 3: persist settings so the solver can reconstruct full-period constraints.
    n.meta["data_center_flexibility"] = {
        "scenario": scenario, "add_dr": add_dr,
        "shifting_horizon_hours": h, "flexible_fraction": fraction,
        "utilization": utilization, "shift_buses": shift_buses,
        "boundary": "cyclic balance and shifting deadlines per investment period",
    }
