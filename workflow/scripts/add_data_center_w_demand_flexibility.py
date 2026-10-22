"""Add Senga-style data center demand after add_sectors and before optimization.

Demand is constant within each planning year. CSV hourly_mwh is numerically MW
for a one-hour interval. Zone totals are split equally across AC buses in that
reeds_zone (never duplicated). Only zones represented in the network are used.
Missing zone/year rows default to zero. No interpolation or year fallback.
The custom base-demand builder already uses no_data_center parquet files.

Senga et al. (2026), DOI 10.1016/j.isci.2026.116497, Eqs. 1-5:
Y[t]=Y[t-1]+D[t]-S[t]; future h-hour S >= Y; future h-hour D >= -Y;
D <= s*L; S <= C-L+D. Store e=-Y; discharger=D and charger=S.
Capacity C=L/utilization varies with planning year. No flexibility cost.
Full-period solves use cyclic shifting balances and deadlines. Daily rolling
solves instead start/end at zero and truncate deadlines at midnight.
This is not a reproduction of the full GenX study.

The build section remains linear. The two reusable helpers handle demand mapping
and solve-time constraints. solve_network.extra_functionality calls the latter
on every model creation; NetCDF alone cannot retain custom constraints.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def read_data_center_demand(n, path, scenario):
    """Return MW by snapshot and AC bus, conserving each zone/year total."""
    if not isinstance(n.snapshots, pd.MultiIndex) or n.snapshots.nlevels != 2:
        raise ValueError("Data centers require (investment period, timestamp) snapshots")
    if n.snapshots.empty or n.snapshots.has_duplicates:
        raise ValueError("Require nonempty, unique snapshots")
    if "reeds_zone" not in n.buses:
        raise ValueError("Network buses need reeds_zone metadata")
    buses = n.buses.loc[n.buses.carrier.eq("AC"), "reeds_zone"]
    if buses.empty or buses.isna().any() or buses.isin(["", "na"]).any():
        raise ValueError("Every AC bus must map to one valid reeds_zone")
    frame = pd.read_csv(path)
    required = {"scenario", "reeds_zone", "year", "hourly_mwh"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Demand table requires {sorted(required)}")
    frame = frame.loc[frame.scenario.eq(scenario)].copy()
    if frame.empty:
        raise ValueError(f"No data center demand for scenario {scenario!r}")
    frame["year"] = pd.to_numeric(frame.year, errors="raise")
    frame["hourly_mwh"] = pd.to_numeric(frame.hourly_mwh, errors="raise")
    if frame.duplicated(["reeds_zone", "year"]).any():
        raise ValueError("Duplicate scenario/zone/year data center demand")
    if not np.isfinite(frame.hourly_mwh).all() or (frame.hourly_mwh < 0).any():
        raise ValueError("Demand must be finite and nonnegative")
    table = frame.set_index(["reeds_zone", "year"]).hourly_mwh
    periods = n.snapshots.get_level_values(0)
    counts = buses.value_counts()
    demand = pd.DataFrame(0.0, index=n.snapshots, columns=buses.index)
    for year in periods.unique():
        keys = pd.MultiIndex.from_tuples([(zone, int(year)) for zone in counts.index])
        missing = keys.difference(table.index)
        if len(missing):
            logger.info("Defaulting missing zone/year demand to zero: %s", missing.tolist())
        zone_demand = table.reindex(keys, fill_value=0.0)
        zone_demand.index = counts.index
        bus_demand = buses.map(zone_demand) / buses.map(counts)
        demand.loc[periods == year, :] = bus_demand.to_numpy()
        if not np.isclose(bus_demand.sum(), zone_demand.reindex(counts.index).sum()):
            raise ValueError("Zone demand allocation failed conservation check")
        logger.info("Data centers %s, %s: %.3f MW", scenario, year, bus_demand.sum())
    return demand.loc[:, demand.max() > 0]


def daily_snapshot_groups(snapshots):
    """Validate complete consecutive hourly days within each investment period."""
    if not isinstance(snapshots, pd.MultiIndex) or snapshots.nlevels != 2 or snapshots.empty or snapshots.has_duplicates:
        raise ValueError("Daily data center flexibility requires unique (period, timestamp) snapshots")
    periods = snapshots.get_level_values(0)
    times = pd.DatetimeIndex(snapshots.get_level_values(1))
    groups = []
    for year in periods.unique():
        positions = np.flatnonzero(periods == year)
        dates = times[positions]
        if (not (np.diff(positions) == 1).all()
                or len(dates) % 24
                or dates[0] != dates[0].normalize()
                or not (dates.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()):
            raise ValueError("Daily data center flexibility requires complete consecutive 24-hour days starting at midnight")
        groups.extend(positions[offset:offset + 24] for offset in range(0, len(positions), 24))
    return groups


def add_data_center_constraints(n, snapshots):
    """Apply Eqs. 2-3 to the current solve, including myopic period solves."""
    settings = n.meta.get("data_center_flexibility", {})
    if not settings.get("add_dr", False):
        return
    h = int(settings["shifting_horizon_hours"])
    snapshots = pd.MultiIndex.from_tuples(list(snapshots), names=n.snapshots.names)
    # prepare_network disables cyclic shifting only for rolling horizons.
    daily = not n.stores.loc[settings["shift_buses"], "e_cyclic"].all()
    periods = snapshots.get_level_values(0)
    groups = daily_snapshot_groups(snapshots) if daily else [
        np.flatnonzero(periods == year) for year in periods.unique()
    ]
    if not daily:
        for year in periods.unique():
            if not snapshots[periods == year].equals(n.snapshots[n.snapshots.get_level_values(0) == year]):
                raise ValueError("Cyclic data center flexibility requires complete investment periods")
    for node_number, shift_bus in enumerate(settings["shift_buses"]):
        S = n.model["Link-p"].sel(Link=f"{shift_bus}-charger")
        D = n.model["Link-p"].sel(Link=f"{shift_bus}-discharger")
        e = n.model["Store-e"].sel(Store=shift_bus)
        for period_number, positions in enumerate(groups):
            selected = positions[:-1] if daily else positions
            for offset, position in enumerate(selected):
                if daily:
                    future = positions[offset + 1:offset + 1 + h]
                else:
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
    utilization = float(cfg.get("utilization", 0.8))
    fraction = float(cfg.get("flexible_fraction", 0.2))
    h = cfg.get("shifting_horizon_hours", 4)
    if not np.isfinite([utilization, fraction]).all() or not 0 < utilization <= 1 or not 0 <= fraction <= 1:
        raise ValueError("Require 0 < utilization <= 1 and 0 <= flexible_fraction <= 1")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("shifting_horizon_hours must be a positive integer")
    add_dr = bool(cfg.get("add_dr", True)) and fraction > 0
    periods = n.snapshots.get_level_values(0)
    groups = [np.flatnonzero(periods == year) for year in periods.unique()]
    if add_dr:
        times = n.snapshots.get_level_values(1)
        for positions in groups:
            if not (np.diff(positions) == 1).all() or not (times[positions].to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
                raise ValueError("Data center flexibility requires contiguous hourly periods")
        if not np.allclose(n.snapshot_weightings.stores, 1):
            raise ValueError("Data center flexibility requires one-hour Store weights")
    n.add("Carrier", "dc-elec", nice_name="Data center electricity")
    shift_buses = []
    for bus in demand.columns:
        sector_bus = f"{bus} dc-elec"
        L = demand[bus]
        C = L / utilization
        capacity = float(C.max())
        n.add("Bus", sector_bus, carrier="dc-elec", v_nom=1,
              x=n.buses.at[bus, "x"], y=n.buses.at[bus, "y"])
        for field in ("reeds_zone", "reeds_ba", "country", "STATE", "STATE_NAME", "interconnect"):
            if field in n.buses:
                n.buses.at[sector_bus, field] = n.buses.at[bus, field]
        n.add("Link", f"{sector_bus} supply", bus0=bus, bus1=sector_bus,
              carrier="dc-elec", efficiency=1, p_nom=capacity,
              p_nom_extendable=False, p_min_pu=0, p_max_pu=C / capacity)
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
    # PART 3: persist settings so the solver can reconstruct rolling constraints.
    n.meta["data_center_flexibility"] = {
        "scenario": scenario, "add_dr": add_dr,
        "shifting_horizon_hours": h, "flexible_fraction": fraction,
        "utilization": utilization, "shift_buses": shift_buses,
        "boundary": "cyclic balance and shifting deadlines per investment period",
    }
