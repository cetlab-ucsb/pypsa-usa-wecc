"""Add fixed-size behind-the-meter batteries to existing data-center buses.

This stage consumes the data-center network; it does not add demand again.
Battery power is a fraction of facility capacity (baseline demand/utilization).
Energy capacity is power times battery_ep_ratio. Efficiencies are applied once
by the StorageUnit, on its charging and discharging sides respectively.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
CARRIER = "dc-elec"


def add_data_center_batteries(n, cfg):
    """Attach a cyclic battery at each facility, sized for each planning year."""
    if not cfg.get("add_btm_battery", False):
        return
    if not cfg.get("enable", False):
        raise ValueError("Data-center batteries require data_centers.enable: true")
    if n.meta.get("data_center_btm_battery") or n.storage_units.index.astype(str).str.contains(r" btm-battery-\d+$").any():
        raise ValueError("Data-center batteries already exist in this network")
    if not isinstance(n.snapshots, pd.MultiIndex) or n.snapshots.nlevels != 2 or n.snapshots.empty:
        raise ValueError("Data-center batteries require (investment period, timestamp) snapshots")

    factor = float(cfg.get("battery_power_rating_sizing_factor", 0.2))
    duration = float(cfg.get("battery_ep_ratio", 4.0))
    round_trip_efficiency = float(cfg.get("battery_round_trip_efficiency", 0.85))
    utilization = float(cfg.get("utilization", 0.9))
    if not np.isfinite([factor, duration, round_trip_efficiency, utilization]).all():
        raise ValueError("Battery sizing, efficiencies, and utilization must be finite")
    if factor < 0 or duration <= 0:
        raise ValueError("Require battery_power_rating_sizing_factor >= 0 and battery_ep_ratio > 0")
    if not (0 < round_trip_efficiency <= 1 and 0 < utilization <= 1):
        raise ValueError("battery_round_trip_efficiency and utilization must be in (0, 1]")
    eta_charge = eta_discharge = round_trip_efficiency ** 0.5
    dc_meta = n.meta.get("data_center_flexibility")
    if dc_meta is None:
        raise ValueError("Run add_data_center_w_demand_flexibility before adding batteries")
    if not np.isclose(utilization, dc_meta["utilization"]):
        raise ValueError("Battery utilization must match the existing data-center network")

    dc_buses = n.buses.index[n.buses.carrier.eq("dc-elec")]
    dc_loads = n.loads.index[n.loads.bus.isin(dc_buses)]
    demand = n.get_switchable_as_dense("Load", "p_set")[dc_loads]
    demand = demand.T.groupby(n.loads.loc[dc_loads, "bus"]).sum().T
    if not np.isfinite(demand.to_numpy()).all() or (demand < 0).any().any():
        raise ValueError("Data-center demand must be finite and nonnegative")

    periods = n.snapshots.get_level_values(0)
    years = sorted(periods.unique())
    if CARRIER not in n.carriers.index:
        n.add("Carrier", CARRIER, nice_name="Data center electricity", co2_emissions=0)
    batteries = []
    for bus in demand.columns:
        for year_number, year in enumerate(years):
            active = periods == year
            power = factor * float(demand.loc[active, bus].max()) / utilization
            if power == 0:
                continue
            # Separate fixed-size assets use build_year/lifetime for period activity.
            lifetime = int(years[year_number + 1] - year) if year_number + 1 < len(years) else 1
            name = f"{bus} btm-battery-{year}"
            n.add(
                "StorageUnit", name, bus=bus, carrier=CARRIER,
                p_nom=power, p_nom_extendable=False, max_hours=duration,
                p_min_pu=-1, p_max_pu=1,
                efficiency_store=eta_charge, efficiency_dispatch=eta_discharge,
                cyclic_state_of_charge=True, cyclic_state_of_charge_per_period=True,
                build_year=int(year), lifetime=lifetime,
                standing_loss=0, marginal_cost=0, marginal_cost_storage=0,
            )
            batteries.append(name)
            logger.info("BTM battery %s: %.3f MW, %.3f MWh", name, power, power * duration)

    n.meta["data_center_btm_battery"] = {
        "battery_power_rating_sizing_factor": factor,
        "battery_ep_ratio": duration,
        "battery_round_trip_efficiency": round_trip_efficiency,
        "utilization": utilization,
        "storage_units": batteries,
        "sizing_basis": "baseline demand / utilization, separately per investment period",
        "boundary": "cyclic state of charge per investment period",
    }

