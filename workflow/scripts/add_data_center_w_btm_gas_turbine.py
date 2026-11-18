"""Add fixed-size behind-the-meter gas generators to existing data-center buses.

This stage consumes the data-center network; it does not add demand again.
Generator power is a fraction of facility capacity (baseline demand/utilization).
Costs use the 25th percentile and efficiencies use the arithmetic mean of OCGT generators
at each parent AC node, using assets active in each planning year and preserving
time-varying inputs. Nodes without active OCGT generators use the network-wide
statistics of active OCGT generators. Run before gas Generators are converted to
fuel-supply Links.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
CARRIER = "OCGT"

def _parent_nodes(n, dc_buses):
    """Resolve actual grid connections, independently of bus naming conventions."""
    supply = n.links.loc[
        n.links.bus1.isin(dc_buses)
        & n.links.bus0.isin(n.buses.index[n.buses.carrier.eq("AC")]),
        ["bus0", "bus1"],
    ].drop_duplicates()
    counts = supply.groupby("bus1").size().reindex(dc_buses, fill_value=0)
    if counts.ne(1).any():
        raise ValueError(f"Each data-center bus requires one parent AC node: {list(counts.index[counts.ne(1)])}")
    return supply.set_index("bus1").bus0.reindex(dc_buses)

def _node_parameters(n, nodes, years):
    """Read electrical-output costs and efficiencies from existing OCGT assets."""
    plants = n.generators.loc[n.generators.carrier.eq(CARRIER)]
    if plants.empty:
        raise ValueError("No OCGT Generators in the network; run before "
                         "gas Generators are converted to Links")
    efficiency = n.get_switchable_as_dense("Generator", "efficiency")[plants.index]
    cost = n.get_switchable_as_dense("Generator", "marginal_cost")[plants.index]
    parameters = {}
    for year in years:
        active = n.get_active_assets("Generator", int(year)).reindex(plants.index)
        network_names = plants.index[active]
        if network_names.empty:
            raise ValueError(f"No active OCGT generators in the network in {year}")
        for node in pd.unique(nodes):
            names = plants.index[plants.bus.eq(node) & active]
            if names.empty:
                names = network_names
                logger.info("AC node %s in %s: using network-wide cost 25th percentile and mean efficiency of %d active OCGT generators",
                            node, year, len(names))
            # Validate source values before aggregation, so missing data is not skipped.
            eta = efficiency[names]
            mc = cost[names]
            if (not np.isfinite(eta.to_numpy()).all()
                    or not ((eta > 0) & (eta <= 1)).all().all()
                    or not np.isfinite(mc.to_numpy()).all()):
                raise ValueError("Network OCGT costs must be finite and efficiencies in (0, 1]")
            parameters[node, year] = (eta.mean(axis=1), mc.quantile(0.25, axis=1, interpolation="linear"))
    return parameters


def add_data_center_gas_turbines(n, cfg):
    """Attach fixed gas generators at each facility for each planning year."""
    if not cfg.get("add_btm_gas_plant", False):
        return
    if not cfg.get("enable", False):
        raise ValueError("Data-center btm gas plants require data_centers.enable: true")
    if n.meta.get("data_center_btm_gas") or n.generators.index.astype(str).str.contains(r" btm-gas-\d+$").any():
        raise ValueError("Data-center gas plants already exist in this network")
    if not isinstance(n.snapshots, pd.MultiIndex) or n.snapshots.nlevels != 2 or n.snapshots.empty:
        raise ValueError("Data-center gas plants require (investment period, timestamp) snapshots")

    factor = float(cfg.get("gas_plant_sizing_factor", 0.5))
    utilization = float(cfg.get("utilization", 1))
    if not np.isfinite([factor, utilization]).all():
        raise ValueError("Sizing factor and utilization must be finite")
    if factor < 0:
        raise ValueError("Require gas_plant_sizing_factor >= 0")
    if not (0 < utilization <= 1):
        raise ValueError("utilization must be in (0, 1]")
    dc_meta = n.meta.get("data_center_flexibility")
    if dc_meta is None:
        raise ValueError("Run add_data_center_w_demand_flexibility before adding gas plants")
    if not np.isclose(utilization, dc_meta["utilization"]):
        raise ValueError("Gas plant utilization must match the existing data-center network")

    dc_buses = n.buses.index[n.buses.carrier.eq("dc-elec")]
    dc_loads = n.loads.index[n.loads.bus.isin(dc_buses)]
    demand = n.get_switchable_as_dense("Load", "p_set")[dc_loads]
    demand = demand.T.groupby(n.loads.loc[dc_loads, "bus"]).sum().T
    if not np.isfinite(demand.to_numpy()).all() or (demand < 0).any().any():
        raise ValueError("Data-center demand must be finite and nonnegative")

    periods = n.snapshots.get_level_values(0)
    years = sorted(periods.unique())
    if any(not np.isfinite(float(year)) or float(year) != int(year) for year in years):
        raise ValueError("Investment periods must be integer years")
    active_buses = demand.columns[(demand.max() > 0) & (factor > 0)]
    # Validate all inputs before modifying the network.
    if len(active_buses):
        nodes = _parent_nodes(n, active_buses)
        parameters = _node_parameters(n, nodes, years)
        if CARRIER not in n.carriers.index:
            raise ValueError("Network requires an OCGT carrier with gas emissions configured")
    btm_gas_turbines = []
    for bus in active_buses:
        for year_number, year in enumerate(years):
            active = periods == year
            power = factor * float(demand.loc[active, bus].max()) / utilization
            if power == 0:
                continue
            # Separate fixed-size assets use build_year/lifetime for period activity.
            lifetime = int(years[year_number + 1] - year) if year_number + 1 < len(years) else 1
            efficiency, marginal_cost = parameters[nodes.at[bus], year]
            name = f"{bus} btm-gas-{year}"
            n.add(
                "Generator", name, bus=bus, carrier=CARRIER,
                p_nom=power, p_nom_extendable=False,
                p_min_pu=0, p_max_pu=1,
                efficiency=efficiency,
                build_year=int(year), lifetime=lifetime,
                marginal_cost=marginal_cost,
            )
            btm_gas_turbines.append(name)
            logger.info("BTM gas generator %s: %.3f MW", name, power)

    n.meta["data_center_btm_gas"] = {
        "gas_plant_sizing_factor": factor,
        "utilization": utilization,
        "generators": btm_gas_turbines,
        "sizing_basis": "baseline demand / utilization, separately per investment period",
        "parameter_basis": "25th percentile marginal cost (linear interpolation) and arithmetic mean efficiency of active network OCGT generators by parent AC node and planning year",
        "parameter_fallback": "network-wide 25th percentile marginal cost and arithmetic mean efficiency of active OCGT generators for nodes without active OCGT",
    }

