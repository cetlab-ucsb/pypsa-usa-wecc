import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

MAX_PARALLEL_SCENARIOS = 4

script_dir = Path(__file__).resolve().parent
results_dir = script_dir.parent / "results"

# Scenario folders to process.
scenarios = [
    'PCM_2032_GROWTH_HIGH_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_GROWTH_HIGH_DC_NO_FLEX_REFERENCE_2026_11_17',
    'PCM_2032_GROWTH_LOW_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_GROWTH_LOW_DC_NO_FLEX_REFERENCE_2026_11_17',
    'PCM_2032_GROWTH_MEDIUM_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_GROWTH_MEDIUM_DC_NO_FLEX_REFERENCE_2026_11_17',
    'PCM_2032_INDELEC_HIGH_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_INDELEC_HIGH_DC_NO_FLEX_REFERENCE_2026_11_17',
    'PCM_2032_INDELEC_LOW_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_INDELEC_LOW_DC_NO_FLEX_REFERENCE_2026_11_17',
    'PCM_2032_INDELEC_MEDIUM_DC_NO_FLEX_OPTIMISTIC_2026_11_17',
    'PCM_2032_INDELEC_MEDIUM_DC_NO_FLEX_REFERENCE_2026_11_17'
]


def process_scenario(scenario):
    """Process one network with scenario-specific data kept in local scope."""
    network_path = results_dir / scenario / "western/networks/elec_s_c32a_ec_lv1.0__E.nc"
    if not network_path.is_file():
        print(f"Skipping {scenario}: network file not found at {network_path}")
        return
    output_dir = script_dir / "processed_results_raw" / scenario
    n = pypsa.Network(network_path)

    # Generation and load shedding
    # =============================================================================

    grid = n.buses.index[n.buses.carrier.str.upper().eq('AC')]

    gens = n.generators.index

    load_shedding_gens = n.generators.loc[n.generators['carrier'] == 'load'].index
    physical_gens = n.generators.loc[n.generators['carrier'] != 'load'].index

    if n.generators_t.p.lt(0).any().any():
        raise ValueError('Negative generator dispatch found in n.generators_t.p.')

    # Each snapshot is one hour: summing dispatch (MW) gives energy in MWh.
    generation = n.generators_t.p.sum()
    # Group auxiliary buses such as 'p1 dc-elec' under their base bus ('p1').
    base_bus = n.generators['bus'].str.split().str[0]
    grouping_bus_carrier = [base_bus, n.generators['carrier']]
    grouping_carrier = [n.generators['carrier']]

    shedding_bus_carrier = generation.loc[load_shedding_gens].groupby(grouping_bus_carrier).sum() / 1e6
    physical_generation_bus_carrier = generation.loc[physical_gens].groupby(grouping_bus_carrier).sum() / 1e6

    shedding_wecc = generation.loc[load_shedding_gens].groupby(grouping_carrier).sum() / 1e6
    physical_generation_wecc_carrier = generation.loc[physical_gens].groupby(grouping_carrier).sum() / 1e6

    # Total load shedding by zone, converted from TWh to GWh.
    load_shedding_gwh_by_zone = shedding_bus_carrier.groupby(level=0).sum() * 1e3

    shedding_hours = n.generators_t.p[load_shedding_gens] > 0.5

    shedding_hours.columns = shedding_hours.columns.str.removesuffix(' load')

    shedding_hours_by_bus = shedding_hours.sum()

    shedding_hours_across_wecc = shedding_hours.any(axis=1).sum()

    # Available generation and curtailment
    # =============================================================================

    capacity = n.generators.loc[physical_gens, 'p_nom']
    # Include static p_max_pu values as well as time-varying profiles.
    available_power = n.get_switchable_as_dense('Generator', 'p_max_pu')[physical_gens].mul(capacity, axis=1)
    generators = n.generators.loc[physical_gens, ['bus', 'carrier', 'p_nom']].copy()
    conventional = generators.index[generators.carrier.isin(n.meta['electricity']['conventional_carriers'])]
    # The network builder applies the winter rating to all months outside June-August.
    summer = pd.DatetimeIndex(n.snapshots.get_level_values(-1)).month.isin([6, 7, 8])
    storage = n.storage_units
    storage_capacity = storage[['bus', 'carrier', 'p_nom']].assign(storage_energy_capacity_mwh=storage.p_nom * storage.max_hours)

    for column, snapshots in [('summer_capacity_mw', summer), ('winter_capacity_mw', ~summer)]:
        generators[column] = capacity
        storage_capacity[column] = storage.p_nom
        ratings = available_power.loc[snapshots, conventional]
        if len(conventional) and (ratings.empty or ratings.isna().any().any() or ratings.nunique().gt(1).any()):
            raise ValueError(f'Cannot infer a single {column} for each conventional generator.')
        generators.loc[conventional, column] = ratings.max()
    installed_capacity = pd.concat([generators, storage_capacity,])
    installed_capacity['zone'] = installed_capacity['bus'].str.split().str[0]
    installed_capacity = installed_capacity.groupby(['zone', 'carrier'])[
        ['p_nom', 'summer_capacity_mw', 'winter_capacity_mw', 'storage_energy_capacity_mwh']
    ].sum(min_count=1).rename(columns={'p_nom': 'installed_capacity_mw'})
    installed_capacity_by_technology = installed_capacity.groupby(level='carrier').sum()
    max_generation = available_power.sum()
    max_generation_wecc_carrier = max_generation.loc[physical_gens].groupby(grouping_carrier).sum() / 1e6

    # Clip per snapshot so excess dispatch cannot cancel curtailed energy elsewhere.
    # For thermal carriers this measures unused available energy, not renewable curtailment.
    curtailment = (available_power - n.generators_t.p[physical_gens]).clip(lower=0).sum()
    curtailment_wecc_carrier = curtailment.groupby(grouping_carrier).sum() / 1e6

    # Generator utilization
    # =============================================================================

    # Generator utilization (fraction), keeping only generators with positive generation.
    active_generation = generation.loc[physical_gens]
    active_generation = active_generation[active_generation.gt(0)]
    utilization_generator = active_generation.div(
        max_generation.reindex(active_generation.index).where(lambda values: values.gt(0))
    )
    # Simple average of generator utilization within each carrier across WECC.
    utilization_wecc_carrier = utilization_generator.groupby(grouping_carrier).mean()

    # Emissions
    # =============================================================================

    # Carrier factors are tonnes CO2 per MWh of fuel; map them to generator names.
    emission_factors = n.generators.loc[physical_gens, 'carrier'].map(n.carriers.co2_emissions)
    if not np.isfinite(emission_factors).all():
        raise ValueError('Missing or non-finite carrier CO2 emission factors.')

    # Zero-emission generators do not require a fuel conversion efficiency.
    emitting_gens = emission_factors.index[emission_factors.ne(0)]
    efficiency = n.get_switchable_as_dense('Generator', 'efficiency').reindex(
        index=n.generators_t.p.index, columns=emitting_gens
    )
    if not np.isfinite(efficiency.to_numpy()).all() or efficiency.le(0).any().any():
        raise ValueError('Emitting generators must have finite, positive efficiency.')

    # Hourly dispatch / efficiency gives fuel energy in MWh per snapshot.
    fuel_use = n.generators_t.p[emitting_gens].div(efficiency)
    emissions_generator = pd.Series(0.0, index=physical_gens)
    emissions_generator.loc[emitting_gens] = fuel_use.mul(
        emission_factors.loc[emitting_gens], axis=1
    ).sum()

    # Emissions totals in million tonnes CO2 (MtCO2).
    emissions_bus_carrier = emissions_generator.groupby(grouping_bus_carrier).sum() / 1e6
    emissions_wecc_carrier = emissions_generator.groupby(grouping_carrier).sum() / 1e6
    emissions_wecc = emissions_generator.sum() / 1e6

    # Emission intensity
    # =============================================================================

    # Generation-based intensity in kg CO2/MWh: tonnes / MWh * 1000.
    # Use total emissions / total generation, including zero-emission generation.
    # Load shedding is excluded; zero generation gives undefined (NaN) intensity.
    generation_by_bus = generation.loc[physical_gens].groupby(base_bus).sum()
    emissions_by_bus = emissions_generator.groupby(base_bus).sum()
    emission_intensity_by_bus = emissions_by_bus.div(
        generation_by_bus.where(generation_by_bus.gt(0))
    ) * 1000
    total_generation_mwh = generation.loc[physical_gens].sum()
    emission_intensity_wecc = (
        emissions_generator.sum() / total_generation_mwh * 1000
        if total_generation_mwh > 0 else float('nan')
    )

    # Demand
    # =============================================================================
    non_data_center_load = n.loads_t.p.loc[:, n.loads_t.p.columns.str.endswith(' AC')].copy()
    non_data_center_load.columns = non_data_center_load.columns.str.removesuffix(' AC')

    original_data_center_load = n.loads_t.p.loc[:, n.loads_t.p.columns.str.endswith(' dc-elec')].copy()
    original_data_center_load.columns = original_data_center_load.columns.str.removesuffix(' dc-elec')

    shifted_data_center_load = n.links_t.p0.loc[:, n.links_t.p0.columns.str.endswith(' dc-elec supply')].copy()
    shifted_data_center_load.columns = shifted_data_center_load.columns.str.removesuffix(' dc-elec supply')

    original_total_load = non_data_center_load.add(original_data_center_load, fill_value=0)
    shifted_total_load = non_data_center_load.add(shifted_data_center_load, fill_value=0)

    original_peak_load_by_bus = original_total_load.max()
    shifted_peak_load_by_bus = shifted_total_load.max()

    original_peak_load_across_wecc = original_total_load.sum(axis = 1).max()
    shifted_peak_load_across_wecc = shifted_total_load.sum(axis = 1).max()

    # LMP
    # =============================================================================

    # Subtract load shedding at its bus so the LMP weighting reflects served load.
    load_shedding_by_bus = n.generators_t.p[load_shedding_gens]

    load_shedding_by_bus.columns = n.generators_t.p[load_shedding_gens].columns.str.split().str[0]

    served_energy = shifted_total_load.subtract(load_shedding_by_bus, fill_value=0)

    marginal_prices = n.buses_t.marginal_price[grid]

    demand_weighted_lmp_by_bus = (
        (marginal_prices * served_energy).sum(axis=0)
        / served_energy.sum(axis=0)
    )

    demand_weighted_lmp_across_wecc = (
        (marginal_prices * served_energy).sum(axis=1).sum(axis=0)
        / served_energy.sum(axis=1).sum(axis=0)
    )

    # Export results. Each input network gets its own directory.
    # Summary metric names include units; hourly tables retain period/timestep.
    tables = {
        'results_by_reeds_zone_and_technology': pd.concat([
            installed_capacity,
            pd.DataFrame({
                'generation_twh': physical_generation_bus_carrier,
                'load_shedding_twh': shedding_bus_carrier,
                'emissions_mtco2': emissions_bus_carrier,
            }),
        ], axis=1).rename_axis(index=['zone', 'carrier']),
        'results_by_technology': pd.concat([
            installed_capacity_by_technology,
            pd.DataFrame({
                'generation_twh': physical_generation_wecc_carrier,
                'load_shedding_twh': shedding_wecc,
                'available_generation_twh': max_generation_wecc_carrier,
                'curtailment_twh': curtailment_wecc_carrier,
                'utilization_fraction': utilization_wecc_carrier,
                'emissions_mtco2': emissions_wecc_carrier,
            }),
        ], axis=1).rename_axis('carrier'),
        'results_by_reeds_zone': pd.concat({
            'non_data_center_demand_gwh': non_data_center_load.sum()/1e3,
            'original_data_center_demand_gwh': original_data_center_load.sum()/1e3,
            'shifted_data_center_demand_gwh': shifted_data_center_load.sum()/1e3,
            'original_total_demand_gwh': original_total_load.sum()/1e3,
            'shifted_total_demand_gwh': shifted_total_load.sum()/1e3,
            'served_energy_gwh': served_energy.sum()/1e3,
            'original_peak_demand_mw': original_peak_load_by_bus,
            'shifted_peak_demand_mw': shifted_peak_load_by_bus,
            'load_shedding_hours': shedding_hours_by_bus,
            'load_shedding_gwh': load_shedding_gwh_by_zone,
            'demand_weighted_lmp_usd_per_mwh': demand_weighted_lmp_by_bus,
            'emission_intensity_kgco2_per_mwh': emission_intensity_by_bus,
        }, axis=1).rename_axis('zone'),
        'generation_utilization_and_emissions_by_generator': pd.concat({
            'zone': base_bus,
            'carrier': n.generators.carrier,
            'generation_mwh': generation,
            'available_generation_mwh': max_generation,
            'curtailment_mwh': curtailment,
            'utilization_fraction': utilization_generator,
            'emissions_tco2': emissions_generator,
        }, axis=1).rename_axis('generator'),
    }
    hourly_tables = {
        'hourly_non_data_center_load_by_zone_mw': non_data_center_load,
        'hourly_original_data_center_load_by_zone_mw': original_data_center_load,
        'hourly_shifted_data_center_grid_load_by_zone_mwh': shifted_data_center_load,
        'hourly_original_total_load_by_zone_mwh': original_total_load,
        'hourly_served_energy_by_zone_mwh': served_energy,
        'hourly_lmp_by_zone_usd_per_mwh': marginal_prices,
        'hourly_load_shedding_by_generator_mw': n.generators_t.p[load_shedding_gens],
        'hourly_link_p0_mw': n.links_t.p0,
        'hourly_storage_unit_dispatch_mw': n.storage_units_t.p_dispatch,
        'hourly_storage_unit_charging_mw': n.storage_units_t.p_store,
        'hourly_storage_unit_state_of_charge_mwh': n.storage_units_t.state_of_charge,
        'hourly_store_energy_mwh': n.stores_t.e,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f'{name}.csv')
    hourly_output_dir = output_dir / 'hourly'
    hourly_output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in hourly_tables.items():
        table.to_csv(hourly_output_dir / f'{name}.csv')

    pd.DataFrame([{
        'network_file': str(network_path),
        'generation_twh': total_generation_mwh / 1e6,
        'non_data_center_demand_twh': non_data_center_load.sum().sum()/1e6,
        'original_data_center_demand_twh': original_data_center_load.sum().sum()/1e6,
        'shifted_data_center_demand_twh': shifted_data_center_load.sum().sum()/1e6,
        'original_total_demand_twh': original_total_load.sum().sum()/1e6,
        'shifted_total_demand_twh': shifted_total_load.sum().sum()/1e6,
        'served_energy_twh': served_energy.sum().sum()/1e6,
        'original_peak_demand_gw': original_peak_load_across_wecc/1e3,
        'shifted_peak_demand_gw': shifted_peak_load_across_wecc/1e3,
        'load_shedding_twh': shedding_wecc.sum(),
        'load_shedding_hours': shedding_hours_across_wecc,
        'emissions_mtco2': emissions_wecc,
        'emission_intensity_kgco2_per_mwh': emission_intensity_wecc,
        'demand_weighted_lmp_usd_per_mwh': demand_weighted_lmp_across_wecc,
    }]).T.to_csv(output_dir / 'wecc_overall_results.csv')
    print(f'Saved CSVs to {output_dir}')


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_SCENARIOS) as executor:
        list(executor.map(process_scenario, scenarios))
