from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import geopandas as gpd


def select_census_generators(yearly_generators, report_date=pd.Timestamp("2025-01-01")):
    """Represent builds and retirements during the census year as dated events."""
    snapshot = yearly_generators.loc[
        pd.to_datetime(yearly_generators["report_date"]).eq(report_date)
    ].drop_duplicates(["plant_id_eia", "generator_id"], keep="last").copy()
    existing = snapshot["operational_status"].eq("existing")
    retirement = pd.to_datetime(snapshot["generator_retirement_date"], errors="coerce")
    retired = snapshot["operational_status"].eq("retired") & retirement.dt.year.eq(report_date.year)
    fleet = snapshot.loc[existing | retired].copy()
    fleet.loc[retired, "planned_generator_retirement_date"] = retirement.loc[retired]

    operating = pd.to_datetime(fleet["generator_operating_date"], errors="coerce")
    built = operating.dt.year.eq(report_date.year)
    fleet["operational_status"] = fleet["operational_status"].astype("string")
    fleet.loc[built, "operational_status"] = "proposed"
    fleet.loc[built, "current_planned_generator_operating_date"] = operating.loc[built]
    return fleet


def estimate_component_capacity(project, target_type, reported_capacities, capacity_peers):
    """Return MW and donor metadata using at least three positive reported ratios."""
    estimates = []
    for anchor in (1, 2, 3):
        anchor_type = project[f'type_{anchor}']
        anchor_mw = pd.to_numeric(project[f'mw_{anchor}'], errors='coerce')
        if (pd.isna(anchor_type) or pd.isna(anchor_mw) or anchor_mw <= 0
                or anchor_type == target_type
                or target_type not in reported_capacities
                or anchor_type not in reported_capacities):
            continue
        ratios = reported_capacities[target_type] / reported_capacities[anchor_type]
        ratios = ratios.loc[
            reported_capacities[target_type].gt(0)
            & reported_capacities[anchor_type].gt(0) & np.isfinite(ratios)
        ]
        peers = capacity_peers.loc[ratios.index]
        same_region = peers['region'].eq(project['region'])
        same_hybrid = peers['type_clean'].eq(project['type_clean'])
        groups = [
            ('same hybrid/region', same_hybrid & same_region),
            ('same hybrid/WECC', same_hybrid),
            ('component pair/region', same_region),
            ('component pair/WECC', pd.Series(True, index=peers.index)),
        ]
        for priority, (source, mask) in enumerate(groups):
            donors = ratios.loc[mask]
            if len(donors) >= 3:
                ratio = donors.median()
                estimates.append((priority, -len(donors), anchor,
                                  anchor_mw * ratio, source, ratio, len(donors), anchor_type))
                break
    if not estimates:
        return None
    # Prefer specificity, then donor count, then component order.
    return min(estimates)


def plot_capacity_comparison(eia, ads, title, *, eia_label='EIA', ads_label='ADS', note=None):
    fleets = {
        'EIA': eia.rename(columns={'state': 'State', 'comparison_capacity_mw': 'mw'}),
        'ADS': ads.rename(columns={'PSSEMaxCap(MW)': 'mw'}),
    }
    colors = {'EIA': '#34739C', 'ADS': '#EC912F'}
    panels = {}
    for key in ['type', 'State']:
        totals = pd.DataFrame({
            source: fleet.groupby(key, dropna=False)['mw'].sum() / 1000
            for source, fleet in fleets.items()
        }).fillna(0)
        panels[key] = totals.loc[totals.max(axis=1).sort_values(ascending=False).index]

    row_height = 66
    top = 192
    height = top + row_height * max(map(len, panels.values())) + 95
    chart = Image.new('RGB', (2000, height), 'white')
    draw = ImageDraw.Draw(chart)


    def label(x, y, message, size=22, anchor='lm'):
        draw.text((x, y), message, font=ImageFont.load_default(size=size),
                  fill='#263441', anchor=anchor)


    label(1000, 42, title, 40, 'mm')
    for source, x in [('ADS', 560), ('EIA', 1020)]:
        draw.rectangle((x, 87, x + 26, 106), fill=colors[source])
        total_gw = fleets[source]['mw'].sum() / 1000
        name = eia_label if source == 'EIA' else ads_label
        label(x + 40, 97, f'{name}   {total_gw:,.2f} GW')

    for panel, (key, totals) in enumerate(panels.items()):
        origin = panel * 980 + 202
        label(origin, 153, 'By technology type' if key == 'type' else 'By state', 30)
        peak = totals.max().max()
        scale = 660 / peak if peak > 0 else 0
        for row, (category, values) in enumerate(totals.iterrows()):
            y = top + row * row_height
            label(origin - 16, y + 19, str(category), anchor='rm')
            for offset, source in [(0, 'ADS'), (22, 'EIA')]:
                value = values[source]
                end = origin + value * scale
                if value > 0:
                    draw.rectangle((origin, y + offset, end, y + offset + 16), fill=colors[source])
                label(end + 8, y + offset + 8, f'{value:.2f}', 18)
        label(origin, top + len(totals) * row_height + 6, 'Capacity (GW)', 20)

    label(45, height - 30, note or 'EIA uses summer capacity with nameplate capacity as a fallback. '
          'ADS uses PSSE maximum capacity.', 20)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(chart)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def plot_ads_additions(additions, selected, selected_mw):
    additional_mw = additions['PSSEMaxCap(MW)'].sum()
    tier = additions['DevStatus'].str.extract(r'\b(T[123])\b', expand=False).fillna('Other projects')
    tier.loc[additions['DevStatus'].eq('Existing')] = 'Existing'
    tier_colors = {'T1': '#34739C', 'T2': '#EC912F', 'T3': '#459580',
                   'Other projects': '#999999', 'Existing': '#865CA0'}
    tier_labels = {'T1': 'T1 - Under Construction', 'T2': 'T2 - Future Planned',
                   'T3': 'T3 - Future Conceptual', 'Other projects': 'Other Added Projects',
                   'Existing': 'Existing'}
    stage = selected['scenario_phase'].replace({
        'In Progress (unknown study)': 'Study', 'Cluster Study': 'Study',
        'Facility Study': 'Study', 'System Impact Study': 'Study', 'Feasibility Study': 'Study',
    }).fillna('Unreported').rename('Stage')
    stage_colors = {'Construction': '#482878', 'IA Executed': '#156B64',
                    'IA Pending': '#76B7B2', 'Study': '#C85C78',
                    'Not Started': '#EACB71', 'Unreported': '#B9B9B9'}
    selected_by = {
        'type': selected_mw.groupby(stage).sum().T,
        'State': selected_mw.sum(axis=1).groupby([selected['state'], stage]).sum().unstack(fill_value=0),
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f'2026–2032 capacity additions: ADS target {additional_mw / 1000:.2f} GW'
                 f'  |  Selected queue {selected_mw.sum().sum() / 1000:.2f} GW')
    for ax, key, title in zip(axes, ['type', 'State'], ['By fuel type', 'By state']):
        capacity_gw = (
            additions.groupby([key, tier], dropna=False)['PSSEMaxCap(MW)'].sum()
            .unstack(fill_value=0).reindex(columns=tier_colors, fill_value=0) / 1000
        )
        queue_gw = selected_by[key] / 1000
        categories = capacity_gw.index.union(queue_gw.index[queue_gw.sum(axis=1).gt(0)])
        categories = capacity_gw.sum(axis=1).reindex(categories, fill_value=0).sort_values().index
        y = np.arange(len(categories))
        for table, offset, colors, labels in [
            (capacity_gw, 0.21, tier_colors, tier_labels),
            (queue_gw, -0.21, stage_colors, {}),
        ]:
            table = table.reindex(index=categories, fill_value=0)
            left = np.zeros(len(categories))
            for status, color in colors.items():
                if status in table:
                    values = table[status].to_numpy()
                    ax.barh(y + offset, values, height=0.37, left=left, color=color,
                            label=labels.get(status, status))
                    left += values
            for pos, total in zip(y + offset, left):
                ax.annotate(f'{total:.2f}', (total, pos), xytext=(4, 0),
                            textcoords='offset points', va='center', fontsize=8)
        ax.set_yticks(y, labels=categories)
        ax.margins(x=0.15, y=0.03)
        ax.set_title(title)
        ax.set_xlabel('Capacity (GW)')

    handles, labels = axes[0].get_legend_handles_labels()
    split = len(tier_colors)
    for ax, entries, names, heading in [
        (axes[0], handles[:split], labels[:split], 'Upper bar · ADS development status'),
        (axes[1], handles[split:], labels[split:], 'Lower bar · Selected queue stage'),
    ]:
        ax.legend(entries, names, title=heading, loc='upper center',
                  bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)


end_of_2025 = pd.Timestamp('2025-12-31')
end_of_2032 = pd.Timestamp('2032-12-31')

# EIA 2025

workflow_run = "snakemake" in globals()
if workflow_run:
    pudl_release = snakemake.params.pudl_path.rstrip("/")
    ads_file = snakemake.input.ads
    queue_file = snakemake.input.queue
    counties_file = snakemake.input.counties
    fleet_file = Path(snakemake.output.future_fleet)
else:
    workflow_dir = Path(__file__).resolve().parents[1]
    pudl_release = "s3://pudl.catalyst.coop/nightly"
    ads_file = workflow_dir / "repo_data/WECC ADS data/2036 ADS PCM V2LNR Public Data/GeneratorList.csv"
    queue_file = workflow_dir / "repo_data/LBNL_Ix_Queue_Data_File_thru2025.xlsx"
    counties_file = workflow_dir / "data/counties/cb_2020_us_county_500k.shp"
    fleet_file = workflow_dir / "resources/future_fleet.csv"

yearly_generators = pd.read_parquet(
    f'{pudl_release}/out_eia__yearly_generators.parquet',
    dtype_backend='pyarrow',
)
plant_regions = pd.read_parquet(
    f'{pudl_release}/core_eia860__scd_plants.parquet',
    columns=['plant_id_eia', 'report_date', 'nerc_region'],
    dtype_backend='pyarrow',
)
plant_regions['report_date'] = pd.to_datetime(plant_regions['report_date'])

latest_region = (
    plant_regions.dropna(subset=['plant_id_eia', 'nerc_region'])
    .sort_values('report_date', ascending=False, kind='mergesort')
    .drop_duplicates('plant_id_eia', keep='first')
    .set_index('plant_id_eia')['nerc_region']
)

yearly_generators['nerc_region'] = yearly_generators['plant_id_eia'].map(latest_region)

print('Yearly generator records missing NERC region:',
        yearly_generators['nerc_region'].isna().sum())

eia_2025 = select_census_generators(yearly_generators)
eia_2025 = eia_2025.loc[eia_2025['nerc_region'].eq('WECC')].copy()

eia_2025['type'] = eia_2025['fuel_type_code_pudl'].map({
    'gas': 'Gas', 'hydro': 'Hydro', 'solar': 'Solar', 'wind': 'Wind',
    'coal': 'Coal', 'nuclear': 'Nuclear',
}).fillna('Other')
technology_types = eia_2025['technology_description'].map({
    'Geothermal': 'Geothermal', 'Batteries': 'Battery',
    'Hydroelectric Pumped Storage': 'Pumped storage'
})
eia_2025['type'] = technology_types.fillna(eia_2025['type'])

eia_2025['comparison_capacity_mw'] = eia_2025['summer_capacity_mw'].fillna(eia_2025['capacity_mw'])

# Apply commissioning and retirement events through the end of the comparison year.
planned_online = pd.to_datetime(eia_2025['current_planned_generator_operating_date'])
planned_retirement = pd.to_datetime(eia_2025['planned_generator_retirement_date'])
commissioned = eia_2025['operational_status'].eq('existing') | (
    eia_2025['operational_status'].eq('proposed') & planned_online.le(end_of_2025)
)
still_operating = planned_retirement.isna() | planned_retirement.gt(end_of_2025)
eia_year_end = eia_2025.loc[commissioned & still_operating].copy()
eia_year_end['operational_status'] = 'existing'

planned_retirement = planned_retirement.loc[eia_year_end.index]
eia_retired_by_2032 = eia_year_end.loc[
    planned_retirement.gt(end_of_2025) & planned_retirement.le(end_of_2032)
]

print('EIA Capacity by 2025 (GW):', eia_year_end['comparison_capacity_mw'].sum()/1e3)
print('EIA Retired Capacity by 2032 (GW):', eia_retired_by_2032['comparison_capacity_mw'].sum()/1e3)

# ADS 2036

ads = pd.read_csv(ads_file, skiprows=2, encoding='cp1252', low_memory=False)

excluded_dev_status = {
    'Power Flow Active - Canceled in LnR', 'CL - Canceled in LnR',
    'Power Flow Unit - Not in LnR', 'Agg - Disabled',
    'GenDist Accounted', 'Netted',
}

ads = ads.loc[
    ads['State'].notna() & ~ads['State'].isin(['BC', 'AB', 'MX'])
    & ads['ServiceStatus'].eq('#TRUE#')
    & ads['BTM'].eq('#FALSE#')
    & ~ads['SubType'].isin(['IndLoad', 'DC-Tie'])
    & ~ads['DevStatus'].isin(excluded_dev_status)
].copy()

ads['type'] = ads['FuelType'].map({
    'NaturalGas': 'Gas', 'Water': 'Hydro', 'Sun': 'Solar', 'Wind': 'Wind',
    'Coal': 'Coal', 'Uranium': 'Nuclear', 'Geothermal': 'Geothermal',
}).fillna('Other')

# The ADS comparison counts ES-Other alongside batteries.
ads.loc[ads['SubType'].isin(['Battery', 'ES-Other']), 'type'] = 'Battery'
pumped_storage = ads['Technology'].eq('PS-Hydro') | ads['SubType'].eq('PS')
ads.loc[pumped_storage, 'type'] = 'Pumped storage'

for column in ['Commission Date', 'Retirement Date']:
    ads[column] = pd.to_datetime(
        ads[column].astype('string').str.strip().str.strip('#'), format='%Y-%m-%d',
    )

# Retain the ADS fleet in service at the end of 2032.

ads_2025 = ads.loc[ 
    (ads['Commission Date'] <= end_of_2025) &
    (ads['Retirement Date'].isna() | (ads['Retirement Date'] > end_of_2025))
].copy()

ads_2032 = ads.loc[ 
    (ads['Commission Date'] <= end_of_2032) &
    (ads['Retirement Date'].isna() | (ads['Retirement Date'] > end_of_2032))
].copy()

ads_2025_to_2032 = ads_2032.loc[
    ~ads_2032.index.isin(ads_2025.index)
    & ~ads_2032['DevStatus'].eq('Existing')
    ].copy()

gas_capacity_by_technology = (
    ads_2025_to_2032.loc[ads_2025_to_2032['FuelType'].eq('NaturalGas')]
    .groupby(['State', 'SubType'], dropna=False)['PSSEMaxCap(MW)'].sum()
)
gas_capacity_by_state = gas_capacity_by_technology.groupby(level='State').transform('sum')

gas_capacity_share_pct = (
    gas_capacity_by_technology.div(gas_capacity_by_state).mul(100)
    .sort_values(ascending=False).sort_index(level='State', sort_remaining=False)
)
print('ADS natural gas capacity additions by state and technology (%):', gas_capacity_share_pct, sep='\n')

ads_2025_to_2032_battery = ads_2025_to_2032.loc[ads_2025_to_2032['SubType'].isin(['ES-Other', 'Battery'])].copy()

ads_2025_to_2032_battery = ads_2025_to_2032_battery[['Name', 'Long Name', 'PSSEMaxCap(MW)', 'type']]

ads_retired_till_2032 = ads_2025.loc[~ads_2025.index.isin(ads_2032.index)].copy()

ads_2025_to_2032_by_type_state = (
    ads_2025_to_2032.groupby(['type', 'State'])['PSSEMaxCap(MW)']
    .sum()
    .unstack('State', fill_value=0)
)

print('ADS Capacity by 2025 (GW):', ads_2025['PSSEMaxCap(MW)'].sum()/1e3)
print('ADS Capacity by 2032 (GW):', ads_2032['PSSEMaxCap(MW)'].sum()/1e3)
print('ADS Additional Capacity from 2025 to 2032 (GW):', ads_2025_to_2032['PSSEMaxCap(MW)'].sum()/1e3)
print('ADS Retired Capacity from 2025 to 2032 (GW):', ads_retired_till_2032['PSSEMaxCap(MW)'].sum()/1e3)

if not workflow_run:
    plot_capacity_comparison(eia_year_end, ads_2025, "2025 WECC generating and storage capacity")
    plot_capacity_comparison(
        eia_retired_by_2032, ads_retired_till_2032,
        "2026 to 2032 WECC capacity retirements: EIA planned vs. ADS modeled",
    )

# LBNL Queue Data Processing

AS_OF_DATE = pd.Timestamp("2025-12-31")
END_DATE = pd.Timestamp("2032-12-31")

def classify_hybrid(technology):
    if pd.isna(technology):
        return technology
    components = {component.strip() for component in technology.split('+')}
    if 'Battery' in components:
        # Solar takes priority for Solar+Wind+Battery and Solar+Gas+Battery.
        for base in ['Solar', 'Wind', 'Gas', 'Other']:
            if base in components:
                return f'{base} hybrid'
    return technology

# 4. Build and select the queue fleet using the calculated target.
queued_up = pd.read_excel(
    queue_file, sheet_name='03. Complete Queue Data', skiprows=1,
)
queued_up = queued_up.loc[
    ~queued_up['state'].isin(['MX', 'BC', 'AB'])
    & ~queued_up['state'].isna()
    ].copy()
queued_up['on_date'] = pd.to_datetime(queued_up['on_date'], errors='coerce')
queued_up['q_date'] = pd.to_datetime(queued_up['q_date'], errors='coerce')
queued_up['prop_date'] = pd.to_datetime(queued_up['prop_date'], errors='coerce')

active = queued_up.loc[
    (queued_up['q_status'] == 'active')
    & (pd.to_numeric(queued_up['q_year'], errors='coerce').le(2025))
    & (queued_up['region'].isin(['West', 'CAISO']))
].copy()

active['type'] = active['type_clean'].map(classify_hybrid)

operational = queued_up.loc[
    (queued_up['q_status'] == 'operational')
    & (queued_up['region'].isin(['West', 'CAISO']))
    # Include all commissioning dates from 2018 through the end of 2025.
    & (queued_up['on_date'].between('2018-01-01', '2025-12-31', inclusive='both'))
].copy()

# Queue-to-operation durations in years (365.25 days per year).
# Operational cohort: 2018–2025.
operational['queue_to_operation_years'] = (
    (operational['on_date'] - operational['q_date']).dt.total_seconds()
    / (365.25 * 24 * 60 * 60)
)
technologies = ['Solar+Battery', 'Solar', 'Battery', 'Wind', 'Gas', 'Other']
valid_duration = operational['queue_to_operation_years'].ge(0)
duration_data = operational.loc[valid_duration].copy()

# WECC pools the two selected queue regions, retaining the state exclusions above.
region_data = {
    'CAISO': duration_data.loc[duration_data['region'].eq('CAISO')],
    'West': duration_data.loc[duration_data['region'].eq('West')],
    'WECC': duration_data,
}

# Calculate the pooled WECC median from projects, not from regional medians.
median_queue_to_operation_years = pd.DataFrame({
    region: regional.groupby('type_clean')['queue_to_operation_years'].median()
    for region, regional in region_data.items()
}).reindex(technologies)

median_queue_to_operation_years.rename(
    index={'Solar+Battery': 'Solar hybrid'}, inplace=True,
)

# Apply each region's solar hybrid premium using unrounded medians.
# Wind, Gas, and Other hybrid values are estimates, not observed medians.
hybrid_premium_years = (
    median_queue_to_operation_years.loc['Solar hybrid']
    - median_queue_to_operation_years.loc['Solar']
)
for technology in ['Wind', 'Gas', 'Other']:
    median_queue_to_operation_years.loc[f'{technology} hybrid'] = (
        median_queue_to_operation_years.loc[technology] + hybrid_premium_years
    )

# Use Solar as the queue-to-operation time proxy for Solar+Wind in each region.
median_queue_to_operation_years.loc['Solar+Wind'] = (
    median_queue_to_operation_years.loc['Solar']
)

# Match each active project to its own region and technology duration.
duration_lookup = median_queue_to_operation_years.stack()
project_keys = pd.MultiIndex.from_frame(active[['type', 'region']])
active['estimated_queue_to_operation_years'] = (
    duration_lookup.reindex(project_keys).to_numpy()
)
queue_based_date = active['q_date'] + pd.to_timedelta(
    active['estimated_queue_to_operation_years'] * 365.25, unit='D',
)
# Remove floating-point noise near midnight before taking the calendar date.
active['queue_based_online_date'] = queue_based_date.dt.round('s').dt.normalize()
# Use the later date; if either date is missing, use the available one.
# Projects missing both dates remain NaT.
active['estimated_online_date'] = active['queue_based_online_date'].where(
    active['queue_based_online_date'].notna()
    & (
        active['prop_date'].isna()
        | active['queue_based_online_date'].ge(active['prop_date'])
    ),
    active['prop_date'].dt.normalize(),
)

# Keep reported dates/statuses intact; all adjustments are scenario-only.
active['excluded_unknown_technology'] = (
    active[['type_1', 'type_2', 'type_3']].isin(['Other', 'Unknown', 'Other Storage']).any(axis=1)
    | active['type_clean'].str.contains(r'(?:^|\+)\s*(?:Other|Unknown|Other Storage)\s*(?:\+|$)', na=False)
)
# Preserve original type_clean/component fields for traceability.
active['type'] = active['type'].str.replace(r'\bOther\b(?! Storage)', 'Unknown', regex=True)
for component in (1, 2, 3):
    active[f'scenario_type_{component}'] = active[f'type_{component}'].replace({'Other': 'Unknown'})
active['scenario_phase'] = active['IA_phase_clean'].fillna('Unreported')
active['scenario_status'] = active['q_status']
active['scenario_online_date'] = active['estimated_online_date']

active['overdue'] = active['estimated_online_date'].le(AS_OF_DATE)
construction = active['IA_phase_clean'].eq('Construction')

# Apply one five-calendar-year grace period to every overdue project.
active.loc[active['overdue'], 'scenario_online_date'] = (
    active.loc[active['overdue'], 'estimated_online_date'] + pd.DateOffset(years=5)
)
active['excluded_five_year_overdue'] = (
    active['overdue'] & active['scenario_online_date'].lt(AS_OF_DATE)
)
active.loc[active['excluded_five_year_overdue'], 'scenario_status'] = 'suspended'
reschedule = active['overdue'] & ~construction & ~active['excluded_five_year_overdue']
active.loc[reschedule & active['IA_phase_clean'].notna(), 'scenario_phase'] = 'In Progress (unknown study)'
# Construction is guaranteed only if its revised date is in the future fleet window.
active['guaranteed_construction'] = (
    construction & ~active['excluded_five_year_overdue']
    & ~active['excluded_unknown_technology']
    & active['scenario_online_date'].ge('2026-01-01')
    & active['scenario_online_date'].le(END_DATE)
)
active['scenario_online_year'] = active['scenario_online_date'].dt.year.astype('Int64')

phase_weights = {
    'IA Executed': 2.0, 'IA Pending': 2.0,
    'In Progress (unknown study)': 1.0, 'Cluster Study': 1.0,
    'Facility Study': 1.0, 'System Impact Study': 1.0, 'Feasibility Study': 1.0,
    'Not Started': 0.5, 'Unreported': 0.5,
}
# Unmatched phase labels are not sampled.
active['phase_weight'] = active['scenario_phase'].map(phase_weights)
active['unmapped_phase'] = (
    active['scenario_phase'].notna() & active['phase_weight'].isna() & ~construction
)
active['phase_weight'] = active['phase_weight'].fillna(0)
# Linear interpolation: 2026 = 1.0, 2032 = 0.5.
active['timing_weight'] = (
    1 - (active['scenario_online_year'].astype(float) - 2026) / 12
).clip(0.5, 1).fillna(0)
active['selection_weight'] = active['phase_weight'] * active['timing_weight']

# Estimate missing hybrid components from reported MW ratios, never from imputations.
# Use active/operational WECC peers: exact hybrid first, then the component pair.
capacity_peers = queued_up.loc[
    queued_up['region'].isin(['CAISO', 'West'])
    & queued_up['q_status'].isin(['active', 'operational'])
    & pd.to_numeric(queued_up['q_year'], errors='coerce').le(2025)
].copy()
component_records = pd.concat([
    pd.DataFrame({
        'project_index': capacity_peers.index,
        'technology': capacity_peers[f'type_{i}'].values,
        'mw': pd.to_numeric(capacity_peers[f'mw_{i}'], errors='coerce').values,
    })
    for i in (1, 2, 3)
], ignore_index=True).dropna(subset=['technology'])
reported_capacities = component_records.groupby(
    ['project_index', 'technology'],
)['mw'].sum(min_count=1).unstack()

for component in (1, 2, 3):
    active[f'mw_{component}_estimated'] = pd.to_numeric(active[f'mw_{component}'], errors='coerce')
    active[f'mw_{component}_imputed'] = False
    active[f'mw_{component}_imputation_source'] = pd.Series('', index=active.index, dtype='object')
    active[f'mw_{component}_donor_count'] = 0
    active[f'mw_{component}_ratio'] = np.nan
    missing = active[f'type_{component}'].notna() & active[f'mw_{component}_estimated'].isna()
    for project_index, project in active.loc[missing].iterrows():
        estimate = estimate_component_capacity(
            project, project[f'type_{component}'], reported_capacities, capacity_peers,
        )
        if estimate is None:
            continue
        _, _, _, mw, source, ratio, donor_count, anchor_type = estimate
        active.loc[project_index, f'mw_{component}_estimated'] = mw
        active.loc[project_index, f'mw_{component}_imputed'] = True
        active.loc[project_index, f'mw_{component}_imputation_source'] = f'{source}; relative to {anchor_type}'
        active.loc[project_index, f'mw_{component}_donor_count'] = donor_count
        active.loc[project_index, f'mw_{component}_ratio'] = ratio

# Keep a hybrid's component MW together when choosing a queue project.
components = pd.concat([
    active[[f'type_{i}', f'mw_{i}_estimated']].set_axis(['type', 'mw'], axis=1).assign(component=i)
    for i in (1, 2, 3)
]).dropna(subset=['type'])
components['queue_technology'] = components['type']
components['type'] = components['type'].replace({
    'Offshore Wind': 'Wind', 'Oil': 'Other', 'Hydrogen': 'Other',
    'Other Storage': 'Other',
})
# Only classify hydro/storage as pumped storage when the queue identifies it.
pumped = active['project_name'].str.contains(r'pumped[ -]*(?:hydro|storage)', case=False, na=False)
components.loc[components.index.isin(active.index[pumped])
               & components['type'].isin(['Hydro', 'Other']), 'type'] = 'Pumped storage'
invalid_projects = components.index[~np.isfinite(components['mw']) | components['mw'].lt(0)]
project_mw = components.pivot_table(index=components.index, columns='type', values='mw',
                                    aggfunc='sum', fill_value=0).reindex(active.index, fill_value=0)
capacity_valid = ~active.index.isin(invalid_projects) & project_mw.sum(axis=1).gt(0)
if (active['guaranteed_construction'] & ~capacity_valid).any():
    raise ValueError('A guaranteed construction project has unresolved component capacity.')

active['eligible_for_sampling'] = (
    ~active['guaranteed_construction'] & ~active['excluded_five_year_overdue']
    & ~active['excluded_unknown_technology'] & capacity_valid
    & active['scenario_online_date'].between('2026-01-01', END_DATE)
    & active['selection_weight'].gt(0)
)
active['selected'] = active['guaranteed_construction']
types = project_mw.columns.union(ads_2025_to_2032_by_type_state.index)
project_mw = project_mw.reindex(columns=types, fill_value=0)
target_mw = ads_2025_to_2032_by_type_state.reindex(index=types, fill_value=0)
sampling_seed = 42
orderings_per_state = 1

# Cost of one MW above a state battery target, relative to a one-MW shortfall.
battery_surplus_weight = 1.02
battery_column = types.get_loc("Battery")


def target_error(residual):
    battery_surplus = np.maximum(-residual[..., battery_column], 0)
    return np.abs(residual).sum(axis=-1) + (battery_surplus_weight - 1) * battery_surplus


for state, candidates in active.loc[active['eligible_for_sampling']].groupby('state'):
    if state not in target_mw:
        continue
    # Each state's first draws are reproducible regardless of other states' trial counts.
    rng = np.random.default_rng([sampling_seed, *state.encode('ascii')])
    guaranteed = active['selected'] & active['state'].eq(state)
    remaining = (target_mw[state] - project_mw.loc[guaranteed].sum()).to_numpy()
    amounts = project_mw.loc[candidates.index].to_numpy()
    weights = candidates['selection_weight'].to_numpy()
    targets = target_mw[state].to_numpy()
    project_sizes = amounts.sum(axis=1)
    best_error = target_error(remaining)
    best_selection = []
    # Draw whole projects without replacement, updating technology needs each time.
    for _ in range(orderings_per_state):
        residual = remaining.copy()
        available = np.ones(len(candidates), dtype=bool)
        chosen = []
        while available.any():
            deficit = residual.clip(min=0)
            fraction_needed = np.divide(deficit, targets, out=np.zeros_like(deficit, dtype=float),
                                        where=targets > 0)
            useful_mw = np.minimum(amounts, deficit)
            # Credit each component up to its deficit, scaled by how much of its
            # target remains. Divide by total project MW so filled components dilute hybrids.
            priority = (useful_mw * fraction_needed).sum(axis=1) / project_sizes
            improves_fit = target_error(residual - amounts) < target_error(residual) - 1e-8
            draw_weights = weights * priority * available * improves_fit
            total_weight = draw_weights.sum()
            if total_weight == 0:
                break
            pos = rng.choice(len(candidates), p=draw_weights / total_weight)
            chosen.append(candidates.index[pos])
            available[pos] = False
            residual -= amounts[pos]
        error = target_error(residual)
        if error < best_error:
            best_error, best_selection = error, chosen
    # A single draw reports the weighted pass directly, before any refinement.
    if orderings_per_state == 1:
        active.loc[best_selection, 'selected'] = True
        continue
    # Revisit greedy choices. Keep mandatory construction fixed and move whole
    # projects only when an addition, removal, or swap reduces the state's error.
    picked = np.asarray(candidates.index.isin(best_selection))
    residual = remaining - amounts[picked].sum(axis=0)
    while True:
        inside = np.flatnonzero(picked)
        outside = np.flatnonzero(~picked)
        best_error = target_error(residual)
        move = []
        for remove in [None, *inside]:
            freed = residual if remove is None else residual + amounts[remove]
            error = target_error(freed)
            changes = [] if remove is None else [remove]
            if error < best_error - 1e-8:
                best_error, move = error, changes
            if outside.size:
                errors = target_error(freed - amounts[outside])
                pos = errors.argmin()
                if errors[pos] < best_error - 1e-8:
                    best_error, move = errors[pos], changes + [outside[pos]]
        if not move:
            break
        picked[move] = ~picked[move]
        residual = remaining - amounts[picked].sum(axis=0)
    active.loc[candidates.index[picked], 'selected'] = True

selected_projects = active.loc[active['selected']]
selected_components = components.loc[components.index.isin(selected_projects.index)]
future_fleet = (
    selected_projects[['project_name', 'county', 'state', 'scenario_online_date']]
    .join(selected_components, how='inner', validate='one_to_many')
    .rename(index=lambda i: f'lbnl-q{i}', columns={
        'mw': 'capacity_mw',
        'scenario_online_date': 'current_planned_generator_operating_date',
    })
    .rename_axis('queue_project_id').reset_index()
)

def county_join_key(names):
    return names.str.normalize('NFKD').str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)


county_map = gpd.read_file(counties_file)
fleet_counties = county_map.loc[county_map['STUSPS'].isin(future_fleet['state'])]
# Compute polygon centroids in CONUS Albers, then return them as WGS84 coordinates.
centroids = fleet_counties.to_crs(5070).geometry.centroid.to_crs(4326)
county_coords = pd.DataFrame({
    'county_key': county_join_key(fleet_counties['NAME']),
    'state': fleet_counties['STUSPS'],
    'latitude': centroids.y,
    'longitude': centroids.x,
})
future_fleet['county_key'] = county_join_key(future_fleet['county'])
# The Wyoming queue lists Fort Bridger (a place) in the county field.
place_to_county = {('WY', 'FORTBRIDGER'): 'UINTA'}
for (state, place), county in place_to_county.items():
    match = future_fleet['state'].eq(state) & future_fleet['county_key'].eq(place)
    future_fleet.loc[match, 'county_key'] = county
future_fleet = future_fleet.merge(
    county_coords, on=['state', 'county_key'], how='left', validate='many_to_one',
).drop(columns='county_key')

future_fleet['ads_subtype'] = pd.Series(pd.NA, index=future_fleet.index, dtype='string')
gas_additions = future_fleet.loc[future_fleet['type'].eq('Gas')]
for state, generators in gas_additions.groupby('state'):
    if state not in gas_capacity_share_pct.index.get_level_values('State'):
        continue
    shares = gas_capacity_share_pct.loc[state].dropna()
    shares = shares.loc[shares.gt(0)]
    if shares.empty:
        continue
    remaining_mw = shares / 100 * generators['capacity_mw'].sum()
    # Assign whole components, largest first, to the subtype furthest below target.
    for idx, mw in generators['capacity_mw'].sort_values(ascending=False, kind='stable').items():
        subtype = remaining_mw.idxmax()
        future_fleet.at[idx, 'ads_subtype'] = subtype
        remaining_mw.loc[subtype] -= mw



# Keep queue IDs distinct from official EIA identifiers, including for hybrids.
future_fleet['plant_id_eia'] = future_fleet['queue_project_id'].str.removeprefix('lbnl-')
future_fleet['generator_id'] = future_fleet['plant_id_eia'] + '-' + future_fleet.pop('component').astype(str)
future_fleet['plant_name_eia'] = future_fleet.pop('project_name').fillna(future_fleet['queue_project_id'])
future_fleet['operational_status'] = 'proposed'
future_fleet['nerc_region'] = 'WECC'
future_fleet['balancing_authority_code_eia'] = pd.NA
future_fleet['summer_capacity_mw'] = future_fleet['capacity_mw']
future_fleet['winter_capacity_mw'] = future_fleet['capacity_mw']
future_fleet['minimum_load_mw'] = 0.0
# set_parameters uses the operating date to form peer groups, including for proposals.
future_fleet['generator_operating_date'] = pd.to_datetime(future_fleet['current_planned_generator_operating_date'])

# Map queue labels to the same EIA descriptions and codes used by build_powerplants.
technology_codes = {
    'Solar': ('Solar Photovoltaic', 'PV', 'SUN'),
    'Battery': ('Batteries', 'BA', 'MWH'),
    'Wind': ('Onshore Wind Turbine', 'WT', 'WND'),
    'Offshore Wind': ('Offshore Wind Turbine', 'WS', 'WND'),
    'Geothermal': ('Geothermal', 'ST', 'GEO'),
    'Nuclear': ('Nuclear', 'ST', 'NUC'),
    'ST-NG': ('Natural Gas Steam Turbine', 'ST', 'NG'),
    'GT': ('Natural Gas Fired Combustion Turbine', 'GT', 'NG'),
    'ICE': ('Natural Gas Internal Combustion Engine', 'IC', 'NG'),
    'CC': ('Natural Gas Fired Combined Cycle', 'CC', 'NG'),
}
gas = future_fleet['type'].eq('Gas')
technology = future_fleet['queue_technology'].mask(gas, future_fleet['ads_subtype'])
codes = pd.DataFrame.from_dict(technology_codes, orient='index', columns=[
    'technology_description', 'prime_mover_code', 'energy_source_code_1',
])
for field in codes:
    future_fleet[field] = technology.map(codes[field])
future_fleet.loc[gas, 'energy_source_code_1'] = 'NG'
battery = future_fleet['type'].eq('Battery')
future_fleet['energy_storage_capacity_mwh'] = 4 * future_fleet['capacity_mw'].where(battery)
fleet_input_columns = [
    'plant_id_eia', 'generator_id', 'plant_name_eia', 'operational_status',
    'generator_operating_date', 'current_planned_generator_operating_date',
    'state', 'latitude', 'longitude', 'nerc_region', 'balancing_authority_code_eia',
    'technology_description', 'prime_mover_code', 'energy_source_code_1',
    'capacity_mw', 'summer_capacity_mw', 'winter_capacity_mw', 'minimum_load_mw',
    'energy_storage_capacity_mwh',
]
fleet_file.parent.mkdir(parents=True, exist_ok=True)
future_fleet.to_csv(fleet_file, columns=fleet_input_columns,
                    index=False, date_format='%Y-%m-%d')

selected_mw = project_mw.loc[selected_projects.index]
selected_mw_by_type_state = selected_mw.groupby(selected_projects['state']).sum().T
target_mw, selected_mw_by_type_state = target_mw.align(selected_mw_by_type_state, fill_value=0)
queue_target_comparison = pd.concat({
    'target_mw': target_mw.stack(),
    'selected_mw': selected_mw_by_type_state.stack(),
}, axis=1).rename_axis(['type', 'State'])
queue_target_comparison['gap_mw'] = queue_target_comparison['target_mw'] - queue_target_comparison['selected_mw']
print('Queue additions by type and state (positive gap = shortfall, MW):')
print(queue_target_comparison.loc[queue_target_comparison[['target_mw', 'selected_mw']].ne(0).any(axis=1)].round(2))

if not workflow_run:
    plot_ads_additions(ads_2025_to_2032, selected_projects, selected_mw)

    retained_eia = eia_year_end.drop(index=eia_retired_by_2032.index)
    queue_additions = (selected_mw_by_type_state.rename_axis(index='type', columns='state')
                       .stack().rename('comparison_capacity_mw').reset_index())
    projected_2032 = pd.concat([
        retained_eia[['type', 'state', 'comparison_capacity_mw']],
        queue_additions.loc[queue_additions['comparison_capacity_mw'].gt(0)],
    ], ignore_index=True)
    plot_capacity_comparison(
        projected_2032, ads_2032, '2032 WECC generating and storage capacity',
        eia_label='EIA 2025 + LBNL queue', ads_label='ADS 2032',
        note='Projection uses 2025 EIA summer capacity (nameplate fallback), subtracts planned '
             'retirements through 2032, and adds sampled queue MW. ADS uses PSSE maximum capacity.',
    )
    plt.show()
