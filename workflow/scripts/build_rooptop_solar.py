import geopandas as gpd
import pandas as pd

lbnl = pd.read_csv(
    '../data/rooftop_solar/Tracking the Sun 2024 Data File/TTS_LBNL_public_file_21-Aug-2024_all.csv',
    na_values=[-1],
    low_memory=False,
    dtype={'zip_code': 'string'}
)

zip_tract = pd.read_excel('../data/rooftop_solar/ZIP-TRACT_032026.xlsx')

zip_tract = zip_tract.rename(columns={
    'zip' : 'zip_code',
    'state' : 'state_hud',
    'geoid': 'tract'
})

# ZIP codes are numeric in LBNL, while census tract GEOIDs must remain
# zero-padded 11-character strings.
zip_tract['zip_code'] = pd.to_numeric(
    zip_tract['zip_code'], errors='coerce'
).astype('Int64')
zip_tract['tract'] = (
    pd.to_numeric(zip_tract['tract'], errors='coerce')
    .astype('Int64')
    .astype('string')
    .str.zfill(11)
)
zip_tract['tot_ratio'] = pd.to_numeric(
    zip_tract['tot_ratio'], errors='coerce'
)

zip_state = (
    zip_tract[["zip_code", "state_hud"]]
    .drop_duplicates()
)

# Ensure exactly one preferred state per ZIP
assert not zip_state["zip_code"].duplicated().any()

required_cols = ['system_ID_1',
                 'installation_date',
                 'PV_system_size_DC',
                 'inverter_loading_ratio',
                 'battery_rated_capacity_kW',
                 'battery_rated_capacity_kWh',
                 'customer_segment',
                 'ground_mounted',
                 'zip_code',
                 'city',
                 'state',
                 'utility_service_territory']

lbnl = lbnl[required_cols].copy()

# Extract five-digit ZIP, convert to numeric, and remove invalid/missing ZIPs
lbnl["zip_code"] = pd.to_numeric(
    lbnl["zip_code"].str.extract(r"^(\d{5})", expand=False),
    errors="coerce",
)
lbnl = lbnl.dropna(subset=["zip_code"])
lbnl["zip_code"] = lbnl["zip_code"].astype("int64")

# Keep data through 2024 only
lbnl['installation_date'] = pd.to_datetime(
    lbnl['installation_date'], errors='coerce'
)

lbnl_2024 = lbnl.loc[
    lbnl['installation_date'] < pd.Timestamp('2025-01-01')
].copy()

# Remove data with system size greater than 1 MW

lbnl_2024 = lbnl_2024.loc[lbnl_2024['PV_system_size_DC'] < 1000]

capacity_cols = [
    'PV_system_size_DC',
    'battery_rated_capacity_kW',
    'battery_rated_capacity_kWh'
]

lbnl_2024 = lbnl_2024.merge(
    zip_state,
    on='zip_code',
    how='left',
    validate='many_to_one'
)

filtered_record_count = len(lbnl_2024)
total_filtered_capacity = lbnl_2024[capacity_cols].sum(min_count=1)

missing_zip_state = lbnl_2024.loc[lbnl_2024['state_hud'].isna()].copy()

state_mismatch = lbnl_2024.loc[
    lbnl_2024['state_hud'].notna()
    & lbnl_2024['state'].ne(lbnl_2024['state_hud'])
].copy()

lbnl_2024 = lbnl_2024.loc[
    lbnl_2024['state'].eq(lbnl_2024['state_hud'])
].copy()

grouped_by_zipcode = lbnl_2024.groupby(
    ['state', 'zip_code'], as_index=False
)[capacity_cols].sum(min_count=1)

# Normalize HUD tot_ratio within each ZIP so allocating a ZIP among tracts
# conserves every capacity metric exactly.
zip_tract_allocation = zip_tract[
    ['zip_code', 'state_hud', 'tract', 'tot_ratio']
].dropna(subset=['zip_code', 'state_hud', 'tract', 'tot_ratio']).copy()

ratio_totals = zip_tract_allocation.groupby(
    ['zip_code', 'state_hud']
)['tot_ratio'].transform('sum')
if ratio_totals.le(0).any():
    raise ValueError('ZIP-tract crosswalk contains a non-positive tot_ratio total')

zip_tract_allocation['allocation_ratio'] = (
    zip_tract_allocation['tot_ratio'] / ratio_totals
)

tract_coordinate_source = pd.read_csv(
    '../data/rooftop_solar/2025_Gaz_tracts_national.txt',
    sep='|',
    usecols=['USPS', 'GEOID', 'INTPTLAT', 'INTPTLONG'],
    dtype={'GEOID': 'string'}
).rename(columns={
    'USPS': 'state',
    'GEOID': 'tract',
    'INTPTLAT': 'latitude',
    'INTPTLONG': 'longitude'
})
tract_coordinate_source['tract'] = (
    tract_coordinate_source['tract'].str.zfill(11)
)
tract_coordinates = tract_coordinate_source[[
    'tract',
    'latitude',
    'longitude'
]].copy()

if tract_coordinates['tract'].duplicated().any():
    raise ValueError('Tract coordinate file contains duplicate GEOIDs')

# The official Census workbook identifies tracts by state, county, and printed
# tract number rather than by an 11-digit GEOID. Use the 2020 Census county
# file to recover the state and county FIPS codes, then append the six-digit
# tract code derived from the printed tract number.
tract_population = pd.read_excel(
    '../data/rooftop_solar/tract-agegroup-2020.xlsx',
    sheet_name='Sheet1',
    header=6,
    usecols=[0, 1, 2, 3]
).rename(columns={
    'State': 'state_name',
    'County': 'county_name',
    'Tract': 'tract_name',
    '2020 Population': 'Pop20'
})
tract_population = tract_population.dropna(
    subset=['state_name', 'county_name', 'tract_name']
).copy()
tract_population['Pop20'] = pd.to_numeric(
    tract_population['Pop20'], errors='coerce'
)

county_lookup = gpd.read_file(
    '../data/counties/cb_2020_us_county_500k.shp',
    ignore_geometry=True
)[[
    'STATEFP',
    'COUNTYFP',
    'STUSPS',
    'STATE_NAME',
    'NAMELSAD'
]].drop_duplicates()

state_reference = county_lookup[[
    'STUSPS',
    'STATE_NAME'
]].drop_duplicates()
state_reference = state_reference.loc[
    ~state_reference['STUSPS'].isin(['AS', 'GU', 'MP', 'PR', 'VI'])
]
missing_lbnl_states = state_reference.loc[
    ~state_reference['STUSPS'].isin(lbnl_2024['state'].dropna().unique())
].sort_values('STUSPS')

tract_population = tract_population.merge(
    county_lookup,
    left_on=['state_name', 'county_name'],
    right_on=['STATE_NAME', 'NAMELSAD'],
    how='left',
    validate='many_to_one'
)

missing_county_fips = tract_population.loc[
    tract_population['STATEFP'].isna()
    | tract_population['COUNTYFP'].isna()
].copy()
if not missing_county_fips.empty:
    raise ValueError(
        'Could not assign county FIPS codes to '
        f'{len(missing_county_fips):,} tract population rows'
    )

tract_number_parts = tract_population['tract_name'].str.extract(
    r'^Census Tract (\d{1,4})(?:\.(\d{1,2}))?$'
)
invalid_tract_names = tract_number_parts[0].isna()
if invalid_tract_names.any():
    raise ValueError(
        'Could not parse the printed tract number for '
        f'{invalid_tract_names.sum():,} tract population rows'
    )

tract_code = (
    tract_number_parts[0].str.zfill(4)
    + tract_number_parts[1].fillna('').str.ljust(2, fillchar='0')
)
tract_population['tract'] = (
    tract_population['STATEFP']
    + tract_population['COUNTYFP']
    + tract_code
)
tract_population['state'] = tract_population['STUSPS']
tract_population = tract_population[[
    'state',
    'tract',
    'Pop20'
]]

if tract_population['tract'].duplicated().any():
    raise ValueError('Tract population workbook contains duplicate GEOIDs')

tract_population = tract_population.merge(
    tract_coordinates,
    on='tract',
    how='left',
    validate='one_to_one'
)

# Connecticut replaced its eight counties with nine planning regions after the
# 2020 Census, which changed the county portion of every tract GEOID. Reconcile
# such missing 2020 GEOIDs to the current Gazetteer by state and six-digit tract
# code, but only where that key identifies exactly one current tract.
population_without_direct_coordinates = tract_population.loc[
    tract_population['latitude'].isna()
    | tract_population['longitude'].isna()
].copy()
population_without_direct_coordinates['tract_code'] = (
    population_without_direct_coordinates['tract'].str[-6:]
)

tract_coordinate_fallback = tract_coordinate_source.copy()
tract_coordinate_fallback['tract_code'] = (
    tract_coordinate_fallback['tract'].str[-6:]
)
tract_coordinate_fallback = tract_coordinate_fallback.loc[
    ~tract_coordinate_fallback.duplicated(
        ['state', 'tract_code'], keep=False
    )
].rename(columns={
    'tract': 'current_tract',
    'latitude': 'current_latitude',
    'longitude': 'current_longitude'
})

population_coordinate_fallbacks = (
    population_without_direct_coordinates.reset_index()
    .merge(
        tract_coordinate_fallback[[
            'state',
            'tract_code',
            'current_tract',
            'current_latitude',
            'current_longitude'
        ]],
        on=['state', 'tract_code'],
        how='left',
        validate='many_to_one'
    )
    .loc[lambda df: df['current_tract'].notna()]
)

if not population_coordinate_fallbacks.empty:
    fallback_index = population_coordinate_fallbacks['index']
    tract_population.loc[fallback_index, 'tract'] = (
        population_coordinate_fallbacks['current_tract'].to_numpy()
    )
    tract_population.loc[fallback_index, 'latitude'] = (
        population_coordinate_fallbacks['current_latitude'].to_numpy()
    )
    tract_population.loc[fallback_index, 'longitude'] = (
        population_coordinate_fallbacks['current_longitude'].to_numpy()
    )

if tract_population['tract'].duplicated().any():
    raise ValueError(
        'Population-to-current-tract reconciliation created duplicate GEOIDs'
    )

missing_population_tract_coordinates = tract_population.loc[
    tract_population['latitude'].isna()
    | tract_population['longitude'].isna()
].copy()


def allocate_zip_capacity_to_tracts(zip_capacity):
    """Allocate combined ZIP capacity to tracts using HUD tot_ratio."""
    allocations = zip_capacity.merge(
        zip_tract_allocation[
            ['zip_code', 'state_hud', 'tract', 'allocation_ratio']
        ],
        left_on=['zip_code', 'state'],
        right_on=['zip_code', 'state_hud'],
        how='left',
        validate='one_to_many'
    )

    missing_crosswalk = allocations.loc[
        allocations['tract'].isna()
    ].copy()
    allocations = allocations.loc[allocations['tract'].notna()].copy()

    for capacity_col in capacity_cols:
        allocations[capacity_col] = (
            allocations[capacity_col] * allocations['allocation_ratio']
        )

    allocations = allocations.merge(
        tract_coordinates,
        on='tract',
        how='left',
        validate='many_to_one'
    )
    missing_coordinates = allocations.loc[
        allocations['latitude'].isna() | allocations['longitude'].isna()
    ].copy()

    grouped_by_tract = (
        allocations.groupby(['state', 'tract'], as_index=False)[capacity_cols]
        .sum(min_count=1)
        .merge(
            tract_coordinates,
            on='tract',
            how='left',
            validate='many_to_one'
        )
    )

    expected = (
        zip_capacity.merge(
            zip_tract_allocation[['zip_code', 'state_hud']].drop_duplicates(),
            left_on=['zip_code', 'state'],
            right_on=['zip_code', 'state_hud'],
            how='inner',
            validate='one_to_one'
        )
        .set_index(['state', 'zip_code'])[capacity_cols]
        .sort_index()
    )
    allocated = (
        allocations.groupby(['state', 'zip_code'])[capacity_cols]
        .sum(min_count=1)
        .sort_index()
    )
    maximum_allocation_error = (
        allocated.subtract(expected).abs().max().max()
    )

    return (
        grouped_by_tract,
        allocations,
        missing_crosswalk,
        missing_coordinates,
        maximum_allocation_error,
    )


(
    _,
    tract_allocations_unscaled,
    missing_zip_tract,
    missing_location,
    _,
) = allocate_zip_capacity_to_tracts(grouped_by_zipcode)


def percentage(numerator, denominator):
    """Return a percentage while safely handling missing or zero totals."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return pd.NA
    return 100 * numerator / denominator


record_diagnostic_report = pd.DataFrame([
    {
        'diagnostic': 'State mismatch',
        'count_type': 'LBNL records',
        'affected_count': len(state_mismatch),
        'reference_count': filtered_record_count,
        'affected_percent': percentage(
            len(state_mismatch), filtered_record_count
        )
    },
    {
        'diagnostic': 'Missing ZIP-to-state mapping',
        'count_type': 'LBNL records',
        'affected_count': len(missing_zip_state),
        'reference_count': filtered_record_count,
        'affected_percent': percentage(
            len(missing_zip_state), filtered_record_count
        )
    },
    {
        'diagnostic': 'Missing ZIP-to-tract mapping',
        'count_type': 'State-matched ZIP groups',
        'affected_count': len(missing_zip_tract),
        'reference_count': len(grouped_by_zipcode),
        'affected_percent': percentage(
            len(missing_zip_tract), len(grouped_by_zipcode)
        )
    },
    {
        'diagnostic': 'Missing tract location',
        'count_type': 'ZIP-to-tract allocations',
        'affected_count': len(missing_location),
        'reference_count': len(tract_allocations_unscaled),
        'affected_percent': percentage(
            len(missing_location), len(tract_allocations_unscaled)
        )
    }
])

capacity_labels = {
    'PV_system_size_DC': 'PV capacity (kW)',
    'battery_rated_capacity_kW': 'Battery power (kW)',
    'battery_rated_capacity_kWh': 'Battery energy (kWh)'
}

capacity_diagnostic_data = {
    'State mismatch': state_mismatch,
    'Missing ZIP-to-state mapping': missing_zip_state,
    'Missing ZIP-to-tract mapping': missing_zip_tract,
    'Missing tract location': missing_location
}

capacity_diagnostic_rows = []
for diagnostic, affected_data in capacity_diagnostic_data.items():
    affected_capacity = affected_data[capacity_cols].sum(min_count=1)
    for capacity_col in capacity_cols:
        capacity_diagnostic_rows.append({
            'diagnostic': diagnostic,
            'capacity_metric': capacity_labels[capacity_col],
            'affected_capacity': affected_capacity[capacity_col],
            'total_filtered_capacity': total_filtered_capacity[capacity_col],
            'affected_percent': percentage(
                affected_capacity[capacity_col],
                total_filtered_capacity[capacity_col]
            )
        })

all_affected_data = pd.concat(
    [data[capacity_cols] for data in capacity_diagnostic_data.values()],
    ignore_index=True
)
total_affected_capacity = all_affected_data[capacity_cols].sum(min_count=1)

for capacity_col in capacity_cols:
    capacity_diagnostic_rows.append({
        'diagnostic': 'Total mismatch across all diagnostics',
        'capacity_metric': capacity_labels[capacity_col],
        'affected_capacity': total_affected_capacity[capacity_col],
        'total_filtered_capacity': total_filtered_capacity[capacity_col],
        'affected_percent': percentage(
            total_affected_capacity[capacity_col],
            total_filtered_capacity[capacity_col]
        )
    })

capacity_diagnostic_report = pd.DataFrame(capacity_diagnostic_rows)

print('\nRecord and location diagnostics:')
print(record_diagnostic_report.to_string(
    index=False,
    formatters={
        'affected_percent': lambda value: f'{value:.4f}%'
    }
))

print('\nCapacity diagnostics:')
print(capacity_diagnostic_report.to_string(
    index=False,
    formatters={
        'affected_capacity': lambda value: f'{value:,.2f}',
        'total_filtered_capacity': lambda value: f'{value:,.2f}',
        'affected_percent': lambda value: f'{value:.4f}%'
    }
))

statewide_actual = pd.read_excel(
    "../data/rooftop_solar/net_metering_2024.xlsx",
    sheet_name="Monthly_Totals-States",
    header=[1, 2, 3],
    na_values=".",
)

# Locate the identification columns without depending on their upper headers
year_col = next(col for col in statewide_actual.columns if col[-1] == "Year")
month_col = next(col for col in statewide_actual.columns if col[-1] == "Month")
state_col = next(col for col in statewide_actual.columns if col[-1] == "State")
status_col = next(
    col for col in statewide_actual.columns
    if col[-1] == "Data Status"
)

capacity_columns = {
    "pv_capacity_mw": (
        "Photovoltaic",
        "Capacity MW",
        "Total",
    ),
    "battery_power_mw": (
        "Battery",
        "PV-Paired Battery Capacity (MW)",
        "Total",
    ),
    "battery_energy_mwh": (
        "Battery",
        "PV-Paired Energy Capacity (MWh)",
        "Total",
    ),
}

# Keep the final December 2024 statewide totals
december_2024 = statewide_actual.loc[
    pd.to_numeric(statewide_actual[year_col], errors="coerce").eq(2024)
    & pd.to_numeric(statewide_actual[month_col], errors="coerce").eq(12)
    & statewide_actual[status_col].eq("Final"),
    [state_col, *capacity_columns.values()],
].copy()

december_2024.columns = [
    "state",
    "pv_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
]

# Ensure capacity columns are numeric
capacity_mw_cols = [
    "pv_capacity_mw",
    "battery_power_mw",
    "battery_energy_mwh",
]

december_2024[capacity_mw_cols] = december_2024[
    capacity_mw_cols
].apply(pd.to_numeric, errors="coerce")

# Convert to the same units used by LBNL
statewide_capacity = december_2024.assign(
    PV_system_size_DC=lambda df: df["pv_capacity_mw"] * 1_000,
    battery_rated_capacity_kW=lambda df: df["battery_power_mw"] * 1_000,
    battery_rated_capacity_kWh=lambda df: df["battery_energy_mwh"] * 1_000,
)[
    [
        "state",
        "PV_system_size_DC",
        "battery_rated_capacity_kW",
        "battery_rated_capacity_kWh",
    ]
]

# Confirm there is only one target row per state
assert not statewide_capacity["state"].duplicated().any()

# Negative installed-capacity targets are reporting adjustments, not physical
# capacities. Preserve them for diagnostics, then treat them as missing so the
# corresponding state/metric can retain its LBNL-derived value during scaling.
negative_statewide_targets = (
    statewide_capacity.melt(
        id_vars="state",
        value_vars=capacity_cols,
        var_name="capacity_metric",
        value_name="reported_capacity",
    )
    .loc[lambda df: df["reported_capacity"] < 0]
    .reset_index(drop=True)
)

if not negative_statewide_targets.empty:
    print("\nNegative statewide targets treated as missing:")
    print(negative_statewide_targets.to_string(index=False))

statewide_capacity[capacity_cols] = statewide_capacity[capacity_cols].mask(
    statewide_capacity[capacity_cols] < 0
)

missing_lbnl_state_capacity = missing_lbnl_states.merge(
    statewide_capacity,
    left_on='STUSPS',
    right_on='state',
    how='left',
    validate='one_to_one'
).drop(columns='state')
missing_lbnl_state_capacity[capacity_cols] = (
    missing_lbnl_state_capacity[capacity_cols] / 1_000
)
missing_lbnl_state_capacity = missing_lbnl_state_capacity.rename(columns={
    'PV_system_size_DC': 'Rooftop PV capacity (MW)',
    'battery_rated_capacity_kW': 'Battery power capacity (MW)',
    'battery_rated_capacity_kWh': 'Battery energy capacity (MWh)'
})

print('\nStates with no usable LBNL records through 2024:')
if missing_lbnl_state_capacity.empty:
    print('None')
else:
    print(missing_lbnl_state_capacity.to_string(index=False))

# Calculate one proportional scaling factor per state and capacity metric.
state_capacity_before_scaling = (
    grouped_by_zipcode.groupby('state', as_index=False)[capacity_cols]
    .sum(min_count=1)
)

state_capacity_comparison = state_capacity_before_scaling.merge(
    statewide_capacity,
    on='state',
    how='outer',
    suffixes=('_LBNL', '_EIA'),
    validate='one_to_one'
).sort_values('state')

comparison_names = {
    'PV_system_size_DC': 'Rooftop PV (MW)',
    'battery_rated_capacity_kW': 'Battery power (MW)',
    'battery_rated_capacity_kWh': 'Battery energy (MWh)'
}
comparison_columns = {'state': 'State'}
for capacity_col, capacity_name in comparison_names.items():
    lbnl_col = f'{capacity_col}_LBNL'
    eia_col = f'{capacity_col}_EIA'
    difference_col = f'{capacity_col}_difference'
    percent_col = f'{capacity_col}_percent_reported'
    state_capacity_comparison[[lbnl_col, eia_col]] /= 1_000
    state_capacity_comparison[difference_col] = (
        state_capacity_comparison[eia_col]
        - state_capacity_comparison[lbnl_col]
    )
    state_capacity_comparison[percent_col] = (
        state_capacity_comparison[lbnl_col]
        / state_capacity_comparison[eia_col].where(
            state_capacity_comparison[eia_col] > 0
        )
        * 100
    )
    comparison_columns.update({
        lbnl_col: f'LBNL {capacity_name}',
        eia_col: f'EIA {capacity_name}',
        difference_col: f'EIA - LBNL {capacity_name}',
        percent_col: f'LBNL as % of EIA {capacity_name}'
    })

state_capacity_comparison = state_capacity_comparison[
    list(comparison_columns)
].rename(columns=comparison_columns)

print('\nUnscaled LBNL capacity compared with December 2024 EIA totals:')
print(state_capacity_comparison.to_string(index=False))
state_capacity_comparison.to_csv(
    '../data/rooftop_solar/lbnl_eia_state_capacity_comparison.csv',
    index=False
)

state_scaling_factors = state_capacity_before_scaling.merge(
    statewide_capacity,
    on='state',
    how='outer',
    suffixes=('_before_scaling', '_target'),
    validate='one_to_one'
)

scaling_factor_cols = []
for capacity_col in capacity_cols:
    before_col = f'{capacity_col}_before_scaling'
    target_col = f'{capacity_col}_target'
    factor_col = f'{capacity_col}_scaling_factor'
    scaling_factor_cols.append(factor_col)

    valid_scaling = (
        state_scaling_factors[target_col].notna()
        & state_scaling_factors[before_col].gt(0)
    )

    state_scaling_factors[factor_col] = 1.0
    state_scaling_factors.loc[valid_scaling, factor_col] = (
        state_scaling_factors.loc[valid_scaling, target_col]
        / state_scaling_factors.loc[valid_scaling, before_col]
    )

grouped_by_zipcode = grouped_by_zipcode.merge(
    state_scaling_factors[['state', *scaling_factor_cols]],
    on='state',
    how='left',
    validate='many_to_one'
)

for capacity_col in capacity_cols:
    factor_col = f'{capacity_col}_scaling_factor'
    grouped_by_zipcode[capacity_col] = (
        grouped_by_zipcode[capacity_col]
        * grouped_by_zipcode[factor_col].fillna(1.0)
    )

grouped_by_zipcode = grouped_by_zipcode.drop(columns=scaling_factor_cols)

state_capacity_after_scaling = (
    grouped_by_zipcode.groupby('state', as_index=False)[capacity_cols]
    .sum(min_count=1)
)

# Allocate the scaled ZIP totals to census tracts. Latitude and longitude are
# Census tract internal points.
(
    grouped_by_tract,
    _,
    missing_zip_tract_scaled,
    _,
    maximum_tract_allocation_error,
) = allocate_zip_capacity_to_tracts(grouped_by_zipcode)

print('\nMaximum ZIP-to-tract allocation error:')
print(f'{maximum_tract_allocation_error:,.10f}')

tract_capacity_before_population_fill = (
    grouped_by_tract.groupby('state', as_index=False)[capacity_cols]
    .sum(min_count=1)
)

# Use the statewide target where one is available. For a negative/missing
# statewide target, retain the scaled LBNL total as the goal for that metric.
state_capacity_goals = state_capacity_after_scaling.merge(
    statewide_capacity,
    on='state',
    how='outer',
    suffixes=('_after_scaling', '_target'),
    validate='one_to_one'
)

for capacity_col in capacity_cols:
    after_col = f'{capacity_col}_after_scaling'
    target_col = f'{capacity_col}_target'
    goal_col = f'{capacity_col}_goal'
    state_capacity_goals[goal_col] = (
        state_capacity_goals[target_col]
        .combine_first(state_capacity_goals[after_col])
    )

# Only tracts with a positive official 2020 population and usable coordinates
# can receive a population-based allocation.
population_eligible_tracts = tract_population.loc[
    tract_population['Pop20'].gt(0)
    & tract_population['latitude'].notna()
    & tract_population['longitude'].notna()
].copy()
state_population_totals = (
    population_eligible_tracts.groupby('state', as_index=False)['Pop20']
    .sum()
    .rename(columns={'Pop20': 'state_population_available'})
)
population_eligible_tracts = population_eligible_tracts.merge(
    state_population_totals,
    on='state',
    how='left',
    validate='many_to_one'
)
population_eligible_tracts['population_weight'] = (
    population_eligible_tracts['Pop20']
    / population_eligible_tracts['state_population_available']
)

# After ordinary LBNL scaling, the remaining gaps are normally the state and
# metric combinations that had no positive LBNL seed. Any capacity lost because
# a scaled ZIP lacks a tract crosswalk is also restored through these weights.
capacity_gap_rows = []
for capacity_col in capacity_cols:
    goal_col = f'{capacity_col}_goal'
    current_col = f'{capacity_col}_before_population_fill'
    capacity_gap = state_capacity_goals[
        ['state', goal_col]
    ].merge(
        tract_capacity_before_population_fill[
            ['state', capacity_col]
        ].rename(columns={capacity_col: current_col}),
        on='state',
        how='left',
        validate='one_to_one'
    )
    capacity_gap['capacity_metric'] = capacity_col
    capacity_gap = capacity_gap.rename(columns={
        goal_col: 'capacity_goal',
        current_col: 'capacity_before_population_fill'
    })
    capacity_gap['capacity_before_population_fill'] = (
        capacity_gap['capacity_before_population_fill'].fillna(0)
    )
    capacity_gap['capacity_gap'] = (
        capacity_gap['capacity_goal']
        - capacity_gap['capacity_before_population_fill']
    )
    capacity_gap_rows.append(capacity_gap)

state_capacity_gaps = pd.concat(capacity_gap_rows, ignore_index=True)
state_capacity_gaps = state_capacity_gaps.merge(
    state_population_totals,
    on='state',
    how='left',
    validate='many_to_one'
)

capacity_gap_tolerance = 1e-6
positive_state_capacity_gaps = state_capacity_gaps.loc[
    state_capacity_gaps['capacity_goal'].notna()
    & state_capacity_gaps['capacity_gap'].gt(capacity_gap_tolerance)
].copy()
overfilled_state_targets = state_capacity_gaps.loc[
    state_capacity_gaps['capacity_goal'].notna()
    & state_capacity_gaps['capacity_gap'].lt(-capacity_gap_tolerance)
].copy()
unresolved_population_gaps = positive_state_capacity_gaps.loc[
    ~positive_state_capacity_gaps['state_population_available'].gt(0)
].copy()
fillable_state_capacity_gaps = positive_state_capacity_gaps.loc[
    positive_state_capacity_gaps['state_population_available'].gt(0)
].copy()

population_gap_allocations = population_eligible_tracts[[
    'state',
    'tract',
    'Pop20',
    'population_weight',
    'latitude',
    'longitude'
]].merge(
    fillable_state_capacity_gaps[[
        'state',
        'capacity_metric',
        'capacity_gap'
    ]],
    on='state',
    how='inner',
    validate='many_to_many'
)
population_gap_allocations['allocated_capacity'] = (
    population_gap_allocations['capacity_gap']
    * population_gap_allocations['population_weight']
)

if population_gap_allocations.empty:
    population_capacity_by_tract = pd.DataFrame(columns=[
        'state',
        'tract',
        *capacity_cols
    ])
else:
    population_capacity_by_tract = (
        population_gap_allocations.pivot_table(
            index=['state', 'tract'],
            columns='capacity_metric',
            values='allocated_capacity',
            aggfunc='sum',
            fill_value=0
        )
        .reset_index()
    )
    population_capacity_by_tract.columns.name = None
    for capacity_col in capacity_cols:
        if capacity_col not in population_capacity_by_tract.columns:
            population_capacity_by_tract[capacity_col] = 0.0

# grouped_by_tract is the final result: scaled LBNL capacity plus the remaining
# statewide gaps allocated in proportion to official 2020 tract population.
grouped_by_tract = (
    pd.concat([
        grouped_by_tract[
            ['state', 'tract', *capacity_cols]
        ],
        population_capacity_by_tract[
            ['state', 'tract', *capacity_cols]
        ]
    ], ignore_index=True)
    .groupby(['state', 'tract'], as_index=False)[capacity_cols]
    .sum(min_count=1)
    .merge(
        tract_coordinates,
        on='tract',
        how='left',
        validate='many_to_one'
    )
)

tract_capacity_by_state = (
    grouped_by_tract.groupby('state', as_index=False)[capacity_cols]
    .sum(min_count=1)
)
population_fill_by_state = (
    population_gap_allocations.groupby(
        ['state', 'capacity_metric'], as_index=False
    )['allocated_capacity']
    .sum()
)

tract_reconciliation_rows = []
for capacity_col in capacity_cols:
    reconciliation = state_capacity_gaps.loc[
        state_capacity_gaps['capacity_metric'].eq(capacity_col),
        [
            'state',
            'capacity_goal',
            'capacity_before_population_fill',
            'capacity_gap'
        ]
    ].merge(
        population_fill_by_state.loc[
            population_fill_by_state['capacity_metric'].eq(capacity_col),
            ['state', 'allocated_capacity']
        ].rename(columns={
            'allocated_capacity': 'capacity_population_fill'
        }),
        on='state',
        how='left',
        validate='one_to_one'
    ).merge(
        tract_capacity_by_state[
            ['state', capacity_col]
        ].rename(columns={capacity_col: 'capacity_final'}),
        on='state',
        how='left',
        validate='one_to_one'
    )
    reconciliation['capacity_metric'] = capacity_col
    reconciliation['capacity_population_fill'] = (
        reconciliation['capacity_population_fill'].fillna(0)
    )
    reconciliation['capacity_final'] = (
        reconciliation['capacity_final'].fillna(0)
    )
    reconciliation['difference_from_goal'] = (
        reconciliation['capacity_final']
        - reconciliation['capacity_goal']
    )
    tract_reconciliation_rows.append(reconciliation)

tract_reconciliation = pd.concat(
    tract_reconciliation_rows,
    ignore_index=True
)
maximum_final_reconciliation_error = tract_reconciliation.loc[
    tract_reconciliation['capacity_goal'].notna(),
    'difference_from_goal'
].abs().max()

unallocated_capacity = missing_zip_tract_scaled[capacity_cols].sum()
scaled_zip_capacity = grouped_by_zipcode[capacity_cols].sum(min_count=1)

tract_allocation_diagnostic = pd.DataFrame([
    {
        'capacity_metric': capacity_labels[capacity_col],
        'unallocated_capacity': unallocated_capacity[capacity_col],
        'scaled_zip_capacity': scaled_zip_capacity[capacity_col],
        'unallocated_percent': percentage(
            unallocated_capacity[capacity_col],
            scaled_zip_capacity[capacity_col]
        )
    }
    for capacity_col in capacity_cols
])

print('\nScaled capacity without a ZIP-to-tract mapping:')
print(tract_allocation_diagnostic.to_string(
    index=False,
    formatters={
        'unallocated_capacity': lambda value: f'{value:,.2f}',
        'scaled_zip_capacity': lambda value: f'{value:,.2f}',
        'unallocated_percent': lambda value: f'{value:.4f}%'
    }
))

population_location_diagnostic = pd.DataFrame([{
    'population_tract_rows': len(tract_population),
    'tract_rows_matched_by_vintage_fallback': len(
        population_coordinate_fallbacks
    ),
    'tract_rows_with_coordinates': (
        len(tract_population) - len(missing_population_tract_coordinates)
    ),
    'tract_rows_without_coordinates': len(
        missing_population_tract_coordinates
    ),
    'population_with_coordinates': population_eligible_tracts['Pop20'].sum(),
    'population_without_coordinates': (
        missing_population_tract_coordinates['Pop20'].sum()
    ),
    'population_without_coordinates_percent': percentage(
        missing_population_tract_coordinates['Pop20'].sum(),
        tract_population['Pop20'].sum()
    )
}])

print('\nOfficial 2020 tract-population coordinate coverage:')
print(population_location_diagnostic.to_string(
    index=False,
    formatters={
        'population_with_coordinates': lambda value: f'{value:,.0f}',
        'population_without_coordinates': lambda value: f'{value:,.0f}',
        'population_without_coordinates_percent': (
            lambda value: f'{value:.4f}%'
        )
    }
))

population_gap_summary = (
    tract_reconciliation.groupby(
        'capacity_metric', as_index=False
    )[[
        'capacity_gap',
        'capacity_population_fill'
    ]]
    .sum()
)
print('\nPopulation-based capacity-gap allocation:')
print(population_gap_summary.to_string(
    index=False,
    formatters={
        'capacity_gap': lambda value: f'{value:,.2f}',
        'capacity_population_fill': lambda value: f'{value:,.2f}'
    }
))

if not unresolved_population_gaps.empty:
    print('\nCapacity gaps without usable tract population:')
    print(unresolved_population_gaps.to_string(index=False))

if not overfilled_state_targets.empty:
    print('\nState-capacity goals exceeded before population allocation:')
    print(overfilled_state_targets.to_string(index=False))

print('\nMaximum final state-capacity reconciliation error:')
print(f'{maximum_final_reconciliation_error:,.10f}')

missing_grouped_tract_locations = grouped_by_tract[
    'latitude'
].isna() | grouped_by_tract['longitude'].isna()
print('\nFinal tract-location coverage:')
print(
    f'{(~missing_grouped_tract_locations).sum():,} of '
    f'{len(grouped_by_tract):,} tract rows have coordinates'
)

# Build tract-level power-plant records and merge them with powerplants.csv.
# Assign NERC regions from the Census tract internal points, rather than from
# state names, so split states such as Texas, Montana, New Mexico, and South
# Dakota are assigned geographically.
BTM_SOLAR_CARRIER = 'solar-rooftop'
BTM_BATTERY_CARRIER = 'battery-btm'
BASE_POWERPLANTS_PATH = '../resources/powerplants.csv'
BTM_POWERPLANTS_PATH = '../resources/powerplants_w_btm.csv'
NERC_SHAPES_PATH = (
    '../repo_data/geospatial/NERC_Regions/'
    'NERC_Regions_Subregions.shp'
)
NERC_REGION_ALIASES = {
    # Match the six NERC codes already used by powerplants.csv and constants.py.
    'FRCC': 'SERC',
    'SPP': 'MRO',
}
NERC_TO_INTERCONNECTION = {
    'WECC': 'western',
    'TRE': 'texas',
    'SERC': 'eastern',
    'RFC': 'eastern',
    'NPCC': 'eastern',
    'MRO': 'eastern',
}
MODELED_INTERCONNECTIONS = {'western', 'eastern', 'texas'}
NON_CONUS_STATES = {
    'AK',
    'AS',
    'GU',
    'HI',
    'MP',
    'PR',
    'VI',
}


def assign_tract_nerc_regions(tract_data):
    """Assign each located tract to a NERC region and interconnection."""
    result = tract_data.reset_index(drop=True).copy()
    result['nerc_region'] = pd.NA
    result['interconnection'] = pd.NA

    has_location = (
        result['latitude'].notna()
        & result['longitude'].notna()
    )
    located = result.loc[has_location].copy()
    if located.empty:
        return result

    nerc_shapes = gpd.read_file(NERC_SHAPES_PATH)
    if 'NAME' not in nerc_shapes.columns:
        raise ValueError('NERC polygon file does not contain a NAME column')
    if nerc_shapes.crs is None:
        raise ValueError('NERC polygon file does not define a CRS')

    nerc_shapes['nerc_region'] = (
        nerc_shapes['NAME']
        .str.extract(r'\(([A-Z]+)\)\s*$', expand=False)
        .replace(NERC_REGION_ALIASES)
    )
    invalid_nerc_shapes = ~nerc_shapes['nerc_region'].isin(
        NERC_TO_INTERCONNECTION
    )
    if invalid_nerc_shapes.any():
        invalid_names = nerc_shapes.loc[
            invalid_nerc_shapes, 'NAME'
        ].drop_duplicates().to_list()
        raise ValueError(
            'NERC polygon names could not be normalized to repository NERC '
            f'codes: {invalid_names}'
        )

    # Combine subregions before the point-in-polygon join. This also combines
    # the legacy FRCC/SPP polygons with the repository's SERC/MRO regions.
    nerc_regions = nerc_shapes[[
        'nerc_region',
        'geometry'
    ]].dissolve(by='nerc_region', as_index=False).rename(columns={
        'nerc_region': 'mapped_nerc_region'
    })

    tract_points = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(
            located['longitude'],
            located['latitude']
        ),
        crs='EPSG:4326'
    )
    tract_regions = gpd.sjoin(
        tract_points,
        nerc_regions.to_crs(tract_points.crs),
        how='left',
        predicate='intersects'
    )

    if tract_regions.index.duplicated().any():
        duplicate_tracts = tract_regions.loc[
            tract_regions.index.duplicated(keep=False), 'tract'
        ].drop_duplicates()
        raise ValueError(
            'NERC polygons assigned more than one region to '
            f'{len(duplicate_tracts):,} census tracts'
        )

    unmatched_conus = (
        tract_regions['mapped_nerc_region'].isna()
        & ~tract_regions['state'].isin(NON_CONUS_STATES)
    )
    if unmatched_conus.any():
        nearest_regions = gpd.sjoin_nearest(
            tract_points.loc[tract_regions.index[unmatched_conus]].to_crs(
                'EPSG:5070'
            ),
            nerc_regions.to_crs('EPSG:5070'),
            how='left',
            distance_col='nerc_distance_m'
        )
        nearest_regions = nearest_regions.sort_values(
            'nerc_distance_m'
        ).loc[~nearest_regions.index.duplicated()]
        tract_regions.loc[
            nearest_regions.index,
            'mapped_nerc_region'
        ] = nearest_regions['mapped_nerc_region']

        print(
            '\nAssigned census tracts outside the NERC polygons to their '
            f'nearest region: {len(nearest_regions):,} tracts; maximum '
            f'distance {nearest_regions["nerc_distance_m"].max() / 1_000:,.2f} km'
        )

    result.loc[tract_regions.index, 'nerc_region'] = (
        tract_regions['mapped_nerc_region'].to_numpy()
    )
    result.loc[tract_regions.index, 'interconnection'] = (
        tract_regions['mapped_nerc_region']
        .map(NERC_TO_INTERCONNECTION)
        .to_numpy()
    )
    return result


def initialize_btm_powerplant_rows(source, powerplant_columns):
    """Create blank power-plant rows and populate shared BTM attributes."""
    rows = pd.DataFrame(index=source.index, columns=powerplant_columns)
    tract = source['tract'].astype('string').str.zfill(11)
    plant_id = -pd.to_numeric(tract, errors='raise').astype('int64')

    rows['census_tract'] = tract
    rows['nerc_region'] = source['nerc_region'].to_numpy()
    rows['interconnection'] = source['interconnection'].to_numpy()
    rows['plant_id_eia'] = plant_id.to_numpy()
    rows['plant_name_eia'] = (
        'BTM resources census tract ' + tract
    ).to_numpy()
    rows['state'] = source['state'].to_numpy()
    rows['latitude'] = source['latitude'].to_numpy()
    rows['longitude'] = source['longitude'].to_numpy()

    rows['build_year'] = 2024
    rows['build_decade'] = '2020s'
    rows['operational_status'] = 'existing'
    rows['operational_status_code'] = 'OP'
    rows['data_maturity'] = 'final'
    rows['report_date'] = '2024-12-01'

    rows['ads_mustrun'] = False
    rows['efficiency'] = 1.0
    rows['fuel_cost'] = 0.0
    rows['fuel_cost_per_mwh'] = 0.0
    rows['fuel_cost_source'] = 'assumed_zero'
    rows['heat_rate'] = 3.412
    rows['heat_rate_source'] = 'assumed_zero_fuel'
    rows['marginal_cost'] = 0.0
    rows['minimum_load_mw'] = 0.0
    rows['min_down_time'] = 0.0
    rows['min_up_time'] = 0.0
    rows['ramp_limit_down'] = 1.0
    rows['ramp_limit_up'] = 1.0
    rows['start_fuel_cost'] = 0.0
    rows['start_fuel_mmbtu'] = 0.0
    rows['start_up_cost'] = 0.0
    rows['startup_cost_fixed'] = 0.0
    rows['vom'] = 0.0
    return rows


def build_btm_solar_rows(source, powerplant_columns):
    """Convert positive tract PV capacity from kW DC to plant rows in MW."""
    source = source.loc[source['PV_system_size_DC'].gt(0)].copy()
    rows = initialize_btm_powerplant_rows(source, powerplant_columns)
    rows['carrier'] = BTM_SOLAR_CARRIER
    rows['generator_id'] = 'BTMS'
    rows['generator_name'] = (
        rows['plant_name_eia']
        + '_'
        + rows['plant_id_eia'].astype('string')
        + '_BTMS'
    )
    rows['energy_source_code_1'] = 'SUN'
    rows['fuel_name'] = 'Solar'
    rows['fuel_type'] = 'solar'
    rows['fuel_type_code_pudl'] = 'solar'
    rows['prime_mover_code'] = 'PV'
    rows['prime_mover_name'] = 'Photovoltaic'
    rows['technology_description'] = 'Solar Photovoltaic - Rooftop'

    rows['p_nom'] = source['PV_system_size_DC'].to_numpy() / 1_000
    rows['summer_capacity_mw'] = rows['p_nom']
    rows['winter_capacity_mw'] = rows['p_nom']
    rows['derate_summer_capacity'] = rows['p_nom']
    rows['derate_winter_capacity'] = rows['p_nom']
    rows['summer_derate'] = 1.0
    rows['winter_derate'] = 1.0
    return rows


def build_btm_battery_rows(source, powerplant_columns):
    """Convert complete positive tract battery power/energy pairs to MW/MWh."""
    source = source.loc[
        source['battery_rated_capacity_kW'].gt(0)
        & source['battery_rated_capacity_kWh'].gt(0)
    ].copy()
    rows = initialize_btm_powerplant_rows(source, powerplant_columns)
    rows['carrier'] = BTM_BATTERY_CARRIER
    rows['generator_id'] = 'BTMB'
    rows['generator_name'] = (
        rows['plant_name_eia']
        + '_'
        + rows['plant_id_eia'].astype('string')
        + '_BTMB'
    )
    rows['energy_source_code_1'] = 'MWH'
    rows['fuel_name'] = 'Energy Storage'
    rows['fuel_type'] = 'battery'
    rows['fuel_type_code_pudl'] = 'other'
    rows['prime_mover_code'] = 'BA'
    rows['prime_mover_name'] = 'Energy Storage, Battery'
    rows['storage_technology_code_1'] = 'LIB'
    rows['technology_description'] = 'Batteries - Behind the Meter'

    rows['p_nom'] = (
        source['battery_rated_capacity_kW'].to_numpy() / 1_000
    )
    rows['energy_storage_capacity_mwh'] = (
        source['battery_rated_capacity_kWh'].to_numpy() / 1_000
    )
    rows['max_charge_rate_mw'] = rows['p_nom']
    rows['max_discharge_rate_mw'] = rows['p_nom']
    rows['summer_capacity_mw'] = rows['p_nom']
    rows['winter_capacity_mw'] = rows['p_nom']
    rows['derate_summer_capacity'] = rows['p_nom']
    rows['derate_winter_capacity'] = rows['p_nom']
    rows['summer_derate'] = 1.0
    rows['winter_derate'] = 1.0
    return rows


grouped_by_tract = assign_tract_nerc_regions(grouped_by_tract)

positive_btm_capacity = grouped_by_tract[capacity_cols].fillna(0).gt(0).any(
    axis=1
)
missing_nerc_assignment = (
    positive_btm_capacity
    & grouped_by_tract['nerc_region'].isna()
)
unexpected_missing_nerc_assignment = (
    missing_nerc_assignment
    & ~grouped_by_tract['state'].isin(NON_CONUS_STATES)
)
if unexpected_missing_nerc_assignment.any():
    missing_states = sorted(grouped_by_tract.loc[
        unexpected_missing_nerc_assignment, 'state'
    ].unique())
    raise ValueError(
        'Could not assign a NERC region to '
        f'{unexpected_missing_nerc_assignment.sum():,} positive-capacity '
        f'CONUS tract rows in states {missing_states}'
    )

excluded_non_conus = grouped_by_tract.loc[
    missing_nerc_assignment
    & grouped_by_tract['state'].isin(NON_CONUS_STATES)
].copy()
if not excluded_non_conus.empty:
    print('\nPositive BTM tract rows excluded outside CONUS:')
    print(excluded_non_conus.groupby('state')[capacity_cols].sum().to_string())

modeled_tract_capacity = grouped_by_tract.loc[
    grouped_by_tract['nerc_region'].isin(NERC_TO_INTERCONNECTION)
    & grouped_by_tract['interconnection'].isin(MODELED_INTERCONNECTIONS)
].copy()

battery_has_any_capacity = (
    modeled_tract_capacity['battery_rated_capacity_kW'].gt(0)
    | modeled_tract_capacity['battery_rated_capacity_kWh'].gt(0)
)
battery_has_complete_capacity = (
    modeled_tract_capacity['battery_rated_capacity_kW'].gt(0)
    & modeled_tract_capacity['battery_rated_capacity_kWh'].gt(0)
)
incomplete_battery_rows = modeled_tract_capacity.loc[
    battery_has_any_capacity & ~battery_has_complete_capacity
].copy()
if not incomplete_battery_rows.empty:
    print('\nBTM battery tracts omitted because power or energy is missing:')
    print(incomplete_battery_rows[capacity_cols].sum().to_string())

powerplants = pd.read_csv(BASE_POWERPLANTS_PATH, low_memory=False)
for new_column in ['census_tract', 'interconnection']:
    if new_column not in powerplants.columns:
        powerplants[new_column] = pd.NA
powerplant_columns = powerplants.columns.to_list()

btm_solar = build_btm_solar_rows(
    modeled_tract_capacity,
    powerplant_columns
)
btm_battery = build_btm_battery_rows(
    modeled_tract_capacity,
    powerplant_columns
)
btm_powerplants = pd.concat(
    [btm_solar, btm_battery],
    ignore_index=True
)[powerplant_columns]

merged_powerplants = pd.concat(
    [powerplants, btm_powerplants],
    ignore_index=True
)

if merged_powerplants['generator_name'].isna().any():
    raise ValueError('Merged powerplants contain a missing generator_name')
if merged_powerplants['generator_name'].duplicated().any():
    duplicate_names = merged_powerplants.loc[
        merged_powerplants['generator_name'].duplicated(keep=False),
        'generator_name'
    ].drop_duplicates()
    raise ValueError(
        'Merged powerplants contain duplicate generator names: '
        f'{duplicate_names.head().to_list()}'
    )

expected_solar_mw = modeled_tract_capacity.loc[
    modeled_tract_capacity['PV_system_size_DC'].gt(0),
    'PV_system_size_DC'
].sum() / 1_000
expected_battery_mw = modeled_tract_capacity.loc[
    battery_has_complete_capacity,
    'battery_rated_capacity_kW'
].sum() / 1_000
expected_battery_mwh = modeled_tract_capacity.loc[
    battery_has_complete_capacity,
    'battery_rated_capacity_kWh'
].sum() / 1_000

capacity_checks = {
    'BTM solar power (MW)': (btm_solar['p_nom'].sum(), expected_solar_mw),
    'BTM battery power (MW)': (
        btm_battery['p_nom'].sum(),
        expected_battery_mw
    ),
    'BTM battery energy (MWh)': (
        btm_battery['energy_storage_capacity_mwh'].sum(),
        expected_battery_mwh
    ),
}
for metric, (merged_value, expected_value) in capacity_checks.items():
    tolerance = max(1e-9, abs(expected_value) * 1e-12)
    if abs(merged_value - expected_value) > tolerance:
        raise ValueError(
            f'{metric} was not conserved during the power-plant merge: '
            f'{merged_value} versus {expected_value}'
        )

merged_powerplants.to_csv(BTM_POWERPLANTS_PATH, index=False)

print('\nMerged BTM power plants written to powerplants_w_btm.csv:')
print(pd.DataFrame([
    {
        'carrier': BTM_SOLAR_CARRIER,
        'rows': len(btm_solar),
        'power_mw': btm_solar['p_nom'].sum(),
        'energy_mwh': pd.NA,
    },
    {
        'carrier': BTM_BATTERY_CARRIER,
        'rows': len(btm_battery),
        'power_mw': btm_battery['p_nom'].sum(),
        'energy_mwh': btm_battery['energy_storage_capacity_mwh'].sum(),
    },
]).to_string(
    index=False,
    formatters={
        'power_mw': lambda value: f'{value:,.3f}',
        'energy_mwh': (
            lambda value: '' if pd.isna(value) else f'{value:,.3f}'
        ),
    }
))
