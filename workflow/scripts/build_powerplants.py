"""Function-based build of existing and proposed generator/storage resources."""
import logging
import re
from pathlib import Path
from typing import Any

import constants as const
import numpy as np
import pandas as pd
from _helpers import configure_logging

logger = logging.getLogger(__name__)

DATA_YEAR = 2024
START_DATE = f"{DATA_YEAR}-01-01"
END_DATE = f"{DATA_YEAR}-12-01"
REPORT_START = pd.Timestamp(f"{DATA_YEAR}-01-01")
# Set to False to build power-plant data for all states.
WECC_STATES_ONLY = False


def resolve_workflow_dir() -> Path:
    """Resolve the workflow directory in both script and interactive sessions."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent

    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd / "workflow", cwd.parent, cwd.parent.parent]:
        if (candidate / "repo_data").exists() and (candidate / "scripts").exists():
            return candidate
    return cwd


WORKFLOW_DIR = resolve_workflow_dir()
SCRIPT_DIR = WORKFLOW_DIR / "scripts"
REPO_DATA_DIR = WORKFLOW_DIR / "repo_data"
WECC_ADS_PATH = REPO_DATA_DIR / "WECC_ADS_public"
EIA_ADS_GENERATOR_MAPPING = WECC_ADS_PATH / "eia_ads_generator_mapping_updated.csv"
EPA_CROSSWALK_PATH = REPO_DATA_DIR / "plants" / "epa_eia_crosswalk.csv"
CEMS_PATH = REPO_DATA_DIR / "plants" / "cems_heat_rates.xlsx"

GROUP_COLS = ["plant_id_eia", "generator_id"]
WECC_STATES = [
    "AZ",
    "CA",
    "CO",
    "ID",
    "MT",
    "NE",
    "NV",
    "NM",
    "OR",
    "SD",
    "TX",
    "UT",
    "WA",
    "WY",
]
# Exclude proposed records that EIA no longer treats as active development.
EXCLUDED_PROPOSED_OPERATIONAL_STATUS_CODES = ["CN", "IP", "OT"]

YG_VALUE_COLS = [
    "plant_name_eia",
    "capacity_mw",
    "summer_capacity_mw",
    "winter_capacity_mw",
    "minimum_load_mw",
    "energy_source_code_1",
    "technology_description",
    "operational_status",
    "prime_mover_code",
    "planned_generator_retirement_date",
    "energy_storage_capacity_mwh",
    "generator_operating_date",
    "state",
    "latitude",
    "longitude",
    "current_planned_generator_operating_date",
    "operational_status_code",
    "generator_retirement_date",
    "fuel_type_code_pudl",
    "data_maturity"
]

STORAGE_VALUE_COLS = [
    "max_charge_rate_mw",
    "max_discharge_rate_mw",
    "storage_technology_code_1",
]

PLANT_VALUE_COLS = [
    "nerc_region",
    "balancing_authority_code_eia",
]

MONTHLY_COLS = [
    *GROUP_COLS,
    "report_date",
    "state",
    "operational_status",
    "unit_heat_rate_mmbtu_per_mwh",
    "fuel_cost_per_mwh",
    "fuel_cost_per_mmbtu",
]

YEARLY_VALUE_COLS_FOR_HEAT_RATES = [
    "plant_name_eia",
    "capacity_mw",
    "energy_source_code_1",
    "technology_description",
    "operational_status",
    "prime_mover_code",
    "state",
]


def pudl_file(pudl_path: str, filename: str) -> str:
    """Return a full PUDL parquet URL/path."""
    return f"{pudl_path.rstrip('/')}/{filename}"


def read_pudl_inputs(
    pudl_path: str,
    include_monthly: bool = True,
    include_storage: bool = True,
) -> dict[str, pd.DataFrame]:
    """Read the PUDL parquet inputs needed by this build."""
    yearly_generators_raw = pd.read_parquet(
        pudl_file(pudl_path, "out_eia__yearly_generators.parquet"),
        columns=[*GROUP_COLS, "report_date", *YG_VALUE_COLS],
    )
    pudl_plants_raw = pd.read_parquet(
        pudl_file(pudl_path, "core_eia860__scd_plants.parquet"),
        columns=["plant_id_eia", "report_date", *PLANT_VALUE_COLS],
    )

    raw = {
        "yearly_generators_raw": yearly_generators_raw,
        "pudl_plants_raw": pudl_plants_raw,
    }
    date_dfs = [yearly_generators_raw, pudl_plants_raw]

    if include_monthly:
        monthly_generators_raw = pd.read_parquet(
            pudl_file(pudl_path, "out_eia__monthly_generators.parquet"),
            columns=MONTHLY_COLS,
        )
        raw["monthly_generators_raw"] = monthly_generators_raw
        date_dfs.append(monthly_generators_raw)

    if include_storage:
        storage_raw = pd.read_parquet(
            pudl_file(pudl_path, "core_eia860__scd_generators_energy_storage.parquet"),
            columns=[*GROUP_COLS, "report_date", *STORAGE_VALUE_COLS],
        )
        raw["storage_raw"] = storage_raw
        date_dfs.append(storage_raw)

    for df in date_dfs:
        df["report_date"] = pd.to_datetime(df["report_date"])

    return raw


def latest_non_null_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Return each group's most recent non-null value for every value column."""
    groups = df[group_cols].drop_duplicates()
    if len(group_cols) == 1:
        index = pd.Index(groups[group_cols[0]], name=group_cols[0])
    else:
        index = pd.MultiIndex.from_frame(groups)

    latest = pd.DataFrame(index=index)
    ordered = df.sort_values(date_col, ascending=False, kind="mergesort")
    # Resolve each field independently because its latest non-null observation
    # can come from a different reporting period than neighboring fields.
    for col in value_cols:
        latest[col] = (
            ordered.loc[ordered[col].notna(), [*group_cols, col]]
            .drop_duplicates(group_cols, keep="first")
            .set_index(group_cols)[col]
        )

    return latest.reset_index()


def fill_existing_generators_from_latest(
    yearly_generators_existing: pd.DataFrame,
    yearly_generators_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Fill NA fields in existing generators from latest PUDL yearly records."""
    yearly_generators_latest = latest_non_null_by_group(
        yearly_generators_raw,
        GROUP_COLS,
        YG_VALUE_COLS,
    )

    existing = yearly_generators_existing.set_index(GROUP_COLS)
    latest = yearly_generators_latest.set_index(GROUP_COLS)
    existing[YG_VALUE_COLS] = existing[YG_VALUE_COLS].combine_first(latest[YG_VALUE_COLS])
    return existing.reset_index()


def build_existing_and_proposed_generators(
    yearly_generators_raw: pd.DataFrame,
    report_start: pd.Timestamp = REPORT_START,
    states: list[str] | None = None,
    wecc_states_only: bool = WECC_STATES_ONLY,
) -> pd.DataFrame:
    """Build the combined existing/proposed generator table before enrichment."""
    if states is None and wecc_states_only:
        states = WECC_STATES

    state_filter = (
        yearly_generators_raw["state"].isin(states)
        if states is not None
        else pd.Series(True, index=yearly_generators_raw.index)
    )
    # Existing units use the requested census snapshot.
    yearly_generators_existing = yearly_generators_raw.loc[
        (yearly_generators_raw["operational_status"] == "existing")
        & (yearly_generators_raw["report_date"] == report_start)
        & state_filter
    ].copy()

    # Proposed units may be revised after the census date, so retain the latest
    # final record for each plant/generator pair.
    yearly_generators_proposed = yearly_generators_raw.loc[
        (yearly_generators_raw["operational_status"] == "proposed")
        & (~yearly_generators_raw["operational_status_code"].isin(EXCLUDED_PROPOSED_OPERATIONAL_STATUS_CODES))
        & (yearly_generators_raw["report_date"] >= report_start)
        & state_filter
        # & (yearly_generators_raw["data_maturity"] == 'final')
    ].copy().sort_values("report_date", ascending=False, kind="mergesort")

    yearly_generators_proposed = yearly_generators_proposed.drop_duplicates(
        subset=GROUP_COLS,
        keep="first",
    )

    return (
        pd.concat(
            [yearly_generators_existing, yearly_generators_proposed],
            ignore_index=True,
        )
        .drop_duplicates(subset=GROUP_COLS, keep="first")
        .reset_index(drop=True)
    )


def build_storage(storage_raw: pd.DataFrame) -> pd.DataFrame:
    """Build latest storage attributes by plant/generator."""
    return latest_non_null_by_group(storage_raw, GROUP_COLS, STORAGE_VALUE_COLS)


def build_pudl_plants(pudl_plants_raw: pd.DataFrame) -> pd.DataFrame:
    """Build latest plant-level NERC and balancing-authority attributes."""
    return latest_non_null_by_group(pudl_plants_raw, ["plant_id_eia"], PLANT_VALUE_COLS)


def build_eia_data_operable(
    yearly_generators_existing_and_proposed: pd.DataFrame,
    storage: pd.DataFrame,
    pudl_plants: pd.DataFrame,
) -> pd.DataFrame:
    """Merge base generator records with storage and plant metadata."""
    eia_data_operable = yearly_generators_existing_and_proposed.merge(
        storage,
        on=GROUP_COLS,
        how="left",
    )
    return eia_data_operable.merge(pudl_plants, on="plant_id_eia", how="left")


def load_eia_operable_data(
    parquet_path: str,
    report_start: pd.Timestamp = REPORT_START,
    states: list[str] | None = None,
    wecc_states_only: bool = WECC_STATES_ONLY,
) -> pd.DataFrame:
    """Load operable/proposed generator data from PUDL parquet files."""
    raw = read_pudl_inputs(parquet_path, include_monthly=False)
    yearly_generators_existing_and_proposed = build_existing_and_proposed_generators(
        raw["yearly_generators_raw"],
        report_start,
        states,
        wecc_states_only,
    )
    storage = build_storage(raw["storage_raw"])
    pudl_plants = build_pudl_plants(raw["pudl_plants_raw"])
    return build_eia_data_operable(
        yearly_generators_existing_and_proposed,
        storage,
        pudl_plants,
    )


def build_heat_rates_from_raw(
    monthly_generators_raw: pd.DataFrame,
    yearly_generators_raw: pd.DataFrame,
    pudl_plants: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    states: list[str] | None = None,
    wecc_states_only: bool = WECC_STATES_ONLY,
) -> pd.DataFrame:
    """Build monthly heat-rate and fuel-cost records for temporal averaging."""
    if states is None and wecc_states_only:
        states = WECC_STATES

    monthly_state_filter = (
        monthly_generators_raw["state"].isin(states)
        if states is not None
        else pd.Series(True, index=monthly_generators_raw.index)
    )
    yearly_state_filter = (
        yearly_generators_raw["state"].isin(states)
        if states is not None
        else pd.Series(True, index=yearly_generators_raw.index)
    )
    monthly_generators_heat_rates = monthly_generators_raw.loc[
        (monthly_generators_raw["operational_status"] == "existing")
        & monthly_state_filter
        & monthly_generators_raw["report_date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )
        & monthly_generators_raw["unit_heat_rate_mmbtu_per_mwh"].notna(),
        [
            *GROUP_COLS,
            "report_date",
            "unit_heat_rate_mmbtu_per_mwh",
            "fuel_cost_per_mwh",
            "fuel_cost_per_mmbtu",
        ],
    ].copy()

    # Use generator metadata from the same annual census as the monthly
    # observations rather than allowing a later yearly record to leak backward.
    yearly_heat_rate_metadata = yearly_generators_raw.loc[
        (yearly_generators_raw["operational_status"] == "existing")
        & yearly_state_filter
        & (yearly_generators_raw["report_date"] == pd.Timestamp(start_date)),
        [*GROUP_COLS, *YEARLY_VALUE_COLS_FOR_HEAT_RATES],
    ].copy()

    heat_rates = monthly_generators_heat_rates.merge(
        yearly_heat_rate_metadata,
        on=GROUP_COLS,
        how="left",
    )
    heat_rates = heat_rates.loc[heat_rates["operational_status"] == "existing"].copy()
    heat_rates = heat_rates.merge(pudl_plants, on="plant_id_eia", how="left")
    return heat_rates.sort_values("report_date", ascending=False)


def load_heat_rates_data(
    parquet_path: str,
    start_date: str,
    end_date: str,
    states: list[str] | None = None,
    wecc_states_only: bool = WECC_STATES_ONLY,
) -> pd.DataFrame:
    """Load heat-rate and fuel-cost data from PUDL parquet files."""
    raw = read_pudl_inputs(parquet_path, include_storage=False)
    pudl_plants = build_pudl_plants(raw["pudl_plants_raw"])
    return build_heat_rates_from_raw(
        raw["monthly_generators_raw"],
        raw["yearly_generators_raw"],
        pudl_plants,
        start_date,
        end_date,
        states,
        wecc_states_only,
    )


def filter_outliers_iqr_grouped(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Filter outliers using IQR for each group."""
    filtered_groups = []
    for _, group in df.groupby(group_column):
        q1 = group[value_column].quantile(0.25)
        q3 = group[value_column].quantile(0.75)
        iqr = q3 - q1
        filtered_groups.append(
            group[
                (group[value_column] >= q1 - 1.5 * iqr)
                & (group[value_column] <= q3 + 1.5 * iqr)
            ],
        )
    if filtered_groups:
        return pd.concat(filtered_groups, ignore_index=True)
    return df.iloc[0:0].copy()


def filter_outliers_zscore(
    temporal_data: pd.DataFrame,
    target_field_name: str,
    threshold: float = 3,
) -> pd.DataFrame:
    """Filter outliers using a generator-level z-score."""
    stats = temporal_data.groupby(["generator_name"])[target_field_name].agg(["mean", "std"]).reset_index()
    stats["mean"] = stats["mean"].replace(np.inf, np.nan)
    stats = stats.dropna()
    temporal_stats = temporal_data.merge(
        stats,
        on=["generator_name"],
        how="left",
        suffixes=("", "_stats"),
    )
    temporal_stats["z_score"] = (temporal_stats[target_field_name] - temporal_stats["mean"]) / temporal_stats["std"]
    filtered_temporal = temporal_stats[np.abs(temporal_stats["z_score"]) <= threshold]
    return filtered_temporal.drop(columns=["mean", "std", "z_score"])


def merge_temporal_average(
    plants: pd.DataFrame,
    temporal_data: pd.DataFrame,
    target_field_name: str,
) -> pd.DataFrame:
    """Merge temporal average heat-rate/fuel-cost data onto plant records."""
    temporal_data = temporal_data.copy()
    temporal_data["generator_name"] = (
        temporal_data["plant_name_eia"].astype(str)
        + "_"
        + temporal_data["plant_id_eia"].astype(str)
        + "_"
        + temporal_data["generator_id"].astype(str)
    )
    # Remove generator-specific extremes first, then technology-level extremes,
    # before collapsing monthly observations to one value per generator.
    filtered_temporal = filter_outliers_zscore(temporal_data, target_field_name)
    filtered_temporal = filter_outliers_iqr_grouped(
        filtered_temporal,
        "technology_description",
        target_field_name,
    )
    temporal_average = filtered_temporal.groupby(GROUP_COLS)[target_field_name].mean().reset_index()
    temporal_average[f"{target_field_name}_source"] = "pudl_reciepts"

    if target_field_name in plants.columns:
        plants = plants.drop(columns=[target_field_name])
    return plants.merge(temporal_average, on=GROUP_COLS, how="left")


def merge_heat_rate_and_fuel_costs(
    eia_data_operable: pd.DataFrame,
    heat_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply temporal averages for heat rate and fuel costs."""
    for target_field_name in [
        "unit_heat_rate_mmbtu_per_mwh",
        "fuel_cost_per_mwh",
        "fuel_cost_per_mmbtu",
    ]:
        eia_data_operable = merge_temporal_average(
            eia_data_operable,
            heat_rates,
            target_field_name,
        )
    return eia_data_operable


def apply_cems_heat_rates(
    plants: pd.DataFrame,
    crosswalk_fn: str | Path = EPA_CROSSWALK_PATH,
    cems_fn: str | Path = CEMS_PATH,
) -> pd.DataFrame:
    """Apply CEMS-derived heat rates where CAMD units map to EIA generators."""
    cems_hr = pd.read_excel(cems_fn)[["Facility ID", "Unit ID", "Heat Input (mmBtu/MWh)"]]
    crosswalk = pd.read_csv(crosswalk_fn)[["CAMD_PLANT_ID", "CAMD_UNIT_ID", "EIA_PLANT_ID", "EIA_GENERATOR_ID"]]
    cems_hr = pd.merge(
        cems_hr,
        crosswalk,
        left_on=["Facility ID", "Unit ID"],
        right_on=["CAMD_PLANT_ID", "CAMD_UNIT_ID"],
        how="inner",
    )
    cems_hr["hr_source_cems"] = "cems"

    plants = pd.merge(
        cems_hr,
        plants,
        left_on=["EIA_PLANT_ID", "EIA_GENERATOR_ID"],
        right_on=GROUP_COLS,
        how="right",
    )
    plants = plants.rename(columns={"Heat Input (mmBtu/MWh)": "heat_rate_"})
    plants.heat_rate_ = plants.heat_rate_.fillna(plants.unit_heat_rate_mmbtu_per_mwh)
    plants.unit_heat_rate_mmbtu_per_mwh = plants.pop("heat_rate_")
    plants.hr_source_cems = plants.hr_source_cems.fillna("unit_heat_rate_mmbtu_per_mwh_source")
    plants.unit_heat_rate_mmbtu_per_mwh_source = plants.pop("hr_source_cems")
    return plants.drop(
        columns=[
            "Facility ID",
            "Unit ID",
            "CAMD_PLANT_ID",
            "CAMD_UNIT_ID",
            "EIA_PLANT_ID",
            "EIA_GENERATOR_ID",
        ],
    )


def set_tech_fuels_primer_movers(eia_data_operable: pd.DataFrame) -> pd.DataFrame:
    """Map EIA technology, fuel, and prime mover labels to model fields."""
    eia_data_operable = eia_data_operable.copy()
    eia_tech_map = pd.DataFrame(const.EIA_TECH_MAP).set_index("Technology")
    eia_fuel_map = pd.DataFrame(const.EIA_FUEL_MAP).set_index("Energy Source 1")
    eia_primemover_map = pd.DataFrame(const.EIA_PRIMEMOVER_MAP).set_index("Prime Mover")
    maps = {
        "carrier": (eia_data_operable["technology_description"], eia_tech_map["tech_type"]),
        "fuel_type": (eia_data_operable["energy_source_code_1"], eia_fuel_map["fuel_type"]),
        "fuel_name": (eia_data_operable["energy_source_code_1"], eia_fuel_map["fuel_name"]),
        "prime_mover_name": (eia_data_operable["prime_mover_code"], eia_primemover_map["prime_mover"]),
    }
    for col, (data_col, map_df) in maps.items():
        eia_data_operable[col] = data_col.map(dict(zip(map_df.index, map_df.values)))
    return eia_data_operable


def standardize_col_names(
    columns: pd.Index | list[str],
    prefix: str = "",
    suffix: str = "",
) -> list[str]:
    """Normalize ADS column names."""
    return [prefix + col.lower().replace(" ", "_").replace("(", "").replace(")", "") + suffix for col in columns]


def clean_ads_name(value: Any) -> str:
    """Normalize ADS unit names for matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(value).replace(" ", "")).lower()


def merge_ads_data(
    eia_data_operable: pd.DataFrame,
    wecc_ads_path: str | Path = WECC_ADS_PATH,
    eia_ads_generator_mapping: str | Path = EIA_ADS_GENERATOR_MAPPING,
) -> pd.DataFrame:
    """Merge WECC ADS data into prepared EIA data."""
    path_ads = Path(wecc_ads_path)
    ads_thermal = pd.read_csv(path_ads / "Thermal_General_Info.csv", skiprows=1)
    ads_thermal = ads_thermal[
        [
            "GeneratorName",
            " Turbine Type",
            "MustRun",
            "MinimumDownTime(hr)",
            "MinimumUpTime(hr)",
            "MaxUpTime(hr)",
            "RampUp Rate(MW/minute)",
            "RampDn Rate(MW/minute)",
            "Startup Cost Fixed($)",
            "StartFuel(MMBTu)",
            "Startup Time",
            "VOM Cost",
        ]
    ]
    ads_thermal.columns = standardize_col_names(ads_thermal.columns)

    ads_ioc = pd.read_csv(path_ads / "Thermal_IOCurve_Info.csv", skiprows=1).rename(
        columns={"Generator Name": "GeneratorName"},
    )
    ads_ioc = ads_ioc[["GeneratorName", "IOMaxCap(MW)", "IOMinCap(MW)", "MinInput(MMBTu)"]]
    ads_ioc.columns = standardize_col_names(ads_ioc.columns)
    ads_thermal_ioc = pd.merge(ads_thermal, ads_ioc, on="generatorname", how="left")

    ads = pd.read_csv(path_ads / "GeneratorList.csv", skiprows=2, encoding="unicode_escape")
    ads["Long Name"] = ads["Long Name"].astype(str)
    ads["Name"] = ads["Name"].apply(clean_ads_name)
    ads["Long Name"] = ads["Long Name"].apply(clean_ads_name)
    ads["SubType"] = ads["SubType"].apply(clean_ads_name)
    ads = ads.rename(
        {
            "Name": "ads_name",
            "Long Name": "ads_long_name",
            "SubType": "subtype",
            "Commission Date": "commission_date",
            "Retirement Date": "retirement_date",
            "Area Name": "balancing_area",
        },
        axis=1,
    )
    ads = ads.rename(str.lower, axis="columns")
    ads["long id"] = ads["long id"].astype(str)
    ads = ads.loc[:, ~ads.columns.isin(["save to binary", "county", "city", "zipcode", "internalid"])]

    ads_name_key_dict = dict(zip(ads["ads_name"], ads["generatorkey"]))
    ads_thermal_ioc["generator_name_alt"] = (
        ads_thermal_ioc["generatorname"]
        .str.replace(" ", "")
        .str.lower()
        .str.replace("_", "")
        .str.replace("-", "")
    )
    ads_thermal_ioc["generator_key"] = ads_thermal_ioc["generator_name_alt"].map(ads_name_key_dict)

    ads_complete = ads_thermal_ioc.merge(
        ads,
        left_on="generator_key",
        right_on="generatorkey",
        how="left",
    )
    ads_complete.columns = standardize_col_names(ads_complete.columns, prefix="ads_")
    ads_complete = ads_complete.loc[~ads_complete.ads_state.isin(["MX"])]

    eia_ads_mapper = pd.read_csv(eia_ads_generator_mapping)
    eia_ads_mapper = eia_ads_mapper.loc[
        :,
        ["generatorkey", "ads_name", "plant_id_ads", "plant_id_eia", "generator_id_ads"],
    ]
    eia_ads_mapper.columns = standardize_col_names(eia_ads_mapper.columns, prefix="mapper_")
    eia_ads_mapper = eia_ads_mapper.dropna(subset=["mapper_plant_id_eia"])
    eia_ads_mapper.mapper_plant_id_eia = eia_ads_mapper.mapper_plant_id_eia.astype(int)
    eia_ads_mapper.mapper_ads_name = eia_ads_mapper.mapper_ads_name.astype(str)
    eia_ads_mapper.mapper_generatorkey = eia_ads_mapper.mapper_generatorkey.astype(int)

    ads_complete = ads_complete.dropna(subset=["ads_generator_key"])
    ads_complete.ads_generator_key = ads_complete.ads_generator_key.astype(int)

    eia_ads_mapping = pd.merge(
        ads_complete,
        eia_ads_mapper,
        left_on="ads_generator_key",
        right_on="mapper_generatorkey",
        how="inner",
    )
    eia_ads_merged = pd.merge(
        left=eia_data_operable,
        right=eia_ads_mapping,
        left_on=GROUP_COLS,
        right_on=["mapper_plant_id_eia", "mapper_generator_id_ads"],
        how="left",
    )
    eia_ads_merged = eia_ads_merged.drop(columns=eia_ads_mapper.columns)
    eia_ads_merged = eia_ads_merged.drop(
        columns=[
            "ads_generator_name_alt",
            "ads_generator_key",
            "ads_generatorkey",
            "ads_ads_name",
            "ads_bus_id",
            "ads_bus_name",
            "ads_bus_kv",
            "ads_unit_id",
            "ads_generator_typeid",
            "ads_subtype",
            "ads_long_id",
            "ads_ads_long_name",
            "ads_state",
            "ads_btm",
            "ads_devstatus",
            "ads_retirement_date",
            "ads_commission_date",
            "ads_servicestatus",
        ],
    )
    return eia_ads_merged.drop_duplicates(subset=GROUP_COLS, keep="first")


def weighted_avg(df: pd.DataFrame, values: str, weights: str) -> float:
    """Return a weighted average while ignoring missing values."""
    valid = df[values].notna()
    if valid.sum() == 0:
        return np.nan
    return np.average(df.loc[valid, values], weights=df.loc[valid, weights])


def impute_missing_plant_data(
    plants: pd.DataFrame,
    aggregation_fields: list[str],
    data_fields: list[str],
) -> pd.DataFrame:
    """Impute missing fields with capacity-weighted group averages."""
    weighted_average_fields = [*data_fields, "p_nom"]
    weighted_averages = (
        plants.groupby(aggregation_fields, observed=False)[weighted_average_fields]
        .apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in data_fields},
            ),
            include_groups=False,
        )
        .reset_index()
    )
    plants_merged = pd.merge(
        plants.reset_index(),
        weighted_averages,
        on=aggregation_fields,
        suffixes=("", "_weighted"),
    )

    for field in data_fields:
        plants_merged[field] = plants_merged[field].fillna(plants_merged[f"{field}_weighted"])
        if field in ["fuel_cost", "heat_rate"]:
            plants_merged[f"{field}_source"] = np.where(
                plants_merged[field] == plants_merged[f"{field}_weighted"],
                "weighted_average",
                plants_merged[f"{field}_source"],
            )

    plants_merged = plants_merged.drop(columns=[f"{field}_weighted" for field in data_fields])
    return plants_merged.set_index("generator_name")


def set_derates(plants: pd.DataFrame) -> pd.DataFrame:
    """Set seasonal derate columns."""
    plants = plants.copy()
    plants["derate_summer_capacity"] = np.minimum(
        plants.summer_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )
    plants["derate_winter_capacity"] = np.minimum(
        plants.winter_capacity_mw,
        plants.ads_maxcapmw.fillna(np.inf),
    )
    plants["summer_derate"] = 1 - ((plants.p_nom - plants.derate_summer_capacity) / plants.p_nom)
    plants["winter_derate"] = 1 - ((plants.p_nom - plants.derate_winter_capacity) / plants.p_nom)
    plants.summer_derate = plants.summer_derate.clip(upper=1).clip(lower=0)
    plants.winter_derate = plants.winter_derate.clip(upper=1).clip(lower=0)
    return plants


def set_parameters(plants: pd.DataFrame) -> pd.DataFrame:
    """Set generator names, costs, commitment parameters, efficiency, and derates."""
    # plants = plants[plants.nerc_region.isin(["WECC", "TRE", "MRO", "SERC", "RFC", "NPCC"])].copy()
    plants = plants.rename(
        {
            "fuel_cost_per_mwh_source": "fuel_cost_source",
            "unit_heat_rate_mmbtu_per_mwh_source": "heat_rate_source",
        },
        axis=1,
    )
    plants["generator_name"] = (
        plants.plant_name_eia.astype(str)
        + "_"
        + plants.plant_id_eia.astype(str)
        + "_"
        + plants.generator_id.astype(str)
    )
    plants = plants.set_index("generator_name")
    plants["p_nom"] = plants.pop("capacity_mw")
    generator_operating_date = pd.to_datetime(
        plants.pop("generator_operating_date"),
        errors="coerce",
    )
    plants["build_year"] = generator_operating_date.dt.year
    plants["build_decade"] = plants.build_year.astype(str).str[:3] + "0s"
    plants["heat_rate"] = plants.pop("unit_heat_rate_mmbtu_per_mwh")
    plants["vom"] = plants.pop("ads_vom_cost")
    plants["fuel_cost"] = plants.pop("fuel_cost_per_mmbtu")

    zero_mc_fuel_types = ["solar", "wind", "hydro", "geothermal", "battery"]
    plants.loc[plants.fuel_type.isin(zero_mc_fuel_types), "fuel_cost"] = 0
    # Impute from the most specific peer group to progressively broader groups.
    for aggregation_fields in [
        ["state", "fuel_name"],
        ["balancing_authority_code_eia", "fuel_name"],
        ["nerc_region", "fuel_name"],
        ["fuel_name"],
        ["prime_mover_code"],
    ]:
        plants = impute_missing_plant_data(plants, aggregation_fields, ["fuel_cost"])
    plants.loc[plants.carrier.isin(["nuclear"]), "fuel_cost"] = np.float32(0.71)

    plants["start_fuel_mmbtu"] = plants.pop("ads_startfuelmmbtu")
    plants["startup_cost_fixed"] = plants.pop("ads_startup_cost_fixed$")
    plants["min_down_time"] = plants.pop("ads_minimumdowntimehr")
    plants["min_up_time"] = plants.pop("ads_minimumuptimehr")
    plants.loc[plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]), "start_fuel_mmbtu"] = 0
    plants.loc[plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]), "startup_cost_fixed"] = 0

    # ADS reports MW/minute; PyPSA expects per-unit change per hour.
    plants["ramp_limit_up"] = (plants.pop("ads_rampup_ratemw/minute") / plants.p_nom * 60).clip(0, 1)
    plants["ramp_limit_down"] = (plants.pop("ads_rampdn_ratemw/minute") / plants.p_nom * 60).clip(0, 1)

    uc_data_fields = [
        "startup_cost_fixed",
        "start_fuel_mmbtu",
        "min_down_time",
        "min_up_time",
        "ramp_limit_up",
        "ramp_limit_down",
        "vom",
    ]
    # Prefer technology and vintage peers, then fall back to technology alone.
    for aggregation_fields in [
        ["technology_description", "build_decade"],
        ["prime_mover_code", "build_decade"],
        ["carrier", "build_decade"],
        ["technology_description"],
        ["prime_mover_code"],
        ["carrier"],
    ]:
        plants = impute_missing_plant_data(plants, aggregation_fields, uc_data_fields)

    plants["start_fuel_cost"] = plants.start_fuel_mmbtu * plants.fuel_cost
    plants["start_up_cost"] = plants.startup_cost_fixed + plants.start_fuel_cost

    plants.loc[plants.heat_rate < 3.412, "heat_rate"] = np.nan
    plants.loc[plants.fuel_type.isin(["solar", "wind", "hydro", "battery"]), "heat_rate"] = 3.412
    # Fill thermal heat rates from regional technology peers before using
    # nationwide technology or carrier averages.
    for aggregation_fields in [
        ["nerc_region", "prime_mover_code"],
        ["nerc_region", "technology_description"],
        ["nerc_region", "prime_mover_code"],
        ["prime_mover_code"],
        ["technology_description"],
        ["carrier"],
    ]:
        plants = impute_missing_plant_data(plants, aggregation_fields, ["heat_rate"])

    plants["marginal_cost"] = plants.vom + (plants.fuel_cost * plants.heat_rate)
    plants["efficiency"] = 1 / (plants["heat_rate"] / 3.412)
    plants = set_derates(plants)
    plants["heat_rate_source"] = plants["heat_rate_source"].fillna("NA")
    plants["fuel_cost_source"] = plants["fuel_cost_source"].fillna("NA")

    if plants["heat_rate"].isna().sum() > 0:
        logger.warning("Missing %s heat rate records.", plants["heat_rate"].isna().sum())
    if plants["fuel_cost"].isna().sum() > 0:
        logger.warning("Missing %s fuel cost records.", plants["fuel_cost"].isna().sum())

    plants = plants.loc[:, ~plants.columns.str.startswith("ads_") | (plants.columns == "ads_mustrun")]
    return plants.round(4).reset_index()


def finalize_powerplants(plants: pd.DataFrame) -> pd.DataFrame:
    """Drop missing-location plants and sort rows/columns for stable output."""
    missing_locations = plants[plants.longitude.isna() | plants.latitude.isna()]
    logger.warning("Tossing out plants without locations: %s", missing_locations.shape[0])
    plants = plants[~plants.index.isin(missing_locations.index)]
    logger.info("Exporting Powerplants, with %s entries.", plants.shape[0])
    plants = plants.reindex(sorted(plants.columns), axis=1)
    return plants.sort_values("generator_name")


def build_powerplants(
    pudl_path: str,
    wecc_ads_path: str | Path,
    eia_ads_generator_mapping: str | Path,
    epa_crosswalk_path: str | Path,
    cems_path: str | Path,
    data_year: int = DATA_YEAR,
    wecc_states_only: bool = WECC_STATES_ONLY,
) -> pd.DataFrame:
    """Run the existing-and-proposed power-plant build."""
    report_start = pd.Timestamp(f"{data_year}-01-01")
    start_date = f"{data_year}-01-01"
    end_date = f"{data_year}-12-01"

    eia_data_operable = load_eia_operable_data(
        pudl_path,
        report_start,
        wecc_states_only=wecc_states_only,
    )
    heat_rates = load_heat_rates_data(
        pudl_path,
        start_date,
        end_date,
        wecc_states_only=wecc_states_only,
    )
    eia_data_operable = merge_heat_rate_and_fuel_costs(eia_data_operable, heat_rates)
    eia_data_operable = apply_cems_heat_rates(
        eia_data_operable,
        epa_crosswalk_path,
        cems_path,
    )
    eia_data_operable = set_tech_fuels_primer_movers(eia_data_operable)
    eia_ads_merged = merge_ads_data(
        eia_data_operable,
        wecc_ads_path,
        eia_ads_generator_mapping,
    )
    return finalize_powerplants(set_parameters(eia_ads_merged))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants")

    configure_logging(snakemake)

    # Preserve the new-functions implementation's data vintage. The renewable
    # weather year controls profiles and should not change the generator census.
    data_year = int(snakemake.params.get("powerplants_data_year", DATA_YEAR))
    plants = build_powerplants(
        pudl_path=snakemake.params.pudl_path,
        wecc_ads_path=snakemake.input.wecc_ads,
        eia_ads_generator_mapping=snakemake.input.eia_ads_generator_mapping,
        epa_crosswalk_path=snakemake.input.epa_crosswalk,
        cems_path=snakemake.input.cems,
        data_year=data_year,
        wecc_states_only=WECC_STATES_ONLY,
    )
    plants.to_csv(snakemake.output.powerplants, index=False)
