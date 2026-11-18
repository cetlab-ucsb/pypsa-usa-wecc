"""Plot generation, curtailment, and scenario summary metrics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
input_dir = script_dir / "processed_results_raw"
output_dir = script_dir / "plots" / "plots_v_11_12"

# Label, folder, color. Baselines and 10% flexibility cases have bars.
scenarios = [
    ('2024 baseline', 'PCM_2024_BASELINE_REFERENCE_OT_10_16', '#777777'),
    ('2032 ind-low', 'PCM_2032_INDELEC_LOW_DC_NO_FLEX_2026_11_10', '#f1b6da'),
    ('2032 ind-medium', 'PCM_2032_INDELEC_MEDIUM_DC_NO_FLEX_2026_11_10', '#df65b0'),
    ('2032 ind-high', 'PCM_2032_INDELEC_HIGH_DC_NO_FLEX_2026_11_10', '#c51b8a'),
    ('2032 low — no flex', 'PCM_2032_GROWTH_LOW_DC_NO_FLEX_2026_11_10', '#4c78a8'),
    ('2032 medium — no flex', 'PCM_2032_GROWTH_MEDIUM_DC_NO_FLEX_2026_11_10', '#72b7b2'),
    ('Medium — shift 4h, 10%', 'PCM_2032_GROWTH_MEDIUM_DC_DR_10PCT_4H_2026_11_10', '#ffbf79'),
    ('Medium — shift 8h, 10%', 'PCM_2032_GROWTH_MEDIUM_DC_DR_10PCT_8H_2026_11_10', '#d99435'),
    ('Medium — battery 4h, 10%', 'PCM_2032_GROWTH_MEDIUM_DC_BTM_BATTERY_10PCT_4H_2026_11_10', '#c6b0df'),
    ('Medium — battery 8h, 10%', 'PCM_2032_GROWTH_MEDIUM_DC_BTM_BATTERY_10PCT_8H_2026_11_10', '#9b80bd'),
    ('Medium — BTM gas, 10%', 'PCM_2032_GROWTH_MEDIUM_DC_BTM_GAS_10PCT_2026_11_10', '#86a95d'),
    ('2032 high — no flex', 'PCM_2032_GROWTH_HIGH_DC_NO_FLEX_2026_11_10', '#e45756'),
    ('High — shift 4h, 10%', 'PCM_2032_GROWTH_HIGH_DC_DR_10PCT_4H_2026_11_10', '#f28e2b'),
    ('High — shift 8h, 10%', 'PCM_2032_GROWTH_HIGH_DC_DR_10PCT_8H_2026_11_10', '#a85d0e'),
    ('High — battery 4h, 10%', 'PCM_2032_GROWTH_HIGH_DC_BTM_BATTERY_10PCT_4H_2026_11_10', '#8054a5'),
    ('High — battery 8h, 10%', 'PCM_2032_GROWTH_HIGH_DC_BTM_BATTERY_10PCT_8H_2026_11_10', '#8ecae6'),
    ('High — BTM gas, 10%', 'PCM_2032_GROWTH_HIGH_DC_BTM_GAS_10PCT_2026_11_10', '#477a35'),
]

# Matching 20% cases supply comparison annotations and markers.
comparison_20pct = {
    'Medium — shift 4h, 10%': 'PCM_2032_GROWTH_MEDIUM_DC_DR_20PCT_4H_2026_11_10',
    'Medium — shift 8h, 10%': 'PCM_2032_GROWTH_MEDIUM_DC_DR_20PCT_8H_2026_11_10',
    'Medium — battery 4h, 10%': 'PCM_2032_GROWTH_MEDIUM_DC_BTM_BATTERY_20PCT_4H_2026_11_10',
    'Medium — battery 8h, 10%': 'PCM_2032_GROWTH_MEDIUM_DC_BTM_BATTERY_20PCT_8H_2026_11_10',
    'Medium — BTM gas, 10%': 'PCM_2032_GROWTH_MEDIUM_DC_BTM_GAS_20PCT_2026_11_10',
    'High — shift 4h, 10%': 'PCM_2032_GROWTH_HIGH_DC_DR_20PCT_4H_2026_11_10',
    'High — battery 4h, 10%': 'PCM_2032_GROWTH_HIGH_DC_BTM_BATTERY_20PCT_4H_2026_11_10',
    'High — battery 8h, 10%': 'PCM_2032_GROWTH_HIGH_DC_BTM_BATTERY_20PCT_8H_2026_11_10',
    'High — BTM gas, 10%': 'PCM_2032_GROWTH_HIGH_DC_BTM_GAS_20PCT_2026_11_10',
}

# Keep only comparisons actually present in the raw-results directory.
comparison_20pct = {label: folder for label, folder in comparison_20pct.items()
                    if (input_dir / folder).is_dir()}

technologies = ["CCGT", "OCGT", "biomass", "coal", "geothermal", "hydro",
                "nuclear", "oil", "onwind", "solar", "waste"]


def read_metrics(folder):
    directory = input_dir / folder
    summary_file = ("wecc_generation_utilization_emissions_by_technology.csv"
                    if (directory / "wecc_generation_utilization_emissions_by_technology.csv").exists()
                    else "results_by_technology.csv")
    summary = pd.read_csv(directory / summary_file, index_col="carrier")
    generation = summary["generation_twh"].reindex(technologies).copy()
    curtailment = summary["curtailment_twh"].reindex(technologies).copy()
    if generation.isna().any() or curtailment.isna().any():
        raise ValueError(f"Missing technology metrics in {folder}")
    # Storage discharge is absent from the generator summary. Keep grid and BTM separate.
    hourly_dir = directory / "hourly"
    if not hourly_dir.exists():
        hourly_dir = directory
    dispatch = pd.read_csv(hourly_dir / "hourly_storage_unit_dispatch_mw.csv", index_col=[0, 1])
    batteries = (dispatch.columns.str.contains("battery", case=False)
                 & ~dispatch.columns.str.contains("btm", case=False))
    generation.loc["battery discharge"] = dispatch.loc[:, batteries].sum().sum() / 1e6
    btm_batteries = (dispatch.columns.str.contains("battery", case=False)
                     & dispatch.columns.str.contains("btm", case=False))
    generation.loc["BTM battery discharge"] = dispatch.loc[:, btm_batteries].sum().sum() / 1e6
    return generation, curtailment


def plot_comparison(metrics, metric_index, categories, filename, title, ylabel, note):
    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(len(categories))
    width = 0.82 / len(scenarios)
    for i, (label, folder, color) in enumerate(scenarios):
        values = metrics[folder][metric_index].reindex(categories).to_numpy()
        positions = x + (i - (len(scenarios) - 1) / 2) * width
        ax.bar(positions, values, width=width, color=color, label=label)
    ax.set_xticks(x, categories, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=14)
    # Leave headroom above the tallest bar.
    tallest_bar = max(metrics[folder][metric_index].reindex(categories).max()
                      for _, folder, _ in scenarios)
    ax.set_ylim(0, tallest_bar * 1.25 if tallest_bar > 0 else 1)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.text(0.06, 0.02, note, fontsize=10)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.06, right=0.99)
    fig.savefig(output_dir / f"{filename}.png", dpi=200)
    plt.close(fig)


def plot_generation_waterfalls(metrics, include_20pct=False):
    """Draw Medium and High panels with separate grid and BTM discharge steps."""
    from matplotlib.patches import Patch

    groups = {
        "CCGT": ["CCGT"], "OCGT": ["OCGT"], "coal": ["coal"],
        "solar": ["solar"], "wind": ["onwind"],
        "others": [t for t in technologies
                   if t not in {"CCGT", "OCGT", "coal", "solar", "onwind"}],
        "battery discharge": ["battery discharge"],
        "BTM battery discharge": ["BTM battery discharge"],
    }
    categories = list(groups)
    records = []
    figure_name = ("generation_waterfall_10pct_20pct" if include_20pct
                   else "generation_waterfall_all_scenarios")
    data_name = ("generation_waterfall_differences_10pct_20pct" if include_20pct
                 else "generation_waterfall_differences")
    panels = [(level, pct) for level in ("Medium", "High")
              for pct in ((10, 20) if include_20pct else (10,))]
    cases_per_panel = max(sum(label.startswith(f"{level} — ")
                              for label, _, _ in scenarios) for level in ("Medium", "High"))
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(6 * cases_per_panel, 28 if include_20pct else 14), sharey=True)
    for ax, (level, pct) in zip(axes, panels):
        is_bottom_panel = ax is axes[-1]
        ticks, ticklabels = [], []
        cases = [(label, folder) for label, folder, _ in scenarios
                 if label.startswith(f"{level} — ")]
        if pct == 20:
            cases = [(label, folder) for label, folder in cases if label in comparison_20pct]
            cases = [(label.replace("10%", "20%"), comparison_20pct[label])
                     for label, _ in cases]
        for case_index, (label, folder) in enumerate(cases):
            baseline = next(f for name, f, _ in scenarios if name == f"2032 {level.lower()} — no flex")
            base_raw = metrics[baseline][0]
            flexible_raw = metrics[folder][0]
            base = np.array([base_raw.loc[members].sum() for members in groups.values()])
            flexible = np.array([flexible_raw.loc[members].sum() for members in groups.values()])
            if not np.isfinite(base_raw.to_numpy()).all() or not np.isfinite(flexible_raw.to_numpy()).all():
                raise ValueError(f"Missing or non-finite generation in {folder}")
            delta = flexible - base
            cumulative = np.cumsum(delta)
            starts = np.r_[0.0, cumulative[:-1]]
            if not np.isclose(cumulative[-1], flexible_raw.sum() - base_raw.sum()):
                raise ValueError(f"Generation waterfall does not reconcile for {folder}")
            x = case_index * (len(categories) + 2) + np.arange(len(categories))
            ax.bar(x, np.abs(delta), bottom=np.minimum(starts, cumulative),
                   color=np.where(delta >= 0, "#278578", "#d76b42"), width=0.7)
            for position, endpoint in zip(x, cumulative):
                ax.plot([position + 0.35, position + 0.65], [endpoint, endpoint],
                        color="#999999", linewidth=0.8)
            ax.bar(x[-1] + 1, cumulative[-1], color="#46566b", width=0.7)
            ticks.extend([*x, x[-1] + 1])
            ticklabels.extend(categories + ["Net change"])
            ax.text((x[0] + x[-1] + 1) / 2,
                    1.025, label.replace(" — ", "\n"),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=14)
            if case_index:
                ax.axvline(x[0] - 1, color="#dddddd", linewidth=0.8)
            for technology, b, v, change in zip(categories, base, flexible, delta):
                records.append({"scenario": label, "scenario_folder": folder,
                                "baseline_folder": baseline, "technology": technology,
                                "baseline_twh": b, "flexibility_twh": v, "difference_twh": change})
        ax.axhline(0, color="#555555", linewidth=0.8)
        if is_bottom_panel:
            ax.set_xticks(ticks, ticklabels, rotation=65, ha="right",
                          fontsize=16 if include_20pct else 12,
                          fontweight="bold" if include_20pct else "normal")
        else:
            ax.set_xticks([])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.01, y=0.15)
    if include_20pct:
        # Apply after all panels are drawn so the shared lower limit autoscales.
        axes[-1].set_ylim(top=1.5)
    fig.supylabel("Cumulative generation / discharge change (TWh)", x=0.015, fontsize=16)
    fig.legend(handles=[Patch(color="#278578", label="Increase"),
                       Patch(color="#d76b42", label="Decrease"),
                       Patch(color="#46566b", label="Net change")], frameon=False, ncol=3, loc="upper center")
    fig.text(0.05, 0.025, "Each waterfall starts at zero. Changes = flexibility − matching no-flex case. Grid and BTM battery discharge are separate; discharge is not net generation. Others = biomass, geothermal, hydro, nuclear, oil, waste.", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94 if include_20pct else 0.90,
                        bottom=0.14 if include_20pct else 0.20, hspace=0.25)
    fig.savefig(output_dir / f"{figure_name}.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output_dir / f"{data_name}.csv", index=False)


def percent_change(base, comparison):
    if base == 0:
        return "n/a"
    change = (comparison - base) / base * 100
    return f"{change:+.3f}%" if abs(change) >= 0.0005 else "0.000%"


def read_emissions(folder):
    directory = input_dir / folder
    filename = ("wecc_generation_utilization_emissions_by_technology.csv"
                if (directory / "wecc_generation_utilization_emissions_by_technology.csv").exists()
                else "results_by_technology.csv")
    return pd.read_csv(directory / filename, index_col="carrier")["emissions_mtco2"].dropna()


def read_summary(folder):
    directory = input_dir / folder
    filename = ("wecc_overall_generation_demand_emissions_and_lmp.csv"
                if (directory / "wecc_overall_generation_demand_emissions_and_lmp.csv").exists()
                else "wecc_overall_results.csv")
    frame = pd.read_csv(directory / filename, index_col=0)
    if filename == "wecc_overall_results.csv":
        def parse_number_or_text(value):
            try:
                return pd.to_numeric(value)
            except (TypeError, ValueError):
                return value

        return frame.iloc[:, 0].map(parse_number_or_text)
    return frame.iloc[0]


def format_scenario_axis(ax):
    labels = [label.replace(" — ", "\n").replace(", 10%", "\n10%")
              .replace("2032 ind-", "2032\nind-")
              .replace("2024 baseline", "2024\nbaseline") for label, _, _ in scenarios]
    ax.set_xticks(np.arange(len(scenarios)), labels, fontsize=10)
    # Separate baseline, low, medium (including flexibility), and high.
    for i, (label, _, _) in enumerate(scenarios):
        if i and label in ("2032 low — no flex", "2032 medium — no flex", "2032 high — no flex"):
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_summary_bars(ax, summaries, column, ylabel, title):
    """Draw a summary metric on an existing axis."""
    values = [summaries[folder][column] for _, folder, _ in scenarios]
    ax.bar(np.arange(len(scenarios)), values, color=[color for _, _, color in scenarios])
    for i, ((label, folder, _), value) in enumerate(zip(scenarios, values)):
        if column == "load_shedding_hours":
            text = f"{value:.0f}"
        elif column == "load_shedding_twh":
            text = f"{value:.3f}"
        else:
            text = f"{value:.4g}"
        label_height = value
        ax.annotate(text, (i, label_height), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    comparison_positions = [i for i, (label, _, _) in enumerate(scenarios) if label in comparison_20pct]
    comparison_values = [summaries[comparison_20pct[scenarios[i][0]]][column]
                         for i in comparison_positions]
    ax.scatter(comparison_positions, comparison_values, marker="D", s=55,
               color="black", edgecolors="white", linewidths=0.8, zorder=5,
               label="20% flexibility result")
    maximum = max(values + comparison_values)
    ax.set_ylim(0, maximum * 1.3 if maximum > 0 else 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=16)
    format_scenario_axis(ax)


def plot_summary_bars(summaries, column, filename, ylabel, title):
    fig, ax = plt.subplots(figsize=(22, 8))
    draw_summary_bars(ax, summaries, column, ylabel, title)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.06, 0.025, "Black diamonds: matching 20% flexibility results.", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.90)
    fig.savefig(output_dir / f"{filename}.png", dpi=200)
    plt.close(fig)


def plot_load_shedding_panels(summaries):
    """Combine load-shedding energy and hours with a shared scenario axis."""
    fig, axes = plt.subplots(2, 1, figsize=(24, 13), sharex=True)
    draw_summary_bars(axes[0], summaries, "load_shedding_twh",
                      "Load shedding (TWh)", "WECC load shedding")
    draw_summary_bars(axes[1], summaries, "load_shedding_hours",
                      "Hours with load shedding anywhere in WECC", "WECC load-shedding hours")
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.92, hspace=0.20)
    fig.savefig(output_dir / "load_shedding_twh_and_hours.png", dpi=200)
    plt.close(fig)


def plot_emissions(summaries):
    fig, ax = plt.subplots(figsize=(22, 9))
    x = np.arange(len(scenarios))
    emissions = pd.DataFrame({
        label: read_emissions(folder)
        for label, folder, _ in scenarios
    }).fillna(0)
    emissions = emissions.loc[emissions.ne(0).any(axis=1)]
    colors = {"CCGT": "#4c78a8", "OCGT": "#9ebbd5", "coal": "#555555",
              "geothermal": "#ef9477", "oil": "#222222", "waste": "#aa8070"}
    bottom = np.zeros(len(scenarios))
    for carrier, row in emissions.iterrows():
        values = row.to_numpy()
        ax.bar(x, values, bottom=bottom, label=carrier, color=colors.get(carrier, "#999999"))
        bottom += values
    totals = np.array([summaries[folder]["emissions_mtco2"] for _, folder, _ in scenarios])
    if not np.allclose(bottom, totals):
        raise ValueError("Technology emissions do not sum to WECC emissions.")
    intensity = np.array([summaries[folder]["emission_intensity_kgco2_per_mwh"]
                          for _, folder, _ in scenarios])
    right = ax.twinx()
    right.scatter(x, intensity, color="black", marker="D", label="CO₂ intensity", zorder=5)
    right.set_ylabel("CO₂ intensity (kg CO₂/MWh)")
    # Match numeric scales and ticks while retaining each axis's distinct units.
    upper_limit = max(50, np.ceil(max(max(totals), max(intensity)) * 1.1 / 50) * 50)
    ticks = np.arange(0, upper_limit + 1, 50)
    for axis in (ax, right):
        axis.set_ylim(0, upper_limit)
        axis.set_yticks(ticks)
    ax.set_ylabel("CO₂ emissions (MtCO₂)")
    ax.set_title("WECC emissions and generation-based CO₂ intensity", loc="left", pad=16)
    format_scenario_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    line_handles, line_labels = right.get_legend_handles_labels()
    fig.legend(handles + line_handles, labels + line_labels, loc="upper center", ncol=7, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.86)
    fig.savefig(output_dir / "emissions_and_intensity.png", dpi=200)
    plt.close(fig)


def plot_demand(summaries):
    """Plot annual demand components with coincident WECC peak demand."""
    non_data_center = []
    data_center = []
    peak = []
    for _, folder, _ in scenarios:
        summary = summaries[folder]
        if "non_data_center_demand_twh" in summary.index:
            non_data_center.append(summary["non_data_center_demand_twh"])
            data_center.append(summary["shifted_data_center_demand_twh"])
            peak.append(summary["shifted_peak_demand_gw"])
            continue

        # The 2024 baseline summary predates these aggregate fields. Rebuild
        # annual demand from its zone totals and peak from coincident hourly load.
        directory = input_dir / folder
        zone_demand = pd.read_csv(directory / "demand_load_shedding_and_weighted_lmp_by_zone.csv")
        total_twh = zone_demand["total_demand_mwh"].sum() / 1e6
        data_center_twh = zone_demand["shifted_data_center_demand_mwh"].sum(min_count=1) / 1e6
        if pd.isna(data_center_twh):
            data_center_twh = 0.0
        hourly_dir = directory / "hourly"
        if not hourly_dir.exists():
            hourly_dir = directory
        hourly_load = pd.read_csv(hourly_dir / "hourly_total_load_by_zone_mw.csv")
        zone_columns = [column for column in hourly_load.columns
                        if column not in {"period", "timestep"}]
        peak_gw = hourly_load[zone_columns].sum(axis=1).max() / 1000
        non_data_center.append(total_twh - data_center_twh)
        data_center.append(data_center_twh)
        peak.append(peak_gw)

    x = np.arange(len(scenarios))
    total_demand = np.array(non_data_center) + np.array(data_center)
    fig, (demand_ax, peak_ax) = plt.subplots(
        2, 1, figsize=(22, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    demand_ax.bar(x, non_data_center, color="#4c78a8", label="Non-data-center demand")
    demand_ax.bar(x, data_center, bottom=non_data_center, color="#f1b6da",
                  label="Shifted data-center demand")
    for position, total in zip(x, total_demand):
        demand_ax.annotate(f"{total:.2f}", (position, total),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9)
    demand_ax.set_ylabel("Annual demand (TWh)")
    demand_ax.set_title("WECC annual demand", loc="left", pad=12)
    demand_ax.set_ylim(0, max(total_demand) * 1.2)
    demand_ax.grid(axis="y", alpha=0.18)
    demand_ax.set_axisbelow(True)
    demand_ax.spines[["top", "right"]].set_visible(False)

    peak_ax.bar(x, peak, color="#555555", label="WECC peak demand")
    for position, peak_gw in zip(x, peak):
        peak_ax.annotate(f"{peak_gw:.2f}", (position, peak_gw),
                         xytext=(0, 5), textcoords="offset points", ha="center",
                         va="bottom", fontsize=9)
    peak_ax.set_ylabel("WECC peak demand (GW)")
    peak_ax.set_title("WECC peak demand", loc="left", pad=10)
    peak_ax.set_ylim(0, max(peak) * 1.2)
    peak_ax.grid(axis="y", alpha=0.18)
    peak_ax.set_axisbelow(True)
    peak_ax.spines[["top", "right"]].set_visible(False)
    format_scenario_axis(peak_ax)

    handles, labels = demand_ax.get_legend_handles_labels()
    peak_handles, peak_labels = peak_ax.get_legend_handles_labels()
    fig.legend(handles + peak_handles, labels + peak_labels, loc="upper center",
               ncol=3, frameon=False)
    fig.text(0.06, 0.025,
             "Labels above bars show total demand (TWh) in the upper panel and peak demand (GW) in the lower panel.",
             fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.21, top=0.88, hspace=0.26)
    fig.savefig(output_dir / "demand_and_peak.png", dpi=200)
    plt.close(fig)


def plot_emissions_difference(summaries):
    """Compare flexibility emissions with the matching electrification no-flex case."""
    cases = [(label, folder, color) for label, folder, color in scenarios
             if label.startswith(("Medium — ", "High — "))]
    baseline_folders = [
        next(folder for name, folder, _ in scenarios
             if name == f"2032 {label.split(' — ')[0].lower()} — no flex")
        for label, _, _ in cases
    ]
    baseline = np.array([summaries[folder]["emissions_mtco2"]
                         for folder in baseline_folders])
    totals = np.array([summaries[folder]["emissions_mtco2"]
                       for _, folder, _ in cases])
    differences = totals - baseline
    comparison_positions = [i for i, (label, _, _) in enumerate(cases)
                            if label in comparison_20pct]
    comparison_folders = [comparison_20pct[cases[i][0]] for i in comparison_positions]
    comparison_differences = np.array([
        summaries[folder]["emissions_mtco2"] - baseline[i]
        for i, folder in zip(comparison_positions, comparison_folders)
    ])
    if not (np.isfinite(differences).all() and np.isfinite(comparison_differences).all()):
        raise ValueError("Missing or non-finite total emissions for baseline comparison.")
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(np.arange(len(cases)), differences,
           color=[color for _, _, color in cases])
    ax.scatter(comparison_positions, comparison_differences, marker="D", s=55,
               color="black", edgecolors="white", linewidths=0.8, zorder=5,
               label="20% flexibility result")
    ax.legend(frameon=False)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Total CO₂ emissions change from no flex (MtCO₂)")
    ax.set_title("WECC flexibility emissions change from matching no-flex cases", loc="left", pad=16)
    ax.set_xticks(np.arange(len(cases)),
                  [label.replace(" — ", "\n").replace(", 10%", "\n10%")
                   for label, _, _ in cases], fontsize=10)
    medium_count = sum(label.startswith("Medium — ") for label, _, _ in cases)
    ax.axvline(medium_count - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.text(0.06, 0.025, "Change = flexibility − matching Medium or High no-flex emissions. Negative values indicate emissions reductions.", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.90)
    fig.savefig(output_dir / "emissions_difference_from_no_flex.png", dpi=200)
    plt.close(fig)
    pd.DataFrame({
        "scenario": [label for label, _, _ in cases],
        "scenario_folder": [folder for _, folder, _ in cases],
        "baseline_folder": baseline_folders,
        "total_emissions_mtco2": totals,
        "baseline_no_flex_emissions_mtco2": baseline,
        "emissions_difference_mtco2": differences,
        "comparison_20pct_folder": [comparison_20pct.get(label) for label, _, _ in cases],
        "comparison_20pct_total_emissions_mtco2": [
            summaries[comparison_20pct[label]]["emissions_mtco2"]
            if label in comparison_20pct else np.nan for label, _, _ in cases
        ],
        "comparison_20pct_emissions_difference_mtco2": [
            comparison_differences[comparison_positions.index(i)]
            if i in comparison_positions else np.nan for i in range(len(cases))
        ],
    }).to_csv(output_dir / "emissions_difference_from_no_flex.csv", index=False)


def read_2032_zone_heatmap_data():
    """Read annual zone metrics for the 2024 baseline and 2032 cases."""
    baseline_label, baseline_folder, _ = scenarios[0]
    cases = [(baseline_label, baseline_folder)]
    for label, folder, _ in scenarios:
        if not folder.startswith("PCM_2032_"):
            continue
        cases.append((label, folder))
        if label in comparison_20pct:
            cases.append((label.replace("10%", "20%"), comparison_20pct[label]))
    if not cases:
        raise ValueError("No 2032 scenarios configured for regional heatmaps.")
    energy, hours, prices = {}, {}, {}
    expected_zones = None
    for label, folder in cases:
        directory = input_dir / folder
        if (directory / "demand_load_shedding_and_weighted_lmp_by_zone.csv").exists():
            demand = pd.read_csv(directory / "demand_load_shedding_and_weighted_lmp_by_zone.csv",
                                 index_col="zone")
            generation = pd.read_csv(directory / "generation_emissions_and_load_shedding_by_zone_and_technology.csv")
            shedding = generation.loc[generation["carrier"].eq("load")].groupby("zone")["load_shedding_twh"].sum(min_count=1)
            hours_col, lmp_col = "load_shedding_hours", "demand_weighted_lmp_usd_per_mwh"
        else:
            demand = pd.read_csv(directory / "results_by_reeds_zone.csv", index_col="zone")
            shedding = demand.set_index(demand.index).groupby(level=0)["load_shedding_gwh"].sum(min_count=1) / 1000
            hours_col, lmp_col = "load_shedding_hours", "demand_weighted_lmp_usd_per_mwh"
        if not demand.index.is_unique:
            raise ValueError(f"Duplicate zones in {folder}")
        zones = set(demand.index)
        if expected_zones is None:
            expected_zones = zones
        if zones != expected_zones or set(shedding.index) != zones:
            raise ValueError(f"Inconsistent zone coverage in {folder}")
        energy[label] = shedding.reindex(demand.index)
        hours[label] = demand[hours_col]
        prices[label] = demand[lmp_col]
        for values in (energy[label], hours[label]):
            if not np.isfinite(values.to_numpy()).all() or values.lt(0).any():
                raise ValueError(f"Invalid regional load-shedding metrics in {folder}")
        # Undefined LMP (e.g. zero demand) stays missing rather than becoming zero.
        if np.isinf(prices[label].to_numpy()).any():
            raise ValueError(f"Infinite regional LMP in {folder}")
    zone_order = sorted(expected_zones, key=lambda z: (0, int(z[1:]))
                        if z.startswith("p") and z[1:].isdigit() else (1, z))
    return tuple(pd.DataFrame(values).reindex(zone_order).rename_axis("ReEDS zone")
                 for values in (energy, hours, prices))


def draw_zone_heatmap(fig, ax, data, title, units, number_format, hide_zero=False):
    """Draw an annotated zone-by-scenario matrix with its own color scale."""
    values = data.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError(f"No finite data for {title}")
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#dddddd")
    minimum = 0 if hide_zero else min(0, float(finite.min()))
    maximum = float(finite.max())
    if maximum <= minimum:
        maximum = minimum + 1
    heatmap = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                        cmap=cmap, vmin=minimum, vmax=maximum)
    labels = [label.replace("2032 ind-", "2032\nind-")
              .replace("2032 ", "").replace(" — ", "\n")
              for label in data.columns]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(data.index)), data.index, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("ReEDS zone")
    ax.set_title(title, loc="left", fontsize=16, pad=12)
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        if not np.isfinite(value):
            text = "—"
            color = "#555555"
        else:
            text = format(value, number_format)
            if hide_zero and float(text) == 0:
                continue
            r, g, b, _ = cmap(heatmap.norm(value))
            color = "white" if 0.2126*r + 0.7152*g + 0.0722*b < 0.5 else "black"
        ax.text(col, row, text, ha="center", va="center", color=color,
                fontsize=8, fontweight="bold")
    fig.colorbar(heatmap, ax=ax, pad=0.015, fraction=0.025, label=units)


def plot_2032_zone_heatmaps():
    energy, hours, prices = read_2032_zone_heatmap_data()
    width = max(18, 0.85 * len(energy.columns))
    panel_height = max(8, 0.32 * len(energy.index))
    fig, axes = plt.subplots(2, 1, figsize=(width, 2 * panel_height + 3),
                             sharex=True, layout="constrained")
    draw_zone_heatmap(fig, axes[0], energy, "2024 baseline and 2032 load shedding by ReEDS zone", "Load shedding (TWh)", ".3f", True)
    draw_zone_heatmap(fig, axes[1], hours, "2024 baseline and 2032 load-shedding hours by ReEDS zone", "Hours with load shedding", ".0f", True)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[1].set_xlabel("Scenario — blank cell labels round to zero")
    fig.savefig(output_dir / "load_shedding_heatmap_by_reeds_zone_2032.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(width, panel_height + 3), layout="constrained")
    draw_zone_heatmap(fig, ax, prices, "2024 baseline and 2032 demand-weighted LMP by ReEDS zone",
                      "Demand-weighted LMP ($/MWh)", ".1f")
    ax.set_xlabel("Scenario — gray cells indicate unavailable LMP")
    fig.savefig(output_dir / "lmp_heatmap_by_reeds_zone_2032.png", dpi=200)
    plt.close(fig)
    for name, data in (("load_shedding_twh", energy), ("load_shedding_hours", hours),
                       ("demand_weighted_lmp", prices)):
        data.to_csv(output_dir / f"{name}_by_reeds_zone_2032.csv")


def export_plot_workbooks(metrics, summaries):
    # Include each 20% case immediately after its matching 10% case.
    rows = []
    for label, folder, _ in scenarios:
        rows.append((label, folder))
        if label in comparison_20pct:
            if label in comparison_20pct:
                rows.append((label.replace("10%", "20%"), comparison_20pct[label]))

    generation_rows, curtailment_rows, emissions_rows = [], [], []
    for label, folder in rows:
        identity = {"scenario": label, "scenario_folder": folder}
        generation, curtailment = metrics[folder]
        for technology in technologies[:2] + ["battery discharge"] + technologies[2:]:
            generation_rows.append({**identity, "technology": technology,
                                    "generation_or_discharge_twh": generation[technology]})
        for technology in technologies:
            curtailment_rows.append({**identity, "technology": technology,
                                    "curtailment_or_unused_generation_twh": curtailment[technology]})
        emissions = read_emissions(folder)
        emissions_rows.append({
            **identity,
            **{f"{carrier}_emissions_mtco2": value for carrier, value in emissions.items() if carrier != "load"},
            "total_emissions_mtco2": summaries[folder]["emissions_mtco2"],
            "emission_intensity_kgco2_per_mwh": summaries[folder]["emission_intensity_kgco2_per_mwh"],
        })
    for filename, data in [("generation_by_technology", generation_rows),
                           ("curtailment_by_technology", curtailment_rows),
                           ("emissions_and_intensity", emissions_rows)]:
        pd.DataFrame(data).to_excel(output_dir / f"{filename}.xlsx", index=False, sheet_name="Plot data")
    for filename, column in [
        ("demand_weighted_lmp", "demand_weighted_lmp_usd_per_mwh"),
        ("load_shedding_wecc_by_scenario", "load_shedding_twh"),
        ("wecc_load_shedding_hours_all_scenarios", "load_shedding_hours"),
    ]:
        pd.DataFrame([{"scenario": label, "scenario_folder": folder,
                       column: summaries[folder][column]} for label, folder in rows]
                     ).to_excel(output_dir / f"{filename}.xlsx", index=False, sheet_name="Plot data")


if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = [folder for _, folder, _ in scenarios] + list(comparison_20pct.values())
    metrics = {folder: read_metrics(folder) for folder in folders}
    plot_generation_waterfalls(metrics)
    plot_generation_waterfalls(metrics, include_20pct=True)
    plot_comparison(metrics, 0, technologies[:2] + ["battery discharge"] + technologies[2:],
                    "generation_by_technology", "WECC generation and battery discharge",
                    "Generation / battery discharge (TWh)",
                    "")
    plot_comparison(metrics, 1, technologies, "curtailment_by_technology",
                    "WECC curtailment / unused available generation by technology",
                    "Curtailment / unused available generation (TWh)",
                    "Thermal values = unused available generation.")
    summaries = {folder: read_summary(folder) for folder in folders}
    export_plot_workbooks(metrics, summaries)
    plot_emissions(summaries)
    plot_demand(summaries)
    plot_emissions_difference(summaries)
    plot_summary_bars(summaries, "demand_weighted_lmp_usd_per_mwh", "demand_weighted_lmp",
                      "Demand-weighted LMP ($/MWh)", "WECC demand-weighted LMP")
    plot_load_shedding_panels(summaries)
    plot_2032_zone_heatmaps()
    print(f"Saved plots to {output_dir}")
