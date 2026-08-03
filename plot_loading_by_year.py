import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pypsa
from cartopy import crs as ccrs
from shapely.geometry import LineString

scenario = 'wecc_tamu'
NETWORK = "/Users/Gabe/Documents/GitHub/pypsa-usa-wecc/workflow/results/WECC_TAMU_ALL_1_week_v2/western/networks/elec_s_call_ec_lv1.0__E.nc"
REGIONS = "/Users/Gabe/Documents/GitHub/pypsa-usa-wecc/workflow/resources/WECC_TAMU_ALL_1_week_v2/western/Geospatial/regions_onshore_s_all.geojson"
OUTDIR = Path(".")

YEARS = [2024]

os.makedirs(OUTDIR, exist_ok=True)

n = pypsa.Network(NETWORK)

regions = gpd.read_file(REGIONS)
EXPORT_CRS = regions.crs or "EPSG:4326"

def snapshots_for_year(year):
    return n.snapshots[n.snapshots.get_level_values("period") == year]


def line_loading(year):
    snapshots = snapshots_for_year(year)

    capacity = n.lines.s_nom_opt * n.lines.s_max_pu
    loading = n.lines_t.p0.loc[snapshots].abs().div(capacity, axis=1) * 100

    return loading.quantile(0.95)


def line_max_loading(year):
    snapshots = snapshots_for_year(year)

    capacity = n.lines.s_nom_opt * n.lines.s_max_pu
    loading = n.lines_t.p0.loc[snapshots].abs().div(capacity, axis=1) * 100

    return loading.max()


def link_loading(year):
    snapshots = snapshots_for_year(year)

    tx_links = n.links[n.links.carrier.isin(["AC", "DC"])].index

    capacity = n.links.loc[tx_links, "p_nom_opt"] * n.links.loc[tx_links, "p_max_pu"]
    loading = n.links_t.p0.loc[snapshots, tx_links].abs().div(capacity, axis=1) * 100

    return loading.quantile(0.95)


def link_max_loading(year):
    snapshots = snapshots_for_year(year)

    tx_links = n.links[n.links.carrier.isin(["AC", "DC"])].index

    capacity = n.links.loc[tx_links, "p_nom_opt"] * n.links.loc[tx_links, "p_max_pu"]
    loading = n.links_t.p0.loc[snapshots, tx_links].abs().div(capacity, axis=1) * 100

    return loading.max()


def branch_geometries(branches):
    bus_coords = n.buses[["x", "y"]]
    geometries = []

    for branch in branches.itertuples():
        if branch.bus0 not in bus_coords.index or branch.bus1 not in bus_coords.index:
            geometries.append(None)
            continue

        coords = [
            (bus_coords.at[branch.bus0, "x"], bus_coords.at[branch.bus0, "y"]),
            (bus_coords.at[branch.bus1, "x"], bus_coords.at[branch.bus1, "y"]),
        ]
        if any(pd.isna(value) for coord in coords for value in coord):
            geometries.append(None)
            continue

        geometries.append(LineString(coords))

    return geometries


def branch_loading_gdf(branches, component, loading, max_loading, nominal_col, max_pu_col):
    capacity = branches[nominal_col] * branches[max_pu_col]

    df = pd.DataFrame(
        {
            "name": branches.index.astype(str),
            "component": component,
            "carrier": branches["carrier"].astype(str),
            "bus0": branches["bus0"].astype(str),
            "bus1": branches["bus1"].astype(str),
            "nominal_capacity_mw": branches[nominal_col],
            "max_pu": branches[max_pu_col],
            "effective_capacity_mw": capacity,
            "loading_95_pct": loading,
            "loading_max_pct": max_loading,
        },
        index=branches.index,
    )

    gdf = gpd.GeoDataFrame(
        df.reset_index(drop=True),
        geometry=branch_geometries(branches),
        crs=EXPORT_CRS,
    )
    return gdf.dropna(subset=["geometry"])


def save_loading_gpkg(year, line_values, link_values):
    tx_links = n.links[n.links.carrier.isin(["AC", "DC"])].index
    line_max_values = line_max_loading(year)
    link_max_values = link_max_loading(year)
    loading_gdf = pd.concat(
        [
            branch_loading_gdf(
                n.lines.loc[line_values.index],
                "Line",
                line_values,
                line_max_values,
                "s_nom_opt",
                "s_max_pu",
            ),
            branch_loading_gdf(
                n.links.loc[tx_links],
                "Link",
                link_values,
                link_max_values,
                "p_nom_opt",
                "p_max_pu",
            ),
        ],
        ignore_index=True,
    )
    loading_gdf["year"] = year

    output = OUTDIR / f"loading_map_{scenario}_{year}.gpkg"
    if output.exists():
        output.unlink()

    regions.to_file(output, layer="regions", driver="GPKG")
    loading_gdf.to_file(output, layer="transmission_loading", driver="GPKG")
    print(f"Saved {len(loading_gdf):,} transmission records to {output}")


# Shared color scale for both years
all_values = []
for year in YEARS:
    all_values.append(line_loading(year))
    all_values.append(link_loading(year))

vmax = max(100, pd.concat(all_values).quantile(0.98))
norm = plt.Normalize(vmin=0, vmax=vmax)

def plot_loading(year):
    line_values = line_loading(year)
    link_values = link_loading(year)

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    n.plot(
        ax=ax,
        bus_sizes=0,
        line_widths=pd.Series(2.0, index=line_values.index),
        link_widths=0,
        line_colors=line_values,
        line_cmap="plasma",
        line_norm=norm,
        branch_components=["Line"],
        color_geomap=False,
    )

    link_collections = n.plot(
        ax=ax,
        bus_sizes=0,
        line_widths=0,
        link_widths=pd.Series(2.0, index=link_values.index),
        link_colors=link_values,
        link_cmap="plasma",
        link_norm=norm,
        branch_components=["Link"],
        color_geomap=False,
    )
    if len(link_collections) >= 2:
        link_collections[-1].set_linestyle(":")

    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])
    ax.set_title(f"95th Percentile Transmission Loading, {year}")

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label="95th percentile loading (%)")

    fig.savefig(OUTDIR / f"loading_map_{scenario}_{year}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_loading_gpkg(year, line_values, link_values)


for year in YEARS:
    plot_loading(year)


for year in YEARS:
    snapshots = snapshots_for_year(year)

    line_capacity = n.lines.s_nom_opt * n.lines.s_max_pu
    line_loading = n.lines_t.p0.loc[snapshots].abs().div(line_capacity, axis=1) * 100

    print(year)
    print("line max:", line_loading.max().max())
    print("line 95th percentile:", line_loading.quantile(0.95).max())
    print("lines ever >80%:", (line_loading.max() > 80).sum(), "out of", len(n.lines))
