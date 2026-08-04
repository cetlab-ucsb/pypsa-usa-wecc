"""Tests for retirement handling while adding multi-horizon generators."""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from add_extra_components import (
    attach_multihorizon_existing_generators,
    split_retirement_gens,
)


@pytest.mark.parametrize("retirement_method", ["technical", "technical_eia"])
def test_technical_retirement_preserves_existing_generator_vintage(retirement_method):
    """Existing capacity must not be relabeled as a first-period investment."""
    n = pypsa.Network()
    snapshots = pd.MultiIndex.from_product(
        [[2030, 2040], [pd.Timestamp("2030-01-01")]],
        names=["period", "timestep"],
    )
    n.set_snapshots(snapshots)
    n.set_investment_periods([2030, 2040])
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "plant",
        bus="bus",
        carrier="CCGT",
        p_nom=100,
        p_nom_min=100,
        p_nom_max=500,
        p_nom_extendable=True,
        build_year=2006,
        lifetime=25,
    )
    n.generators.loc["plant", "land_region"] = "bus"
    n.generators.loc["plant", "original_build_year"] = 1980

    costs = pd.DataFrame(
        {
            "opex_fixed_per_kw": {"CCGT": 10.0},
            "lifetime": {"CCGT": 25.0},
            "annualized_capex_fom": {"CCGT": 100.0},
            "cost_recovery_period_years": {"CCGT": 30.0},
        },
    )

    split_retirement_gens(
        n,
        costs,
        {"CCGT"},
        economic=retirement_method == "economic",
    )

    existing = n.generators.loc["plant existing"]
    assert existing.p_nom == 100
    assert not existing.p_nom_extendable
    assert existing.build_year == 2006
    assert existing.lifetime == 25
    assert existing.original_build_year == 1980

    investment_template = n.generators.loc[["plant"]].copy()
    for investment_year in n.investment_periods:
        attach_multihorizon_existing_generators(
            n,
            costs,
            investment_template,
            investment_year,
        )
    n.mremove("Generator", investment_template.index)

    assert "plant" not in n.generators.index
    assert n.generators.loc["plant 2030", "p_nom"] == 0
    assert n.generators.loc["plant 2030", "build_year"] == 2030
    assert pd.isna(n.generators.loc["plant 2030", "original_build_year"])
    assert n.generators.loc["plant 2040", "build_year"] == 2040
    assert n.get_active_assets("Generator", 2030).loc["plant existing"]
    assert not n.get_active_assets("Generator", 2040).loc["plant existing"]
