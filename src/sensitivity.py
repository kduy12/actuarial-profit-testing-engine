"""Sensitivity matrix, stress testing, and what-if scenario manager."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.assumptions import ActuarialAssumptions, load_mortality_table
from src.engine import ProjectionResult, run_cash_flow_projection
from src.reserving import build_reserve_schedule


def build_standard_scenarios(
    base_assumptions: ActuarialAssumptions | None = None,
) -> dict[str, ActuarialAssumptions]:
    """Construct the 6 standard actuarial stress testing scenarios.

    Scenarios:
    1. Base Case: Standard assumptions (`q` = 100% CSO 2017, `w_1`=15%, `w_2`=10%, `i`=6.0%, `i_e`=3.0%).
    2. Mortality Shock: +15% relative increase in all `q_{x+t}` (`mortality_multiplier = 1.15`).
    3. Early Lapse Shock: Year 1 and Year 2 lapse rates increase by 100% (`30.0%` and `20.0%`).
    4. Interest Rate Downward Shock: Investment yield drops by -100 bps (`5.0%`).
    5. Expense Inflation Shock: Annual expense inflation jumps from `3.0%` to `5.0%`.
    6. Combined Adverse Scenario: All 4 adverse shocks applied simultaneously.

    Parameters
    ----------
    base_assumptions : ActuarialAssumptions | None, optional
        Baseline assumptions configuration.

    Returns
    -------
    dict[str, ActuarialAssumptions]
        Ordered dictionary mapping scenario name to its `ActuarialAssumptions`.
    """
    if base_assumptions is None:
        base_assumptions = ActuarialAssumptions()

    base_lapses = list(base_assumptions.get_lapse_vector())
    shocked_lapses = tuple([0.300, 0.200] + base_lapses[2:])

    return {
        "1. Base Case": base_assumptions,
        "2. Mortality Shock (+15% q_x)": replace(
            base_assumptions,
            mortality_multiplier=1.15,
        ),
        "3. Early Lapse Shock (Y1=30%, Y2=20%)": replace(
            base_assumptions,
            lapse_rates=shocked_lapses,
        ),
        "4. Interest Rate Shock (-100 bps to 5.0%)": replace(
            base_assumptions,
            investment_return=base_assumptions.investment_return - 0.01,
        ),
        "5. Expense Inflation Shock (5.0% p.a.)": replace(
            base_assumptions,
            expense_inflation=0.05,
        ),
        "6. Combined Adverse Scenario": replace(
            base_assumptions,
            mortality_multiplier=1.15,
            lapse_rates=shocked_lapses,
            investment_return=base_assumptions.investment_return - 0.01,
            expense_inflation=0.05,
        ),
    }


def run_sensitivity_suite(
    portfolio_df: pd.DataFrame,
    base_assumptions: ActuarialAssumptions | None = None,
    mortality_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, ProjectionResult]]:
    """Run all 6 stress testing scenarios and compile a comparative summary table.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Synthetic policy portfolio DataFrame.
    base_assumptions : ActuarialAssumptions | None, optional
        Base Case assumptions.
    mortality_df : pd.DataFrame | None, optional
        2017 CSO mortality table.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, ProjectionResult]]
        - `summary_df`: DataFrame comparing profitability metrics across all scenarios.
        - `results_by_scenario`: Dictionary mapping scenario names to `ProjectionResult`.
    """
    if base_assumptions is None:
        base_assumptions = ActuarialAssumptions()
    if mortality_df is None:
        mortality_df = load_mortality_table()

    # Pre-compute contractual Gross Premium & Statutory Net Premium Reserve schedule at issue
    base_reserve_schedule = build_reserve_schedule(
        sum_assured=portfolio_df["sum_assured"].to_numpy(dtype=float),
        issue_ages=portfolio_df["issue_age"].to_numpy(dtype=int),
        genders=portfolio_df["gender"].to_numpy(),
        assumptions=base_assumptions,
        mortality_df=mortality_df,
    )

    scenarios = build_standard_scenarios(base_assumptions)
    results_by_scenario: dict[str, ProjectionResult] = {}
    rows: list[dict[str, object]] = []

    base_npv: float | None = None
    base_margin: float | None = None

    for name, scen_assumptions in scenarios.items():
        res = run_cash_flow_projection(
            portfolio_df=portfolio_df,
            assumptions=scen_assumptions,
            mortality_df=mortality_df,
            reserve_schedule=base_reserve_schedule,
        )
        results_by_scenario[name] = res
        m = res.metrics

        if base_npv is None:
            base_npv = m.npv_profits
            base_margin = m.profit_margin

        delta_npv = m.npv_profits - base_npv
        delta_npv_pct = (delta_npv / abs(base_npv) * 100.0) if base_npv != 0 else 0.0
        delta_margin_bps = (m.profit_margin - (base_margin or 0.0)) * 10_000.0

        rows.append(
            {
                "Scenario": name,
                "NPV of Profits ($)": round(m.npv_profits, 2),
                "NPV Gross Premiums ($)": round(m.npv_gross_premiums, 2),
                "Profit Margin (%)": round(m.profit_margin * 100.0, 2),
                "IRR (%)": round(m.irr * 100.0, 2) if not np.isnan(m.irr) else np.nan,
                "Payback Period (Yrs)": m.discounted_payback_year,
                "Year 1 Strain ($)": round(m.new_business_strain, 2),
                "Delta NPV ($)": round(delta_npv, 2),
                "Delta NPV (%)": round(delta_npv_pct, 2),
                "Delta Margin (bps)": round(delta_margin_bps, 1),
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df, results_by_scenario
