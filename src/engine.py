"""Vectorized multi-decrement cohort cash flow projection and profit testing engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.assumptions import ActuarialAssumptions, load_mortality_table
from src.reserving import ReserveScheduleResult, build_reserve_schedule


@dataclass(frozen=True)
class ProfitabilityMetrics:
    """Summary profitability and capital efficiency metrics for a portfolio projection.

    Attributes
    ----------
    npv_profits : float
        Net Present Value of future distributable profits discounted at `r_d`.
    npv_gross_premiums : float
        Net Present Value of gross premium inflows discounted at `r_d`.
    profit_margin : float
        Ratio `NPV(Profits) / NPV(Gross Premiums)` expressed as a decimal (e.g. `0.115` for 11.5%).
    irr : float
        Internal Rate of Return `r*` where `NPV(Profits; r*) = 0`.
    discounted_payback_year : int | None
        First policy year `t in [1, T]` where cumulative discounted profit >= 0,
        or `None` if the portfolio does not break even within the projection horizon.
    new_business_strain : float
        Year 1 negative profit (`Profit_1`) representing initial shareholder capital strain.
    initial_policies : int
        Initial number of issued contracts `N_0`.
    npv_profit_per_policy : float
        `npv_profits / initial_policies`.
    """

    npv_profits: float
    npv_gross_premiums: float
    profit_margin: float
    irr: float
    discounted_payback_year: int | None
    new_business_strain: float
    initial_policies: int
    npv_profit_per_policy: float


@dataclass(frozen=True)
class ProjectionResult:
    """Complete output of the vectorized cohort cash flow and profit projection.

    Attributes
    ----------
    cohort_df : pd.DataFrame
        20-row annual cohort table containing decrements, cash flows, reserves,
        profit vector, profit signature, and discounted cash flows.
    metrics : ProfitabilityMetrics
        Scalar profitability and capital efficiency summary indicators.
    reserve_schedule : ReserveScheduleResult
        Policy-level `(N, T+1)` reserve and premium matrices.
    inforce_matrix : np.ndarray
        Policy-level `(N, T+1)` survival probability / active policy count matrix `l_t`.
    deaths_matrix : np.ndarray
        Policy-level `(N, T)` annual death counts `Deaths_t`.
    lapses_matrix : np.ndarray
        Policy-level `(N, T)` annual lapse/withdrawal counts `Lapses_t`.
    """

    cohort_df: pd.DataFrame
    metrics: ProfitabilityMetrics
    reserve_schedule: ReserveScheduleResult
    inforce_matrix: np.ndarray
    deaths_matrix: np.ndarray
    lapses_matrix: np.ndarray


def compute_irr(profit_vector: np.ndarray) -> float:
    """Solve for the Internal Rate of Return (IRR) of the annual profit vector.

    Finds the discount rate `r` in `[-0.50, 5.00]` such that:
        `sum_{t=1}^T Profit_t / (1 + r)^t = 0`

    Parameters
    ----------
    profit_vector : np.ndarray
        1D array of annual net profits `Profit_1 .. Profit_T`.

    Returns
    -------
    float
        Internal Rate of Return as a decimal, or `np.nan` if no sign change exists.
    """
    profits = np.asarray(profit_vector, dtype=float)
    if not (np.any(profits < 0) and np.any(profits > 0)):
        return float("nan")

    durations = np.arange(1, len(profits) + 1, dtype=float)

    def npv_at_rate(r: float) -> float:
        return float(np.sum(profits / ((1.0 + r) ** durations)))

    try:
        return float(brentq(npv_at_rate, -0.45, 5.0, xtol=1e-9, maxiter=500))
    except ValueError:
        return float("nan")


def run_cash_flow_projection(
    portfolio_df: pd.DataFrame,
    assumptions: ActuarialAssumptions | None = None,
    mortality_df: pd.DataFrame | None = None,
    reserve_schedule: ReserveScheduleResult | None = None,
) -> ProjectionResult:
    """Execute a vectorized multi-decrement cash flow projection across all policies.

    Implements the recursive cohort projection equations for `t = 1..T`:
    1. In-force recursion:
       `Deaths_t = l_{t-1} * q_{x+t-1}`
       `Lapses_t = l_{t-1} * (1 - q_{x+t-1}) * w_t`
       `l_t = l_{t-1} - Deaths_t - Lapses_t`
    2. Gross Premium Inflow: `P_t = l_{t-1} * P^{gross}`
    3. Expenses Outflow: `E_t` (Acquisition in year 1 + inflated renewal maintenance)
    4. Investment Income: `II_t = (V_{t-1} + P_t - E_t) * i`
    5. Death Claims Outflow: `C_t = l_{t-1} * q_{x+t-1} * S.A.`
    6. Surrender Payouts: `S_t = l_{t-1} * (1 - q_{x+t-1}) * w_t * CSV_t`
    7. Maturity Outflow: `M_T = l_T * S.A.` at `t = T` (for endowment)
    8. Mathematical Reserve: `V_t = l_t * {}_tV`, with `Delta V_t = V_t - V_{t-1}`
    9. Net Profit Vector: `Profit_t = P_t - E_t + II_t - C_t - S_t - Delta V_t`

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio DataFrame with `['issue_age', 'gender', 'sum_assured']`.
    assumptions : ActuarialAssumptions | None, optional
        Actuarial assumptions (defaults to Base Case `ActuarialAssumptions()`).
    mortality_df : pd.DataFrame | None, optional
        2017 CSO mortality table.
    reserve_schedule : ReserveScheduleResult | None, optional
        Pre-computed statutory reserve & premium schedule. Built automatically if `None`.

    Returns
    -------
    ProjectionResult
        Complete projection results, cohort summary table, and profitability metrics.
    """
    if assumptions is None:
        assumptions = ActuarialAssumptions()
    if mortality_df is None:
        mortality_df = load_mortality_table()

    issue_ages = portfolio_df["issue_age"].to_numpy(dtype=int)
    genders = portfolio_df["gender"].to_numpy()
    sum_assured = portfolio_df["sum_assured"].to_numpy(dtype=float)

    n_policies = len(portfolio_df)
    term = assumptions.policy_term

    # 1. Build or reuse policy-level statutory reserve and contractual premium schedules
    if reserve_schedule is None:
        reserve_schedule = build_reserve_schedule(
            sum_assured=sum_assured,
            issue_ages=issue_ages,
            genders=genders,
            assumptions=assumptions,
            mortality_df=mortality_df,
        )

    # 2. Vectorized Multi-Decrement Survival & Decrement Matrices of shape (N, T)
    q_matrix = assumptions.get_mortality_matrix(
        issue_ages=issue_ages, genders=genders, mortality_df=mortality_df
    )
    w_vector = assumptions.get_lapse_vector()  # shape (T,)

    # Annual retention probability per policy-year: p_t^(tau) = (1 - q_{x+t-1}) * (1 - w_t)
    retention_matrix = (1.0 - q_matrix) * (1.0 - w_vector[None, :])

    inforce_matrix = np.ones((n_policies, term + 1), dtype=float)
    inforce_matrix[:, 1:] = np.cumprod(retention_matrix, axis=1)

    l_start = inforce_matrix[:, :-1]  # l_{t-1} for t = 1..T, shape (N, T)
    l_end = inforce_matrix[:, 1:]  # l_t for t = 1..T, shape (N, T)

    deaths_matrix = l_start * q_matrix
    lapses_matrix = l_start * (1.0 - q_matrix) * w_vector[None, :]

    # 3. Vectorized Cash Flow Components across (N, T)
    gross_prem = reserve_schedule.gross_premium[:, None]  # (N, 1)
    sa_col = sum_assured[:, None]  # (N, 1)

    # Gross Premium Inflow: P_t = l_{t-1} * P^{gross}
    premium_matrix = l_start * gross_prem

    # Expenses Outflow: E_t
    t_idx = np.arange(1, term + 1, dtype=float)  # 1 .. T
    inflation_factors = (1.0 + assumptions.expense_inflation) ** (t_idx - 1.0)

    # Year 1 acquisition expense + Year 2..T renewal commission + inflated fixed maintenance
    var_expense_rates = np.full(term, assumptions.maint_expense_pct, dtype=float)
    var_expense_rates[0] = assumptions.acq_expense_pct

    fixed_expense_per_pol = assumptions.maint_expense_fixed * inflation_factors
    fixed_expense_per_pol[0] += assumptions.acq_expense_fixed

    expense_matrix = premium_matrix * var_expense_rates[None, :] + l_start * fixed_expense_per_pol[None, :]

    # Death Claims Outflow: C_t = l_{t-1} * q_{x+t-1} * S.A.
    death_claims_matrix = deaths_matrix * sa_col

    # Surrender Payouts: S_t = l_{t-1} * (1 - q_{x+t-1}) * w_t * CSV_t
    surrender_claims_matrix = lapses_matrix * reserve_schedule.csv_matrix

    # Maturity Outflow: M_T = l_T * S.A. at t = T (for Endowment)
    maturity_matrix = np.zeros((n_policies, term), dtype=float)
    if assumptions.product_type == "endowment":
        maturity_matrix[:, -1] = l_end[:, -1] * sum_assured

    # 4. Mathematical Reserve (V_t) & Change in Reserve (Delta V_t)
    # Cohort reserve at end of year t is l_t * {}_tV
    cohort_reserve_matrix = inforce_matrix * reserve_schedule.net_reserve_matrix  # (N, T+1)
    v_prev_matrix = cohort_reserve_matrix[:, :-1]  # V_{t-1}
    v_curr_matrix = cohort_reserve_matrix[:, 1:]  # V_t
    delta_v_matrix = v_curr_matrix - v_prev_matrix  # Delta V_t = V_t - V_{t-1}

    # Investment Income: II_t = (V_{t-1} + P_t - E_t) * i
    inv_income_matrix = (v_prev_matrix + premium_matrix - expense_matrix) * assumptions.investment_return

    # 5. Net Profit Vector: Profit_t = P_t - E_t + II_t - C_t - S_t - Delta V_t
    # Note: At t = T for Endowment, V_T = l_T * S.A. = M_T, so Delta V_T = M_T - V_{T-1},
    # which automatically accounts for the maturity benefit outflow net of reserve release.
    profit_matrix = (
        premium_matrix
        - expense_matrix
        + inv_income_matrix
        - death_claims_matrix
        - surrender_claims_matrix
        - delta_v_matrix
    )

    # 6. Aggregate across all N policies to build the 20-year Cohort Projection DataFrame
    years = np.arange(1, term + 1, dtype=int)
    inforce_start = np.sum(l_start, axis=0)
    deaths_agg = np.sum(deaths_matrix, axis=0)
    lapses_agg = np.sum(lapses_matrix, axis=0)
    inforce_end = np.sum(l_end, axis=0)

    premium_agg = np.sum(premium_matrix, axis=0)
    expense_agg = np.sum(expense_matrix, axis=0)
    inv_income_agg = np.sum(inv_income_matrix, axis=0)
    death_claims_agg = np.sum(death_claims_matrix, axis=0)
    surrender_agg = np.sum(surrender_claims_matrix, axis=0)
    maturity_agg = np.sum(maturity_matrix, axis=0)

    reserve_boy_agg = np.sum(v_prev_matrix, axis=0)
    reserve_eoy_agg = np.sum(v_curr_matrix, axis=0)
    delta_reserve_agg = np.sum(delta_v_matrix, axis=0)
    profit_vector = np.sum(profit_matrix, axis=0)
    profit_signature = profit_vector / float(n_policies)

    # Discounting at Risk Discount Rate r_d
    discount_factors = (1.0 + assumptions.risk_discount_rate) ** (-years.astype(float))
    discounted_profit = profit_vector * discount_factors
    cum_discounted_profit = np.cumsum(discounted_profit)
    discounted_premium = premium_agg * discount_factors

    cohort_df = pd.DataFrame(
        {
            "policy_year": years,
            "inforce_boy": inforce_start,
            "deaths": deaths_agg,
            "lapses": lapses_agg,
            "inforce_eoy": inforce_end,
            "gross_premium": premium_agg,
            "expenses": expense_agg,
            "investment_income": inv_income_agg,
            "death_claims": death_claims_agg,
            "surrender_payouts": surrender_agg,
            "maturity_outflow": maturity_agg,
            "reserve_boy": reserve_boy_agg,
            "reserve_eoy": reserve_eoy_agg,
            "delta_reserve": delta_reserve_agg,
            "net_profit": profit_vector,
            "profit_signature": profit_signature,
            "discount_factor": discount_factors,
            "discounted_profit": discounted_profit,
            "cum_discounted_profit": cum_discounted_profit,
        }
    )

    # 7. Compute Summary Profitability & Capital Metrics
    npv_profits = float(np.sum(discounted_profit))
    npv_gross_premiums = float(np.sum(discounted_premium))
    profit_margin = npv_profits / npv_gross_premiums if npv_gross_premiums > 0 else 0.0
    irr = compute_irr(profit_vector)

    positive_years = years[cum_discounted_profit >= 0.0]
    payback_year = int(positive_years[0]) if len(positive_years) > 0 else None

    metrics = ProfitabilityMetrics(
        npv_profits=npv_profits,
        npv_gross_premiums=npv_gross_premiums,
        profit_margin=profit_margin,
        irr=irr,
        discounted_payback_year=payback_year,
        new_business_strain=float(profit_vector[0]),
        initial_policies=n_policies,
        npv_profit_per_policy=npv_profits / float(n_policies),
    )

    return ProjectionResult(
        cohort_df=cohort_df,
        metrics=metrics,
        reserve_schedule=reserve_schedule,
        inforce_matrix=inforce_matrix,
        deaths_matrix=deaths_matrix,
        lapses_matrix=lapses_matrix,
    )
