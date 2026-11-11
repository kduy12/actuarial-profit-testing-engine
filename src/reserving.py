"""Prospective and recursive Net/Gross Premium Reserve valuation routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.assumptions import ActuarialAssumptions, ProductType


@dataclass(frozen=True)
class ReserveScheduleResult:
    """Container for vectorized policy-level reserve and premium schedules.

    Attributes
    ----------
    net_premium : np.ndarray
        1D array of shape `(N,)` with annual level Net Premium `P^{net}` per policy.
    gross_premium : np.ndarray
        1D array of shape `(N,)` with annual level Gross Premium `P^{gross}` per policy.
    net_reserve_matrix : np.ndarray
        2D array of shape `(N, T + 1)` with per-policy Net Premium Reserve `{}_tV`
        for `t = 0, 1, ..., T`. Column 0 is `{}_0V = 0`, column `T` is `{}_TV`.
    gross_reserve_matrix : np.ndarray
        2D array of shape `(N, T + 1)` with per-policy Gross Premium Reserve `{}_tV^{gross}`
        for `t = 0, 1, ..., T`.
    csv_matrix : np.ndarray
        2D array of shape `(N, T)` with per-policy Cash Surrender Value `CSV_t`
        for policy years `t = 1, ..., T`.
    """

    net_premium: np.ndarray
    gross_premium: np.ndarray
    net_reserve_matrix: np.ndarray
    gross_reserve_matrix: np.ndarray
    csv_matrix: np.ndarray


def compute_net_premium(
    sum_assured: np.ndarray,
    q_matrix: np.ndarray,
    valuation_rate: float,
    product_type: ProductType = "endowment",
) -> np.ndarray:
    """Compute annual level Net Premium `P^{net}` via Equivalence Principle.

    Uses the Net Premium Valuation (NPV) equivalence equation consistent with
    the recursive reserve specification:
        `V_t = (V_{t-1} + P^{net}) * (1 + i) - q_{x+t-1} * S.A.`
    such that `{}_0V = 0` and `{}_TV = S.A.` (for `'endowment'`) or `{}_TV = 0`
    (for `'term'`).

    Parameters
    ----------
    sum_assured : np.ndarray
        1D array of policy Sum Assured values of shape `(N,)`.
    q_matrix : np.ndarray
        2D array of mortality rates `q_{x+t-1}` of shape `(N, T)`.
    valuation_rate : float
        Technical valuation interest rate `i` (e.g., `0.06`).
    product_type : ProductType, optional
        `'endowment'` or `'term'` (default `'endowment'`).

    Returns
    -------
    np.ndarray
        1D array of annual level Net Premiums of shape `(N,)`.
    """
    sa = np.asarray(sum_assured, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    _, term = q.shape

    v = 1.0 / (1.0 + valuation_rate)
    t_idx = np.arange(1, term + 1, dtype=float)
    v_t = v ** t_idx  # v^1 .. v^T
    annuity_due_factor = np.sum(v ** (t_idx - 1.0))  # sum_{t=1}^T v^{t-1}

    pv_death_benefits = sa * np.sum(q * v_t[None, :], axis=1)
    if product_type == "endowment":
        pv_maturity_benefit = sa * (v ** term)
    elif product_type == "term":
        pv_maturity_benefit = np.zeros_like(sa)
    else:
        raise ValueError(f"Unsupported product_type: {product_type}")

    return (pv_death_benefits + pv_maturity_benefit) / annuity_due_factor


def compute_gross_premium(
    net_premium: np.ndarray,
    sum_assured: np.ndarray,
    assumptions: ActuarialAssumptions,
) -> np.ndarray:
    """Calculate commercial Gross Premium `P^{gross}` from Net Premium and expense loadings.

    Incorporates acquisition commissionamortization, renewal expense loading,
    fixed per-policy expense recovery, and profit/contingency margin.

    Parameters
    ----------
    net_premium : np.ndarray
        1D array of Net Premiums of shape `(N,)`.
    sum_assured : np.ndarray
        1D array of Sum Assured values of shape `(N,)`.
    assumptions : ActuarialAssumptions
        Actuarial assumption set.

    Returns
    -------
    np.ndarray
        1D array of annual Gross Premiums of shape `(N,)`.
    """
    p_net = np.asarray(net_premium, dtype=float)
    sa = np.asarray(sum_assured, dtype=float)

    v = 1.0 / (1.0 + assumptions.valuation_interest_rate)
    t_idx = np.arange(1, assumptions.policy_term + 1, dtype=float)
    annuity_due = float(np.sum(v ** (t_idx - 1.0)))

    # Amortized fixed acquisition and inflated maintenance expense per policy
    infl_factors = (1.0 + assumptions.expense_inflation) ** (t_idx - 1.0)
    pv_fixed_expenses = assumptions.acq_expense_fixed + float(
        np.sum(assumptions.maint_expense_fixed * infl_factors * (v ** (t_idx - 1.0)))
    )
    annualized_fixed = pv_fixed_expenses / annuity_due

    # Commercial gross premium with explicit expense + profit loading
    loaded_premium = (p_net + annualized_fixed) * (1.0 + assumptions.gross_premium_loading)
    return loaded_premium


def compute_recursive_reserves(
    sum_assured: np.ndarray,
    q_matrix: np.ndarray,
    net_premium: np.ndarray,
    valuation_rate: float,
) -> np.ndarray:
    """Compute Net Premium Reserves `{}_tV` via forward recursive accumulation.

    Implements the recursion:
        `V_t = (V_{t-1} + P^{net}) * (1 + i) - q_{x+t-1} * S.A.`
    starting from `{}_0V = 0` for `t = 1, ..., T`.

    Parameters
    ----------
    sum_assured : np.ndarray
        1D array of shape `(N,)` with policy Sum Assured.
    q_matrix : np.ndarray
        2D array of shape `(N, T)` with mortality rates `q_{x+t-1}`.
    net_premium : np.ndarray
        1D array of shape `(N,)` with annual Net Premium `P^{net}`.
    valuation_rate : float
        Technical valuation interest rate `i`.

    Returns
    -------
    np.ndarray
        2D array of shape `(N, T + 1)` containing `{}_tV` for `t = 0, 1, ..., T`.
    """
    sa = np.asarray(sum_assured, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    p_net = np.asarray(net_premium, dtype=float)
    n_policies, term = q.shape

    reserves = np.zeros((n_policies, term + 1), dtype=float)
    growth = 1.0 + valuation_rate

    for t in range(1, term + 1):
        reserves[:, t] = (reserves[:, t - 1] + p_net) * growth - q[:, t - 1] * sa

    return reserves


def compute_prospective_reserves(
    sum_assured: np.ndarray,
    q_matrix: np.ndarray,
    net_premium: np.ndarray,
    valuation_rate: float,
    product_type: ProductType = "endowment",
) -> np.ndarray:
    """Compute Net Premium Reserves `{}_tV` prospectively at each duration `t = 0..T`.

    At duration `t`, `{}_tV` equals the present value of future benefits minus
    the present value of future net premiums over the remaining `T - t` years:
        `V_t = v^{T-t} * B_T + S.A. * sum_{m=1}^{T-t} v^m q_{x+t+m-1} - P^{net} * sum_{m=1}^{T-t} v^{m-1}`

    Parameters
    ----------
    sum_assured : np.ndarray
        1D array of shape `(N,)` with policy Sum Assured.
    q_matrix : np.ndarray
        2D array of shape `(N, T)` with mortality rates `q_{x+t-1}`.
    net_premium : np.ndarray
        1D array of shape `(N,)` with annual Net Premium `P^{net}`.
    valuation_rate : float
        Technical valuation interest rate `i`.
    product_type : ProductType, optional
        `'endowment'` or `'term'` (default `'endowment'`).

    Returns
    -------
    np.ndarray
        2D array of shape `(N, T + 1)` containing prospective `{}_tV` for `t = 0..T`.
    """
    sa = np.asarray(sum_assured, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    p_net = np.asarray(net_premium, dtype=float)
    n_policies, term = q.shape

    reserves = np.zeros((n_policies, term + 1), dtype=float)
    v = 1.0 / (1.0 + valuation_rate)
    terminal_benefit = sa if product_type == "endowment" else np.zeros_like(sa)

    for t in range(term):
        rem = term - t
        m_idx = np.arange(1, rem + 1, dtype=float)
        v_m = v ** m_idx
        pv_rem_deaths = sa * np.sum(q[:, t:] * v_m[None, :], axis=1)
        pv_rem_maturity = terminal_benefit * (v ** rem)
        pv_rem_prems = p_net * float(np.sum(v ** (m_idx - 1.0)))
        reserves[:, t] = pv_rem_deaths + pv_rem_maturity - pv_rem_prems

    reserves[:, term] = terminal_benefit
    return reserves


def compute_gross_premium_reserves(
    sum_assured: np.ndarray,
    q_matrix: np.ndarray,
    gross_premium: np.ndarray,
    assumptions: ActuarialAssumptions,
) -> np.ndarray:
    """Compute prospective Gross Premium Reserves (GPV) accounting for future expenses.

    Parameters
    ----------
    sum_assured : np.ndarray
        1D array of shape `(N,)` with policy Sum Assured.
    q_matrix : np.ndarray
        2D array of shape `(N, T)` with mortality rates `q_{x+t-1}`.
    gross_premium : np.ndarray
        1D array of shape `(N,)` with annual Gross Premium `P^{gross}`.
    assumptions : ActuarialAssumptions
        Actuarial assumption parameters.

    Returns
    -------
    np.ndarray
        2D array of shape `(N, T + 1)` with Gross Premium Reserves for `t = 0..T`.
    """
    sa = np.asarray(sum_assured, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    p_gross = np.asarray(gross_premium, dtype=float)
    n_policies, term = q.shape

    gpv = np.zeros((n_policies, term + 1), dtype=float)
    v = 1.0 / (1.0 + assumptions.valuation_interest_rate)
    terminal_benefit = sa if assumptions.product_type == "endowment" else np.zeros_like(sa)

    for t in range(term):
        rem = term - t
        m_idx = np.arange(1, rem + 1, dtype=float)
        v_m = v ** m_idx
        v_m_due = v ** (m_idx - 1.0)
        future_t = np.arange(t + 1, term + 1, dtype=float)

        pv_rem_deaths = sa * np.sum(q[:, t:] * v_m[None, :], axis=1)
        pv_rem_maturity = terminal_benefit * (v ** rem)

        # Future expenses at durations t+1 .. T
        infl = (1.0 + assumptions.expense_inflation) ** (future_t - 1.0)
        fixed_exp = assumptions.maint_expense_fixed * infl
        pct_exp = np.full(rem, assumptions.maint_expense_pct, dtype=float)
        if t == 0:
            fixed_exp[0] += assumptions.acq_expense_fixed
            pct_exp[0] += assumptions.acq_expense_pct

        pv_rem_expenses = p_gross * float(np.sum(pct_exp * v_m_due)) + float(
            np.sum(fixed_exp * v_m_due)
        )
        pv_rem_gross_prems = p_gross * float(np.sum(v_m_due))
        gpv[:, t] = pv_rem_deaths + pv_rem_maturity + pv_rem_expenses - pv_rem_gross_prems

    gpv[:, term] = terminal_benefit
    return gpv


def build_reserve_schedule(
    sum_assured: np.ndarray,
    issue_ages: np.ndarray,
    genders: np.ndarray,
    assumptions: ActuarialAssumptions,
    mortality_df=None,
) -> ReserveScheduleResult:
    """Build complete policy-level Net Reserve, Gross Reserve, and CSV schedules.

    Always values statutory Net Premium Reserves on the baseline valuation basis
    (`mortality_multiplier = 1.0` and `assumptions.valuation_interest_rate`) so
    that contractual premiums and statutory reserve schedules remain invariant
    under experience stress testing.

    Parameters
    ----------
    sum_assured : np.ndarray
        1D array of policy Sum Assured values of shape `(N,)`.
    issue_ages : np.ndarray
        1D array of policy issue ages of shape `(N,)`.
    genders : np.ndarray
        1D array of policy gender codes (`'M'` or `'F'`) of shape `(N,)`.
    assumptions : ActuarialAssumptions
        Actuarial assumptions configuration.
    mortality_df : pd.DataFrame | None, optional
        Optional pre-loaded 2017 CSO mortality table.

    Returns
    -------
    ReserveScheduleResult
        Populated schedules for Net Premium, Gross Premium, Net Reserves, Gross
        Reserves, and Cash Surrender Values.
    """
    # Use baseline mortality (multiplier = 1.0) for locked-in pricing & statutory valuation
    valuation_assumptions = ActuarialAssumptions(
        policy_term=assumptions.policy_term,
        product_type=assumptions.product_type,
        investment_return=assumptions.valuation_interest_rate,
        valuation_interest_rate=assumptions.valuation_interest_rate,
        risk_discount_rate=assumptions.risk_discount_rate,
        acq_expense_pct=assumptions.acq_expense_pct,
        acq_expense_fixed=assumptions.acq_expense_fixed,
        maint_expense_pct=assumptions.maint_expense_pct,
        maint_expense_fixed=assumptions.maint_expense_fixed,
        expense_inflation=0.03,
        gross_premium_loading=assumptions.gross_premium_loading,
        mortality_multiplier=1.0,
        lapse_rates=assumptions.lapse_rates,
        csv_factors=assumptions.csv_factors,
    )
    q_val = valuation_assumptions.get_mortality_matrix(
        issue_ages=issue_ages, genders=genders, mortality_df=mortality_df
    )
    net_prem = compute_net_premium(
        sum_assured=sum_assured,
        q_matrix=q_val,
        valuation_rate=valuation_assumptions.valuation_interest_rate,
        product_type=valuation_assumptions.product_type,
    )
    gross_prem = compute_gross_premium(
        net_premium=net_prem,
        sum_assured=sum_assured,
        assumptions=valuation_assumptions,
    )
    net_reserves = compute_prospective_reserves(
        sum_assured=sum_assured,
        q_matrix=q_val,
        net_premium=net_prem,
        valuation_rate=valuation_assumptions.valuation_interest_rate,
        product_type=valuation_assumptions.product_type,
    )
    gross_reserves = compute_gross_premium_reserves(
        sum_assured=sum_assured,
        q_matrix=q_val,
        gross_premium=gross_prem,
        assumptions=valuation_assumptions,
    )
    csv_factors = assumptions.get_csv_factor_vector()
    # Surrender value at duration t (t = 1..T) is CSV_factor_t * V_t
    csv_matrix = net_reserves[:, 1:] * csv_factors[None, :]

    return ReserveScheduleResult(
        net_premium=net_prem,
        gross_premium=gross_prem,
        net_reserve_matrix=net_reserves,
        gross_reserve_matrix=gross_reserves,
        csv_matrix=csv_matrix,
    )
