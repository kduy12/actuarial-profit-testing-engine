"""Unit tests verifying cash flow balances, reserve recursion, and sensitivity logic."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.assumptions import ActuarialAssumptions, build_cso2017_table
from src.engine import run_cash_flow_projection
from src.generator import generate_synthetic_portfolio
from src.reserving import (
    build_reserve_schedule,
    compute_net_premium,
    compute_prospective_reserves,
    compute_recursive_reserves,
)
from src.sensitivity import run_sensitivity_suite


@pytest.fixture(scope="module")
def sample_portfolio():
    """Generate a reproducible 1,000-policy synthetic portfolio for fast unit testing."""
    return generate_synthetic_portfolio(n_policies=1_000, seed=42)


def test_conservation_of_policy_numbers(sample_portfolio):
    """Verify conservation of policy numbers: l_t = l_{t-1} - Deaths_t - Lapses_t."""
    assumptions = ActuarialAssumptions()
    result = run_cash_flow_projection(sample_portfolio, assumptions=assumptions)
    df = result.cohort_df

    # 1. Check cohort-level conservation in every policy year t = 1..20
    reconstructed_eoy = df["inforce_boy"] - df["deaths"] - df["lapses"]
    np.testing.assert_allclose(
        df["inforce_eoy"].to_numpy(),
        reconstructed_eoy.to_numpy(),
        rtol=1e-12,
        atol=1e-9,
    )

    # 2. Check continuity between consecutive policy years: l_{t, BOY} == l_{t-1, EOY}
    np.testing.assert_allclose(
        df["inforce_boy"].iloc[1:].to_numpy(),
        df["inforce_eoy"].iloc[:-1].to_numpy(),
        rtol=1e-12,
        atol=1e-9,
    )

    # 3. Check policy-level conservation across the (N, T) matrices
    l_prev = result.inforce_matrix[:, :-1]
    l_curr = result.inforce_matrix[:, 1:]
    np.testing.assert_allclose(
        l_curr,
        l_prev - result.deaths_matrix - result.lapses_matrix,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_lapse_and_zero_mortality_maturity_count(sample_portfolio):
    """Verify that under zero lapse and zero mortality, maturity count matches initial policies."""
    zero_decrement_assumptions = ActuarialAssumptions(
        mortality_multiplier=0.0,
        lapse_rates=tuple([0.0] * 20),
    )
    result = run_cash_flow_projection(sample_portfolio, assumptions=zero_decrement_assumptions)
    df = result.cohort_df

    initial_policies = len(sample_portfolio)
    assert np.isclose(df["inforce_boy"].iloc[0], initial_policies)
    assert np.allclose(df["deaths"].to_numpy(), 0.0)
    assert np.allclose(df["lapses"].to_numpy(), 0.0)
    assert np.isclose(df["inforce_eoy"].iloc[-1], initial_policies)
    assert np.isclose(
        df["maturity_outflow"].iloc[-1],
        sample_portfolio["sum_assured"].sum(),
    )


def test_reserve_boundary_conditions_and_recursion(sample_portfolio):
    """Verify reserve boundary conditions: {}_0V = 0 and {}_{20}V = Sum Assured for pure endowment."""
    sa = sample_portfolio["sum_assured"].to_numpy(dtype=float)
    n_pol = len(sample_portfolio)
    term = 20
    val_rate = 0.06

    # 1. Pure Endowment (zero mortality q_{x+t} = 0)
    q_zero = np.zeros((n_pol, term), dtype=float)
    p_net_pure = compute_net_premium(
        sum_assured=sa,
        q_matrix=q_zero,
        valuation_rate=val_rate,
        product_type="endowment",
    )
    res_rec_pure = compute_recursive_reserves(sa, q_zero, p_net_pure, val_rate)
    res_pro_pure = compute_prospective_reserves(sa, q_zero, p_net_pure, val_rate, "endowment")

    # Boundary conditions: {}_0V = 0 and {}_{20}V = Sum Assured
    np.testing.assert_allclose(res_rec_pure[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(res_rec_pure[:, term], sa, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(res_rec_pure, res_pro_pure, rtol=1e-10, atol=1e-8)

    # 2. Endowment with positive 2017 CSO mortality: verify recursion V_t = (V_{t-1} + P^net)(1+i) - q*SA
    assumptions = ActuarialAssumptions()
    sched = build_reserve_schedule(
        sum_assured=sa,
        issue_ages=sample_portfolio["issue_age"].to_numpy(dtype=int),
        genders=sample_portfolio["gender"].to_numpy(),
        assumptions=assumptions,
    )
    q_mat = assumptions.get_mortality_matrix(
        issue_ages=sample_portfolio["issue_age"].to_numpy(dtype=int),
        genders=sample_portfolio["gender"].to_numpy(),
    )
    res_rec = compute_recursive_reserves(
        sa, q_mat, sched.net_premium, assumptions.valuation_interest_rate
    )
    np.testing.assert_allclose(sched.net_reserve_matrix[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(sched.net_reserve_matrix[:, term], sa, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(sched.net_reserve_matrix, res_rec, rtol=1e-10, atol=1e-8)

    # 3. Level Term Life boundary condition: {}_0V = 0 and {}_{20}V = 0
    p_net_term = compute_net_premium(sa, q_mat, val_rate, product_type="term")
    res_term = compute_recursive_reserves(sa, q_mat, p_net_term, val_rate)
    np.testing.assert_allclose(res_term[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(res_term[:, term], 0.0, atol=1e-8)


def test_interest_rate_drop_strictly_reduces_profit_margin(sample_portfolio):
    """Verify that an interest rate drop strictly reduces the Profit Margin."""
    base_assumptions = ActuarialAssumptions(investment_return=0.06)
    lower_rate_assumptions = replace(base_assumptions, investment_return=0.05)

    # Lock in contractual premium & statutory reserve schedule at issue
    mortality_df = build_cso2017_table()
    base_sched = build_reserve_schedule(
        sum_assured=sample_portfolio["sum_assured"].to_numpy(dtype=float),
        issue_ages=sample_portfolio["issue_age"].to_numpy(dtype=int),
        genders=sample_portfolio["gender"].to_numpy(),
        assumptions=base_assumptions,
        mortality_df=mortality_df,
    )

    base_res = run_cash_flow_projection(
        sample_portfolio,
        assumptions=base_assumptions,
        mortality_df=mortality_df,
        reserve_schedule=base_sched,
    )
    shock_res = run_cash_flow_projection(
        sample_portfolio,
        assumptions=lower_rate_assumptions,
        mortality_df=mortality_df,
        reserve_schedule=base_sched,
    )

    assert shock_res.metrics.profit_margin < base_res.metrics.profit_margin
    assert shock_res.metrics.npv_profits < base_res.metrics.npv_profits


def test_profit_equation_identity_and_sensitivity_suite(sample_portfolio):
    """Verify exact profit accounting identity and scenario manager execution."""
    summary_df, results = run_sensitivity_suite(sample_portfolio)
    assert len(summary_df) == 6
    base_df = results["1. Base Case"].cohort_df

    reconstructed_profit = (
        base_df["gross_premium"]
        - base_df["expenses"]
        + base_df["investment_income"]
        - base_df["death_claims"]
        - base_df["surrender_payouts"]
        - base_df["delta_reserve"]
    )
    np.testing.assert_allclose(
        base_df["net_profit"].to_numpy(),
        reconstructed_profit.to_numpy(),
        rtol=1e-12,
        atol=1e-8,
    )
