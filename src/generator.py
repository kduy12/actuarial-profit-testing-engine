"""Vectorized synthetic life insurance policy portfolio generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from src.assumptions import ActuarialAssumptions, ProductType, load_mortality_table
from src.reserving import compute_gross_premium, compute_net_premium


def generate_synthetic_portfolio(
    n_policies: int = 10_000,
    seed: int = 42,
    product_type: ProductType = "endowment",
    mean_sum_assured_usd: float = 50_000.0,
    vnd_exchange_rate: float = 10_000.0,
    assumptions: ActuarialAssumptions | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic life insurance portfolio of `n_policies` contracts.

    Demographic & Benefit Specifications:
    - Issue age `x`: Truncated Normal `N(35, 8^2)` bounded in `[20, 55]`.
    - Gender: 50% Male (`'M'`), 50% Female (`'F'`).
    - Sum Assured: Log-normal distribution with mean $50,000 USD
      (equivalent to 500,000,000 VND at 10,000 VND/USD scaling, concentrated in
      the $25,000 - $100,000 USD corridor).

    Parameters
    ----------
    n_policies : int, optional
        Number of policies to generate (default 10,000).
    seed : int, optional
        Random seed for exact reproducibility (default 42).
    product_type : ProductType, optional
        `'endowment'` (20-Year Level Premium Endowment) or `'term'` (default `'endowment'`).
    mean_sum_assured_usd : float, optional
        Target mean Sum Assured in USD (default $50,000.0).
    vnd_exchange_rate : float, optional
        Scaling multiplier to express VND equivalent (default 10,000 -> 500M VND mean).
    assumptions : ActuarialAssumptions | None, optional
        Actuarial assumptions used to price initial Net and Gross Premiums.
    save_path : str | Path | None, optional
        If provided, writes the generated portfolio to CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame of synthetic policies with columns:
        `['policy_id', 'issue_age', 'gender', 'product_type', 'policy_term',
          'sum_assured', 'sum_assured_vnd', 'net_premium', 'gross_premium']`.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if assumptions is None:
        assumptions = ActuarialAssumptions(product_type=product_type)

    # 1. Issue Age: Truncated Normal N(35, 8) on [20, 55]
    mu_age, sigma_age = 35.0, 8.0
    low_age, high_age = 20.0, 55.0
    a, b = (low_age - mu_age) / sigma_age, (high_age - mu_age) / sigma_age
    continuous_ages = truncnorm.rvs(a, b, loc=mu_age, scale=sigma_age, size=n_policies, random_state=rng)
    issue_ages = np.clip(np.rint(continuous_ages).astype(int), 20, 55)

    # 2. Gender: Balanced 50% Male / 50% Female (shuffled deterministically)
    half = n_policies // 2
    genders = np.array(["M"] * half + ["F"] * (n_policies - half))
    rng.shuffle(genders)

    # 3. Sum Assured: Log-normal with mean $50,000 USD (500,000,000 VND equivalent)
    sigma_ln = 0.35
    mu_ln = np.log(mean_sum_assured_usd) - 0.5 * (sigma_ln ** 2)
    raw_sa = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_policies)
    # Normalize sample mean to match exact target mean and round to nearest $100
    sum_assured = np.round(raw_sa * (mean_sum_assured_usd / np.mean(raw_sa)), -2)
    sum_assured_vnd = np.round(sum_assured * vnd_exchange_rate, 0)

    # 4. Compute Equivalence Net Premium and Loaded Gross Premium
    mortality_df = load_mortality_table()
    q_matrix = assumptions.get_mortality_matrix(
        issue_ages=issue_ages, genders=genders, mortality_df=mortality_df
    )
    net_premium = compute_net_premium(
        sum_assured=sum_assured,
        q_matrix=q_matrix,
        valuation_rate=assumptions.valuation_interest_rate,
        product_type=product_type,
    )
    gross_premium = compute_gross_premium(
        net_premium=net_premium,
        sum_assured=sum_assured,
        assumptions=assumptions,
    )

    df = pd.DataFrame(
        {
            "policy_id": [f"POL-{i:05d}" for i in range(1, n_policies + 1)],
            "issue_age": issue_ages,
            "gender": genders,
            "product_type": product_type,
            "policy_term": assumptions.policy_term,
            "sum_assured": np.round(sum_assured, 2),
            "sum_assured_vnd": sum_assured_vnd.astype(np.int64),
            "net_premium": np.round(net_premium, 2),
            "gross_premium": np.round(gross_premium, 2),
        }
    )

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
