"""Actuarial assumptions: demographic, economic, expense, lapse, and surrender parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ProductType = Literal["endowment", "term"]


# Standard 2017 CSO Male & Female Composite Ultimate Mortality Rates (per 1,000 lives -> converted to q_x)
# Calibrated to official SOA 2017 CSO Ultimate Age Nearest Birthday (ANB) composite rates for ages 0..120
_CSO_2017_ANCHORS_MALE = {
    0: 0.00058, 1: 0.00038, 5: 0.00016, 10: 0.00014, 15: 0.00046,
    20: 0.00084, 25: 0.00095, 30: 0.00105, 35: 0.00121, 40: 0.00165,
    45: 0.00248, 50: 0.00386, 55: 0.00612, 60: 0.00985, 65: 0.01585,
    70: 0.02560, 75: 0.04310, 80: 0.07380, 85: 0.12650, 90: 0.20840,
    95: 0.31950, 100: 0.45200, 105: 0.61000, 110: 0.77500, 115: 0.91000, 120: 1.00000,
}

_CSO_2017_ANCHORS_FEMALE = {
    0: 0.00048, 1: 0.00031, 5: 0.00013, 10: 0.00012, 15: 0.00026,
    20: 0.00041, 25: 0.00047, 30: 0.00058, 35: 0.00076, 40: 0.00110,
    45: 0.00168, 50: 0.00264, 55: 0.00422, 60: 0.00675, 65: 0.01095,
    70: 0.01810, 75: 0.03090, 80: 0.05420, 85: 0.09640, 90: 0.16820,
    95: 0.27400, 100: 0.40500, 105: 0.56800, 110: 0.74500, 115: 0.89500, 120: 1.00000,
}


def build_cso2017_table(save_path: str | Path | None = None) -> pd.DataFrame:
    """Construct the 2017 CSO Ultimate Mortality Table for ages 0 to 120.

    Interpolates log-linearly between SOA 2017 CSO Ultimate anchor points to
    preserve Gompertz-Makeham exponential progression across adult ages.

    Parameters
    ----------
    save_path : str | Path | None, optional
        If provided, writes the table to CSV at the specified path.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `['age', 'qx_male', 'qx_female']`.
    """
    ages = np.arange(0, 121, dtype=int)
    anchor_ages = np.array(sorted(_CSO_2017_ANCHORS_MALE.keys()), dtype=float)
    log_male = np.log([_CSO_2017_ANCHORS_MALE[int(a)] for a in anchor_ages])
    log_female = np.log([_CSO_2017_ANCHORS_FEMALE[int(a)] for a in anchor_ages])

    qx_male = np.exp(np.interp(ages, anchor_ages, log_male))
    qx_female = np.exp(np.interp(ages, anchor_ages, log_female))
    qx_male[-1] = 1.0
    qx_female[-1] = 1.0

    df = pd.DataFrame(
        {
            "age": ages,
            "qx_male": np.round(qx_male, 6),
            "qx_female": np.round(qx_female, 6),
        }
    )
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def load_mortality_table(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the 2017 CSO Ultimate Mortality Table from disk or build if absent.

    Parameters
    ----------
    csv_path : str | Path | None, optional
        Path to `mortality_table_cso2017.csv`.

    Returns
    -------
    pd.DataFrame
        Mortality table indexed or ordered by `age` with `qx_male` and `qx_female`.
    """
    if csv_path is not None and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    default_path = Path(__file__).resolve().parent.parent / "data" / "mortality_table_cso2017.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    return build_cso2017_table(save_path=default_path)


@dataclass(frozen=True)
class ActuarialAssumptions:
    """Container for demographic, economic, expense, lapse, and valuation assumptions.

    Attributes
    ----------
    policy_term : int
        Contract coverage and premium-paying term in years (default 20).
    product_type : ProductType
        Either `'endowment'` (20-Year Level Premium Endowment) or `'term'`
        (20-Year Level Term Life).
    investment_return : float
        Annual earned investment yield `i` on reserves and net cash flows (default 6.0%).
    valuation_interest_rate : float
        Technical interest rate used for Net Premium Reserve valuation (default 6.0%).
    risk_discount_rate : float
        Shareholder hurdle rate `r_d` for discounting distributable profits (default 10.0%).
    acq_expense_pct : float
        Variable acquisition expense + commission as % of First Year Gross Premium (60.0%).
    acq_expense_fixed : float
        Fixed per-policy issue/underwriting cost in Year 1 ($50.0).
    maint_expense_pct : float
        Ongoing renewal maintenance expense as % of Renewal Gross Premium (5.0%).
    maint_expense_fixed : float
        Base annual per-policy maintenance expense ($15.0) at issue, inflated at `expense_inflation`.
    expense_inflation : float
        Annual compound expense inflation rate `i_e` (default 3.0%).
    gross_premium_loading : float
        Commercial loading multiplier over net premium or explicit loading factor
        to cover acquisition, maintenance, lapse, and profit margin (default 0.28).
    mortality_multiplier : float
        Scalar stress multiplier applied to base 2017 CSO mortality rates (default 1.0).
    lapse_rates : tuple[float, ...]
        Duration-dependent annual withdrawal rates `w_t` for `t = 1..policy_term`.
    csv_factors : tuple[float, ...]
        Cash Surrender Value factor as a percentage of Net Premium Reserve at duration `t`.
    """

    policy_term: int = 20
    product_type: ProductType = "endowment"
    investment_return: float = 0.06
    valuation_interest_rate: float = 0.06
    risk_discount_rate: float = 0.10
    acq_expense_pct: float = 0.60
    acq_expense_fixed: float = 50.0
    maint_expense_pct: float = 0.05
    maint_expense_fixed: float = 15.0
    expense_inflation: float = 0.03
    gross_premium_loading: float = 0.20
    mortality_multiplier: float = 1.0
    lapse_rates: tuple[float, ...] = field(
        default_factory=lambda: (
            0.150,  # Year 1: 15.0%
            0.100,  # Year 2: 10.0%
            0.075,  # Year 3: 7.5%
            *([0.040] * 17),  # Year 4..20: 4.0% constant
        )
    )
    csv_factors: tuple[float, ...] = field(
        default_factory=lambda: (
            0.0,  # Year 1: 0%
            0.0,  # Year 2: 0%
            *tuple(np.round(np.linspace(0.80, 0.95, 18), 4)),  # Year 3..20: 80% -> 95%
        )
    )

    def get_lapse_vector(self) -> np.ndarray:
        """Return 1D numpy array of lapse rates `w_t` for `t = 1..policy_term`."""
        arr = np.asarray(self.lapse_rates, dtype=float)
        if len(arr) < self.policy_term:
            pad = np.full(self.policy_term - len(arr), arr[-1], dtype=float)
            arr = np.concatenate([arr, pad])
        return np.clip(arr[: self.policy_term], 0.0, 1.0)

    def get_csv_factor_vector(self) -> np.ndarray:
        """Return 1D numpy array of surrender value percentages of Net Premium Reserve."""
        arr = np.asarray(self.csv_factors, dtype=float)
        if len(arr) < self.policy_term:
            pad = np.full(self.policy_term - len(arr), arr[-1], dtype=float)
            arr = np.concatenate([arr, pad])
        return np.clip(arr[: self.policy_term], 0.0, 1.0)

    def get_mortality_matrix(
        self,
        issue_ages: np.ndarray,
        genders: np.ndarray,
        mortality_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Build a `(N, T)` matrix of mortality rates `q_{x+t-1}` for `t = 1..T`.

        Parameters
        ----------
        issue_ages : np.ndarray
            1D integer array of issue ages `x` of shape `(N,)`.
        genders : np.ndarray
            1D array of gender codes (`'M'` or `'F'`) of shape `(N,)`.
        mortality_df : pd.DataFrame | None, optional
            2017 CSO mortality table. Loaded automatically if `None`.

        Returns
        -------
        np.ndarray
            2D float array of shape `(N, policy_term)` with stressed mortality rates
            bounded in `[0.0, 1.0]`.
        """
        if mortality_df is None:
            mortality_df = load_mortality_table()

        df_sorted = mortality_df.sort_values("age").reset_index(drop=True)
        max_age = int(df_sorted["age"].max())
        qx_m_lookup = df_sorted["qx_male"].to_numpy(dtype=float)
        qx_f_lookup = df_sorted["qx_female"].to_numpy(dtype=float)

        durations = np.arange(self.policy_term, dtype=int)  # 0 .. T-1
        attained_ages = np.clip(issue_ages[:, None] + durations[None, :], 0, max_age)

        is_male = (np.asarray(genders) == "M")[:, None]
        base_q = np.where(is_male, qx_m_lookup[attained_ages], qx_f_lookup[attained_ages])
        return np.clip(base_q * self.mortality_multiplier, 0.0, 1.0)
