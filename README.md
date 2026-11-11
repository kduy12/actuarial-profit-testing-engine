# Actuarial Cash Flow Projection & Profit Testing Engine

A production-grade, vectorized **Actuarial Cash Flow Projection, Mathematical Reserving, and Profit Testing Engine** for traditional life insurance (**20-Year Level Premium Endowment** and **20-Year Level Term Life**).

Built with Python (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`), this repository models competing mortality and lapse decrements across a heterogeneous synthetic portfolio of **$10,000$ policies** over a 20-year horizon, verifies prospective and recursive reserve equivalence, computes shareholder value metrics (NPV of Future Profits, Profit Margin, IRR, Discounted Payback Period, Profit Signature), and executes a 6-scenario deterministic stress testing suite.

---

## Key Actuarial & Engineering Highlights

- **Vectorized Multi-Decrement Engine (`src/engine.py`):** Projects $10,000 \times 20 = 200,000$ policy-years simultaneously using `numpy` array broadcasting without slow row-by-row Python loops.
- **Prospective & Recursive Reserve Valuation (`src/reserving.py`):** Implements Equivalence Principle Net Premiums ($P^{net}$), commercial Gross Premiums ($P^{gross}$), forward Fackler/Net Premium Reserve recursion, prospective Net Premium Reserves (${}_tV$), Gross Premium Reserves (GPV), and graded Cash Surrender Values ($CSV_t$).
- **2017 CSO Mortality & Select Lapse Framework (`src/assumptions.py`):** Uses gender-distinct **2017 CSO Ultimate Mortality Table** rates ($q_{x+t}$) paired with a select withdrawal schedule ($15.0\%, 10.0\%, 7.5\%, 4.0\%+$).
- **Synthetic Portfolio Generator (`src/generator.py`):** Reproducibly (`seed=42`) generates $10,000$ contracts with Truncated Normal issue ages $x \sim \mathcal{N}(35, 8^2)$ on $[20, 55]$, $50\% / 50\%$ gender split, and Log-Normal Sum Assured (mean $\$50,000\text{ USD} \equiv 500,000,000\text{ VND}$).
- **Deterministic Sensitivity & Tornado Suite (`src/sensitivity.py`):** Evaluates portfolio resilience across Base Case, Mortality ($+15\%$), Early Lapse ($+100\%$ in Years 1–2), Interest Rate ($-100\text{ bps}$), Expense Inflation ($5.0\%\text{ p.a.}$), and Combined Adverse shocks.
- **Publication-Ready Hero Notebook (`notebooks/profit_testing_engine.ipynb`):** Pre-executed end-to-end Jupyter Notebook combining Bowers/Dickson notation, LaTeX derivations, tabular diagnostics, and corporate visualizations.

---

## Repository Directory Structure

```text
actuarial-profit-testing-engine/
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── mortality_table_cso2017.csv
│   └── synthetic_portfolio_10k.csv
├── src/
│   ├── __init__.py
│   ├── assumptions.py       # Demographic, economic, expense, and lapse parameters
│   ├── generator.py         # Vectorized synthetic policy portfolio generator
│   ├── engine.py            # Vectorized multi-decrement cohort cash flow projection
│   ├── reserving.py         # Prospective net/gross premium reserve calculation
│   └── sensitivity.py       # Sensitivity matrix and stress testing framework
├── notebooks/
│   └── profit_testing_engine.ipynb  # Primary hero deliverable (LaTeX + Code + Visuals)
└── tests/
    ├── __init__.py
    └── test_engine.py       # Unit tests verifying cash flow balances and reserve recursion
```

---

## Mathematical Formulations

### 1. Multi-Decrement Competing-Risk In-Force Recursion
For each policy year $t \in \{1, 2, \dots, 20\}$ and initial cohort size $l_0$:

$$\text{Deaths}_t = l_{t-1} \cdot q_{x+t-1}$$

$$\text{Lapses}_t = l_{t-1} \cdot (1 - q_{x+t-1}) \cdot w_t$$

$$l_t = l_{t-1} - \text{Deaths}_t - \text{Lapses}_t = l_{t-1} \cdot (1 - q_{x+t-1})(1 - w_t)$$

### 2. Net Premium Equivalence & Mathematical Reserve Recursion
Let $v = (1 + i)^{-1}$ at technical valuation interest rate $i = 6.0\%$. By the **Actuarial Equivalence Principle** at policy inception ($t = 0$, ${}_0V = 0$):

$$P^{net} = \frac{S.A. \sum_{t=1}^{n} v^t q_{x+t-1} + B_n v^n}{\sum_{t=1}^{n} v^{t-1}}, \quad B_n = \begin{cases} S.A. & \text{(Endowment)} \\ 0 & \text{(Term Life)} \end{cases}$$

The per-policy Net Premium Reserve ${}_tV$ evolves recursively for $t = 1, \dots, n$:

$${}_tV = ({}_{t-1}V + P^{net})(1 + i) - q_{x+t-1} \cdot S.A.$$

which is algebraically identical to the prospective reserve at duration $t$:

$${}_tV = B_n v^{n-t} + S.A. \sum_{m=1}^{n-t} v^m q_{x+t+m-1} - P^{net} \sum_{m=1}^{n-t} v^{m-1}$$

### 3. Annual Cohort Cash Flows & Distributable Profit Vector
For policy year $t \in [1, 20]$:
- **Gross Premium Inflow:** $P_t = l_{t-1} \times P^{gross}$
- **Expenses Outflow:** $E_t = \text{Commissions}_t + l_{t-1} \times E_t^{fixed}(1 + i_e)^{t-1}$
- **Investment Income:** $II_t = (V_{t-1} + P_t - E_t) \times i$
- **Death Claims Outflow:** $C_t = l_{t-1} \times q_{x+t-1} \times S.A.$
- **Surrender Payouts:** $S_t = l_{t-1} \times (1 - q_{x+t-1}) \times w_t \times CSV_t$
- **Maturity Benefit Outflow:** $M_{20} = l_{20} \times S.A.$ (funded via terminal reserve $V_{20} = l_{20} \cdot {}_{20}V = M_{20}$)
- **Cohort Reserve Change:** $\Delta V_t = V_t - V_{t-1}$, where $V_t = l_t \times {}_tV$
- **Net Profit Vector:**

$$Profit_t = P_t - E_t + II_t - C_t - S_t - \Delta V_t$$

### 4. Profitability & Shareholder Capital Metrics
At Risk Discount Rate $r_d = 10.0\%$:

$$\text{NPV}(\text{Profits}) = \sum_{t=1}^{20} \frac{Profit_t}{(1 + r_d)^t}, \qquad \text{Profit Margin} = \frac{\text{NPV}(\text{Profits})}{\text{NPV}(\text{Gross Premiums})}$$

$$\text{IRR}: \quad \sum_{t=1}^{20} \frac{Profit_t}{(1 + \text{IRR})^t} = 0, \qquad \text{Payback Year } t^* = \min \left\{ t : \sum_{k=1}^{t} \frac{Profit_k}{(1 + r_d)^k} \ge 0 \right\}$$

---

## Summary Results: Base Case vs. Sensitivity Scenarios

Evaluated across the **$10,000$-policy 20-Year Level Premium Endowment** portfolio ($\$500.0\text{M USD} \equiv 5.0\text{T VND}$ total Sum Assured, $r_d = 10.0\%$):

| Scenario | NPV of Profits ($) | NPV Gross Premiums ($) | Profit Margin (%) | IRR (%) | Payback Period (Yrs) | Year 1 Strain ($) | Delta NPV ($) | Delta NPV (%) | Delta Margin (bps) |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| **1. Base Case** | `$11,687,056.47` | `$95,023,863.40` | `12.30%` | `45.08%` | `4` | `-$6,150,999.99` | `$0.00` | `0.00%` | `0.0 bps` |
| **2. Mortality Shock (+15% q_x)** | `$11,096,696.54` | `$94,885,023.72` | `11.69%` | `42.77%` | `4` | `-$6,242,280.92` | `-$590,359.94` | `-5.05%` | `-60.4 bps` |
| **3. Early Lapse Shock (Y1=30%, Y2=20%)** | `$11,019,081.49` | `$74,826,912.85` | `14.73%` | `85.36%` | `2` | `-$4,016,919.01` | `-$667,974.99` | `-5.72%` | `+242.7 bps` |
| **4. Interest Rate Shock (-100 bps to 5.0%)** | `$5,577,119.92` | `$95,023,863.40` | `5.87%` | `34.07%` | `4` | `-$6,212,944.60` | `-$6,109,936.55` | `-52.28%` | `-643.0 bps` |
| **5. Expense Inflation Shock (5.0% p.a.)** | `$11,556,294.78` | `$95,023,863.40` | `12.16%` | `44.92%` | `4` | `-$6,150,999.99` | `-$130,761.69` | `-1.12%` | `-13.8 bps` |
| **6. Combined Adverse Scenario** | `$5,969,312.07` | `$74,725,065.14` | `7.99%` | `67.95%` | `2` | `-$4,170,553.93` | `-$5,717,744.40` | `-48.92%` | `-431.1 bps` |

---

## Installation & Execution Instructions

### 1. Environment Setup

**Option A — Using `venv` + `pip`:**
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

**Option B — Using `conda`:**
```bash
conda env create -f environment.yml
conda activate actuarial-profit-testing-engine
```

### 2. Run Unit Tests (`pytest`)
```bash
pytest tests/ -v
```

### 3. Execute the Notebook Headlessly (`nbconvert`)
```bash
jupyter nbconvert --to notebook --execute notebooks/profit_testing_engine.ipynb --inplace
```

### 4. Programmatic Quickstart
```python
from src.assumptions import ActuarialAssumptions, build_cso2017_table
from src.generator import generate_synthetic_portfolio
from src.sensitivity import run_sensitivity_suite

mortality_df = build_cso2017_table("data/mortality_table_cso2017.csv")
portfolio_df = generate_synthetic_portfolio(10_000, seed=42, save_path="data/synthetic_portfolio_10k.csv")
summary_df, scenario_results = run_sensitivity_suite(portfolio_df, ActuarialAssumptions(), mortality_df)
print(summary_df.to_string(index=False))
```
