# Loan Repayment Challenge

Data-science take-home assessment: predict **loan risk / quality (probability of default)** for a
short-term, high-cost consumer installment lender, and tie the model to **profitability**.

## Problem

Predict the loan repayment risk of a given applicant. The underwriting problem decomposes into a
pipeline — eligibility (rules) → fraud (outsourced to a third-party vendor) → **PD** (the modelling
core) → LGD/EAD (downstream, where data permits).

## Data

Three parquet files (raw data is **not** committed — see `data/`, which is gitignored):

| File | Granularity | Role |
|------|-------------|------|
| `loan.parquet` | one row per application / funded loan | application features + outcomes |
| `payment.parquet` | one row per ACH attempt | repayment behaviour (target + LGD source) |
| `clarity_underwriting_variables.parquet` | one row per underwriting report | third-party fraud/identity features |

**Joins:** `loan.loanId ↔ payment.loanId`, `loan.clarityFraudId ↔ clarity.underwritingid`.

## Project structure

```
.
├── src/tht/          # reusable Python package (data loading, EDA, target, metrics)
├── notebooks/        # EDA and modelling notebooks
├── data/raw/         # raw parquet + dictionaries (gitignored)
├── pyproject.toml    # package config (editable install)
├── requirements.txt  # dependencies
├── .gitignore
└── README.md
```

## Setup

```powershell
# 1. Create and activate a dedicated environment
conda create -n tht python=3.11 pandas pyarrow scikit-learn matplotlib seaborn jupyter ipykernel xgboost lightgbm -c conda-forge
conda activate tht

# 2. Install dependencies and the local package (editable)
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import tht

data = tht.load_data()            # returns {'loan', 'payment', 'underwriting'}
outcome = tht.build_target(data["loan"])   # binary 'defaulted' label on resolved loans
tht.segment_default_rate(outcome, "leadType")
```

## Approach (CRISP-DM)

1. **Business & data understanding** — funnel, outcome definitions, feature coverage.
2. **PD modelling** — logistic-regression scorecard and/or gradient-boosted trees.
3. **Evaluation** — AUC, KS, Gini, calibration (accuracy is misleading at ~60% default rate).
4. **Profitability** — Monte-Carlo expected-loss simulation (PD × LGD × EAD).
