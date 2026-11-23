"""Train a transparent logistic-regression scorecard and export it for the dashboard.

Writes ``docs/model_scorecard.json`` (intercept, per-feature coefficients, scaler
params, category encodings, slider bounds, and a default input) consumed by
``docs/index.html``.

The scorecard uses clean, interpretable application-time features only (no
log/ratio transforms), so every input maps 1:1 to a driver the assessor can reason
about.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import tht

# Clean, interpretable application-time features (no derived log/ratio columns).
NUMERIC = [
    "apr",
    "loanAmount",
    "originallyScheduledPaymentAmount",
    "leadCost",
    "clearfraudscore",
    "inq_30d",
    "inq_90d",
    "inq_365d",
    "max_ssn_per_bank",
    "n_prior_loans",
    "n_prior_defaults",
    "prior_default_rate",
]
CATEGORICAL = ["leadType", "state", "payFrequency", "identity_match"]

LABELS = {
    "apr": "APR (%)",
    "loanAmount": "Loan amount ($)",
    "originallyScheduledPaymentAmount": "Total scheduled payments ($)",
    "leadCost": "Lead cost ($)",
    "clearfraudscore": "Clarity fraud score",
    "inq_30d": "Credit inquiries (last 30d)",
    "inq_90d": "Credit inquiries (last 90d)",
    "inq_365d": "Credit inquiries (last 365d)",
    "max_ssn_per_bank": "Max SSNs on a bank account",
    "n_prior_loans": "Prior loans",
    "n_prior_defaults": "Prior defaults",
    "prior_default_rate": "Prior default rate",
    "leadType": "Lead type",
    "state": "State",
    "payFrequency": "Pay frequency",
    "identity_match": "Identity match",
}


def _numeric_slider_bounds(vals):
    lo, hi = np.nanpercentile(vals.astype(float), [1, 99])
    return float(lo), float(hi)


def main():
    data = tht.load_data()
    X_full, y = tht.build_features(data["loan"], data["payment"], data["underwriting"])
    X = X_full[NUMERIC + CATEGORICAL].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    pre = tht.build_preprocessor(NUMERIC, CATEGORICAL)
    X_tr_t = pre.fit_transform(X_tr)
    model = LogisticRegression(max_iter=2000).fit(X_tr_t, y_tr)

    auc = tht.auc(y_te, model.predict_proba(pre.transform(X_te))[:, 1])
    print("scorecard AUC (test):", round(auc, 3))

    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    coef = model.coef_[0]
    intercept = float(model.intercept_[0])
    names = list(pre.get_feature_names_out())

    numeric = {}
    for i, feat in enumerate(NUMERIC):
        vals = X[feat].astype(float)
        lo, hi = _numeric_slider_bounds(vals)
        numeric[feat] = {
            "label": LABELS.get(feat, feat),
            "mean": float(scaler.mean_[i]),
            "std": float(scaler.scale_[i]),
            "coef": float(coef[names.index(f"num__{feat}")]),
            "min": lo,
            "max": hi,
            "default": float(np.nanmedian(vals)),
        }

    categorical = {}
    col_offset = len(NUMERIC)
    for j, feat in enumerate(CATEGORICAL):
        cats = [str(c) for c in onehot.categories_[j]]
        coefs = {}
        for k, cat in enumerate(cats):
            coefs[cat] = float(coef[col_offset + k])
        col_offset += len(cats)
        categorical[feat] = {
            "label": LABELS.get(feat, feat),
            "options": cats,
            "coefs": coefs,
            "default": str(X[feat].mode().iloc[0]),
        }

    payload = {
        "model": "logistic_scorecard",
        "auc": round(auc, 3),
        "default_rate": round(float(y.mean()), 3),
        "intercept": intercept,
        "numeric": numeric,
        "categorical": categorical,
    }

    out = "docs/model_scorecard.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
