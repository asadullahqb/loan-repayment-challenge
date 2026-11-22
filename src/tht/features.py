"""Feature construction for the PD (probability-of-default) model.

Two feature tiers:

* **Application-time (default)** — everything an underwriter knows at origination,
  including an *internal borrower-history* ("mini-bureau") built from ``anon_ssn``.
* **Behavioral (``behavioral=True``)** — adds post-origination *leading delinquency*
  signals (NSF/rejected attempts, skipped payments) plus ``nPaidOff``. Outcome-
  *leaking* aggregates (paydown ratio, successful-payment counts, tenure) are
  intentionally excluded.
"""
import numpy as np
import pandas as pd

from .target import BAD_STATUSES, GOOD_STATUSES, build_target

# Application-time numeric features.
NUMERIC_FEATURES = [
    "apr",
    "loanAmount",
    "originallyScheduledPaymentAmount",
    "leadCost",
    "clearfraudscore",
    "total_fraud_indicators",
    "clarity_missing",
    "inq_30d",
    "inq_90d",
    "inq_365d",
    "max_ssn_per_bank",
    "scheduled_payment_ratio",
    "log_lead_cost",
    "log_max_ssn",
    # Internal borrower history (mini-bureau).
    "n_prior_loans",
    "n_prior_defaults",
    "prior_default_rate",
    "has_prior_history",
]

# Application-time categorical features.
CATEGORICAL_FEATURES = [
    "leadType",
    "state",
    "payFrequency",
    "identity_match",
    "ssn_name_match",
    "name_addr_match",
    "ssn_dob_match",
]

# Behavioral (post-origination) leading-indicator features.
BEHAVIORAL_FEATURES = [
    "nPaidOff",
    "n_rejected",
    "n_skipped",
    "n_cancelled",
    "ever_nsf",
    "ever_skipped",
]

# Clarity source columns -> clean feature names.
_TOTAL_FRAUD_COL = (
    ".underwritingdataclarity.clearfraud.clearfraudindicator.totalnumberoffraudindicators"
)
_MAX_SSN_COL = (
    ".underwritingdataclarity.clearfraud.clearfraudindicator.maxnumberofssnswithanybankaccount"
)
_INQ_30D = ".underwritingdataclarity.clearfraud.clearfraudinquiry.thirtydaysago"
_INQ_90D = ".underwritingdataclarity.clearfraud.clearfraudinquiry.ninetydaysago"
_INQ_365D = ".underwritingdataclarity.clearfraud.clearfraudinquiry.threesixtyfivedaysago"
_IDENTITY_MATCH = (
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.overallmatchresult"
)
_SSN_NAME_MATCH = (
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamematch"
)
_NAME_ADDR_MATCH = (
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressmatch"
)
_SSN_DOB_MATCH = (
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.ssndobmatch"
)

_IDENTITY_COLS = [
    "identity_match",
    "ssn_name_match",
    "name_addr_match",
    "ssn_dob_match",
]

# Payment statuses that represent a failed / insufficient-funds attempt (a leading
# delinquency signal, not the default itself).
_REJECTED_STATUSES = ["Rejected", "Rejected Awaiting Retry"]


def _clarity_features(clarity):
    """Extract and rename the Clarity features used by the PD model."""
    cols = {
        "underwritingid": "clarityFraudId",
        "clearfraudscore": "clearfraudscore",
        _TOTAL_FRAUD_COL: "total_fraud_indicators",
        _MAX_SSN_COL: "max_ssn_per_bank",
        _INQ_30D: "inq_30d",
        _INQ_90D: "inq_90d",
        _INQ_365D: "inq_365d",
        _IDENTITY_MATCH: "identity_match",
        _SSN_NAME_MATCH: "ssn_name_match",
        _NAME_ADDR_MATCH: "name_addr_match",
        _SSN_DOB_MATCH: "ssn_dob_match",
    }
    cf = clarity[list(cols)].rename(columns=cols).copy()
    # One underwriting report can back multiple applications; keep the report unique.
    return cf.drop_duplicates("clarityFraudId")


def build_borrower_history(loan):
    """Per-loan prior-borrower history (an internal "mini-bureau").

    For each loan, counts the borrower's *prior* applications and their resolved
    outcomes, strictly before this loan's application date. This is application-time
    signal — nothing from the loan's own future is used.
    """
    l = loan[["loanId", "anon_ssn", "applicationDate", "loanStatus"]].copy()
    l["app_dt"] = pd.to_datetime(l["applicationDate"], errors="coerce")

    resolved_mask = l["loanStatus"].isin(BAD_STATUSES | GOOD_STATUSES)
    default_mask = l["loanStatus"].isin(BAD_STATUSES)
    l["resolved"] = resolved_mask.astype(int)
    l["hist_default"] = default_mask.astype(int)

    l = l.sort_values(["anon_ssn", "app_dt", "loanId"]).reset_index(drop=True)

    g = l.groupby("anon_ssn", sort=False)
    prior_loans = g.cumcount()
    prior_defaults = g["hist_default"].cumsum() - l["hist_default"]
    prior_resolved = g["resolved"].cumsum() - l["resolved"]

    hist = pd.DataFrame(
        {
            "loanId": l["loanId"],
            "n_prior_loans": prior_loans,
            "n_prior_defaults": prior_defaults,
            "n_prior_resolved": prior_resolved,
        }
    )
    hist["prior_default_rate"] = (
        hist["n_prior_defaults"] / hist["n_prior_resolved"].replace(0, np.nan)
    ).fillna(0)
    hist["has_prior_history"] = (hist["n_prior_resolved"] > 0).astype(int)
    return hist


def build_payment_features(payment):
    """Aggregate leading delinquency indicators to one row per loan.

    Excludes outcome-*leaking* measures (``isCollection``, successful-payment
    counts, principal paid, tenure reached).
    """
    p = payment.copy()
    status = p["paymentStatus"].astype(str)
    p["is_rejected"] = status.isin(_REJECTED_STATUSES).astype(int)
    p["is_skipped"] = status.eq("Skipped").astype(int)
    p["is_cancelled"] = status.eq("Cancelled").astype(int)

    return p.groupby("loanId").agg(
        n_rejected=("is_rejected", "sum"),
        n_skipped=("is_skipped", "sum"),
        n_cancelled=("is_cancelled", "sum"),
    )


def build_features(loan, payment, clarity, behavioral=False, sample_frac=None, random_state=42):
    """Return ``(X, y)`` for the PD model on the resolved cohort.

    ``behavioral=False`` returns application-time features only. Set
    ``behavioral=True`` to add leading delinquency indicators + ``nPaidOff``.

    ``sample_frac`` optionally downsamples the resolved cohort for a quick
    smoke-test pass (e.g. ``0.2`` for 20%).
    """
    outcome = build_target(loan)
    cf = _clarity_features(clarity)
    hist = build_borrower_history(loan)

    df = outcome.merge(cf, on="clarityFraudId", how="left")
    df = df.merge(hist, on="loanId", how="left")

    if behavioral:
        df = df.merge(build_payment_features(payment), on="loanId", how="left")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)

    df["clarity_missing"] = df["clearfraudscore"].isna().astype(int)

    for col in _IDENTITY_COLS:
        df[col] = df[col].astype("object").fillna("missing")

    # Engineered numeric features.
    df["scheduled_payment_ratio"] = (
        df["originallyScheduledPaymentAmount"] / df["loanAmount"].replace(0, np.nan)
    ).fillna(0)
    df["log_lead_cost"] = np.log1p(df["leadCost"])
    df["log_max_ssn"] = np.log1p(df["max_ssn_per_bank"])

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if behavioral:
        for col in ["n_rejected", "n_skipped", "n_cancelled"]:
            df[col] = df[col].fillna(0)
        df["ever_nsf"] = (df["n_rejected"] > 0).astype(int)
        df["ever_skipped"] = (df["n_skipped"] > 0).astype(int)
        df["nPaidOff"] = df["nPaidOff"].fillna(0)
        feature_cols = feature_cols + BEHAVIORAL_FEATURES

    X = df[feature_cols].copy()
    y = df["defaulted"].astype(int)
    return X, y
