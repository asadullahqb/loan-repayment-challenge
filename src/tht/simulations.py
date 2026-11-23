"""Portfolio simulations tying the PD model to economics.

Three analyses (each returns a plain dict for easy printing):

* ``profitability_mc`` — Monte Carlo expected-profit distribution (PD x LGD x EAD).
* ``ifrs9_ecl`` — time-wise expected credit loss with a term structure + staging.
* ``markov_migration`` — delinquency transition matrix and default curve.

All assumptions are stated in the docstrings / output; raw data is never bundled.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .features import build_features
from .model import build_preprocessor
from .target import BAD_STATUSES, build_target

# --- shared helpers -----------------------------------------------------------

def _fit_pd(loan, payment, clarity, test_size=0.3, random_state=42):
    """Return (X_test, y_test, pd_test) with honest out-of-sample PD."""
    X, y = build_features(loan, payment, clarity)
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = list(X.select_dtypes(exclude=[np.number]).columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pre = build_preprocessor(num, cat)
    X_tr_t = pre.fit_transform(X_tr)
    model = LogisticRegression(max_iter=2000).fit(X_tr_t, y_tr)
    pd_te = model.predict_proba(pre.transform(X_te))[:, 1]
    return X_te, y_te, pd_te


def _recovery_rate(loan, payment):
    """Principal recovered / principal outstanding on defaulted loans.

    Only *cleared* payments ("Checked" / "Complete") count as recovered cash.
    """
    outcome = build_target(loan)
    bad = outcome[outcome["defaulted"] == 1]
    paid = payment.merge(bad[["loanId"]], on="loanId", how="inner")
    cleared = paid[paid["paymentStatus"].astype(str).isin(["Checked", "Complete"])]
    recovered = float(cleared["principal"].sum())
    outstanding = float(bad["loanAmount"].sum())
    return (recovered / outstanding) if outstanding else 0.0


def _term_structure(loan, payment):
    """Empirical hazard curve using installmentIndex as the time axis.

    ``default_time`` for a defaulted loan is proxied by ``nPaidOff`` (payments made
    before default); performing loans are censored at their schedule length.
    """
    outcome = build_target(loan)
    terms = payment.groupby("loanId")["installmentIndex"].max()
    df = outcome.merge(terms.rename("term"), on="loanId", how="left")
    df["term"] = df["term"].fillna(1).clip(lower=1).astype(int)
    # default time = installments paid off before default (proxy)
    df["default_time"] = np.where(
        df["defaulted"] == 1,
        df["nPaidOff"].fillna(0).clip(lower=0).astype(int) + 1,
        df["term"],
    )
    df["default_time"] = df["default_time"].clip(upper=df["term"])

    max_t = int(np.percentile(df["term"], 95))
    median_term = int(df["term"].median())
    hazard = {}
    at_risk = len(df)
    for t in range(1, max_t + 1):
        defaults_t = int((df["default_time"] == t).sum())
        hazard[t] = defaults_t / at_risk if at_risk > 0 else 0.0
        at_risk -= defaults_t
    return df, hazard, max_t, median_term


# --- 1. Profitability Monte Carlo --------------------------------------------

def profitability_mc(loan, payment, clarity, n_sims=20_000, seed=42,
                     servicing_cost=25.0, cost_of_funds_rate=0.02, term_months=6):
    """Monte Carlo expected-profit distribution for a portfolio.

    profit_i = interest_i - leadCost_i - servicing - cost_of_funds - (default ? LGD*EAD : 0)
    """
    X, y, pd_ = _fit_pd(loan, payment, clarity)
    lgd = 1.0 - _recovery_rate(loan, payment)

    loan_amt = X["loanAmount"].to_numpy(float)
    sched = X["originallyScheduledPaymentAmount"].to_numpy(float)
    lead_cost = X["leadCost"].to_numpy(float)

    interest = np.clip(sched - loan_amt, 0, None)  # scheduled interest + fees
    cof = cost_of_funds_rate * loan_amt * (term_months / 12.0)
    loss = lgd * loan_amt

    base = (interest - lead_cost - servicing_cost - cof).sum()
    loss_vec = loss
    pd_vec = pd_

    rng = np.random.default_rng(seed)
    n = len(pd_vec)
    profits = np.empty(n_sims)
    for s in range(n_sims):
        defaults = rng.random(n) < pd_vec
        profits[s] = base - float((defaults * loss_vec).sum())

    return {
        "n_loans": int(n),
        "mean_pd": round(float(pd_vec.mean()), 3),
        "lgd": round(lgd, 3),
        "recovery_rate": round(1 - lgd, 3),
        "expected_profit": round(float(profits.mean()), 0),
        "profit_std": round(float(profits.std()), 0),
        "p5_profit": round(float(np.percentile(profits, 5)), 0),
        "p95_profit": round(float(np.percentile(profits, 95)), 0),
        "prob_loss": round(float((profits < 0).mean()), 3),
    }


# --- 2. IFRS-9 time-wise ECL --------------------------------------------------

def ifrs9_ecl(loan, payment, clarity, discount_rate=0.05, lgd_override=None):
    """Lifetime ECL = sum_t marginal_PD(t) * LGD * EAD(t) * DF(t).

    Three FLI scenarios scale the hazard: Baseline x1, Upside x0.8, Downside x1.3.
    """
    df, hazard, max_t, median_term = _term_structure(loan, payment)
    lgd = lgd_override if lgd_override is not None else (1.0 - _recovery_rate(loan, payment))

    # representative loan
    ead0 = float(df["loanAmount"].median())
    term = max(int(median_term), 1)
    r = discount_rate / 12.0  # monthly discount

    def ecl_for(hazard_scale):
        survival = 1.0
        total = 0.0
        for t in range(1, max_t + 1):
            h = min(hazard.get(t, 0.0) * hazard_scale, 1.0)
            marg = survival * h
            survival *= (1.0 - h)
            ead_t = ead0 * (1.0 - (t - 1) / term)  # linear amortization
            df_t = 1.0 / (1.0 + r) ** t
            total += marg * lgd * ead_t * df_t
        return total

    scenarios = {
        "baseline": ecl_for(1.0),
        "upside": ecl_for(0.8),
        "downside": ecl_for(1.3),
    }
    weighted = 0.5 * scenarios["baseline"] + 0.25 * scenarios["upside"] + 0.25 * scenarios["downside"]

    return {
        "ead0": round(ead0, 0),
        "lgd": round(lgd, 3),
        "term_installments": term,
        "scenario_ecl": {k: round(v, 2) for k, v in scenarios.items()},
        "weighted_ecl": round(weighted, 2),
        "weighted_ecl_pct_of_ead": round(weighted / ead0 * 100, 2),
    }


# --- 3. Markov migration ------------------------------------------------------

def markov_migration(loan, payment):
    """3-state delinquency transition matrix (current -> delinquent -> default).

    States are derived per installment from ``paymentStatus`` / ``isCollection``:
      0 = current, 1 = delinquent (skipped / rejected), 2 = default (collection).
    """
    outcome = build_target(loan)
    defaulted_ids = set(outcome.loc[outcome["defaulted"] == 1, "loanId"])

    p = payment.copy()
    status = p["paymentStatus"].astype(str)
    # state: 2 if collection attempt; 1 if skipped/rejected; else 0 (current)
    p["state"] = np.where(
        p["isCollection"], 2,
        np.where(status.isin(["Skipped", "Rejected", "Rejected Awaiting Retry"]), 1, 0),
    )
    p = p.sort_values(["loanId", "installmentIndex"])
    p["next_state"] = p.groupby("loanId")["state"].shift(-1)

    trans = p.dropna(subset=["next_state"])
    counts = pd.crosstab(trans["state"], trans["next_state"])
    counts = counts.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    mat = (counts.div(counts.sum(axis=1), axis=0)).fillna(0).to_numpy()

    # Default curve from the Markov chain, treating 'default' (collection) as
    # absorbing so we measure *cumulative* default probability, not the snapshot.
    def curve(m, horizon=24, default_scale=1.0):
        m = m.copy()
        m[2] = [0.0, 0.0, 1.0]  # absorbing default
        m[:, 2] *= default_scale  # macro overlay: scale transitions INTO default
        m = m / m.sum(axis=1, keepdims=True)
        state = np.array([1.0, 0.0, 0.0])
        out = {}
        for t in range(1, horizon + 1):
            state = state @ m
            out[t] = round(float(state[2]), 4)
        return out

    scenarios = {
        "baseline": curve(mat),
        "upside": curve(mat, default_scale=0.75),
        "downside": curve(mat, default_scale=1.5),
    }

    return {
        "transition_matrix": {
            "states": ["current", "delinquent", "collection"],
            "matrix": np.round(mat, 3).tolist(),
        },
        "scenario_default_by_t": {
            k: {"t12": v.get(12), "t24": v.get(24)} for k, v in scenarios.items()
        },
    }
