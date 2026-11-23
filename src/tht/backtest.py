"""Model backtest: prove economic value on real data.

The backtest:

1. Splits the resolved cohort into train / validation / test (60/20/20).
2. Selects the best model (of 4 families) on the validation AUC.
3. Backtests a decision policy on the test set and measures **uplift** vs
   "approve everyone".
4. Generates a synthetic applicant population from the validation set and runs a
   Monte-Carlo profit simulation.
5. Estimates a bootstrap **prediction interval** on the PD for a confidence threshold.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .eval import auc, ks_stat
from .features import build_features
from .model import build_preprocessor
from .target import build_target

SERVICING_COST = 25.0
COST_OF_FUNDS_RATE = 0.02
TERM_MONTHS = 6


def _model_zoo(seed):
    return {
        "logistic": LogisticRegression(max_iter=2000),
        "hist_gbm": HistGradientBoostingClassifier(random_state=seed),
        "lightgbm": lgb.LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1),
        "xgboost": xgb.XGBClassifier(
            n_jobs=1, tree_method="hist", eval_metric="logloss",
            random_state=seed, verbosity=0,
        ),
    }


def _recovery_rate(loan, payment):
    outcome = build_target(loan)
    bad = outcome[outcome["defaulted"] == 1]
    paid = payment.merge(bad[["loanId"]], on="loanId", how="inner")
    cleared = paid[paid["paymentStatus"].astype(str).isin(["Checked", "Complete"])]
    recovered = float(cleared["principal"].sum())
    outstanding = float(bad["loanAmount"].sum())
    return (recovered / outstanding) if outstanding else 0.0


def _economics(X, lgd):
    loan_amt = X["loanAmount"].to_numpy(float)
    sched = X["originallyScheduledPaymentAmount"].to_numpy(float)
    lead_cost = X["leadCost"].to_numpy(float)
    interest = np.clip(sched - loan_amt, 0, None)
    cof = COST_OF_FUNDS_RATE * loan_amt * (TERM_MONTHS / 12.0)
    cost = lead_cost + SERVICING_COST + cof
    loss = lgd * loan_amt
    return interest, cost, loss


def _profit(approve_mask, interest, cost, loss, defaulted):
    perform = approve_mask & ~defaulted
    default = approve_mask & defaulted
    return float(
        interest[perform].sum()
        - cost[approve_mask].sum()
        - loss[default].sum()
    )


def run_backtest(loan, payment, clarity, n_sims=5000, n_bootstrap=25, seed=42):
    X, y = build_features(loan, payment, clarity)
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = list(X.select_dtypes(exclude=[np.number]).columns)

    # 1. train / validation / test
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )

    pre = build_preprocessor(num, cat)
    X_tr_t = pre.fit_transform(X_tr)
    X_va_t = pre.transform(X_va)
    X_te_t = pre.transform(X_te)

    # 2. select best model by validation AUC
    zoo = _model_zoo(seed)
    val_auc = {}
    fitted = {}
    for name, m in zoo.items():
        m.fit(X_tr_t, y_tr)
        val_auc[name] = auc(y_va, m.predict_proba(X_va_t)[:, 1])
        fitted[name] = m
    best_name = max(val_auc, key=val_auc.get)
    best = fitted[best_name]

    pd_te = best.predict_proba(X_te_t)[:, 1]
    test_auc = auc(y_te, pd_te)
    test_ks = ks_stat(y_te, pd_te)

    # 3. economics + backtest uplift
    lgd = 1.0 - _recovery_rate(loan, payment)
    interest, cost, loss = _economics(X_te, lgd)
    defaulted = y_te.to_numpy().astype(bool)

    approve_all = _profit(np.ones(len(y_te), bool), interest, cost, loss, defaulted)

    curve = {}
    best_threshold, best_profit = None, -np.inf
    for T in np.arange(0.30, 0.96, 0.05):
        mask = pd_te < T
        p = _profit(mask, interest, cost, loss, defaulted)
        curve[round(float(T), 2)] = round(p, 0)
        if p > best_profit:
            best_profit, best_threshold = p, float(T)

    uplift = best_profit - approve_all

    # 4. synthetic population from the validation set (bootstrap + light noise)
    rng = np.random.default_rng(seed)
    n_pop = 5000
    idx = rng.integers(0, len(X_va), n_pop)
    synth = X_va.iloc[idx].copy()
    for c in num:
        if c in synth.columns:
            sd = synth[c].std()
            if sd > 0:
                synth[c] = synth[c] + rng.normal(0, 0.05 * sd, n_pop)
    synth_t = pre.transform(synth)
    pd_synth = best.predict_proba(synth_t)[:, 1]

    # Monte-Carlo profit on the synthetic population
    s_interest, s_cost, s_loss = _economics(synth, lgd)
    base_synth = (s_interest - s_cost).sum()
    penalty = s_interest + s_loss  # defaulting forfeits interest AND incurs the loss
    profits = np.empty(n_sims)
    for s in range(n_sims):
        defaults = rng.random(n_pop) < pd_synth
        profits[s] = base_synth - float((defaults * penalty).sum())

    # 5. bootstrap prediction interval on the selected model
    def factory():
        return _model_zoo(seed)[best_name]

    y_tr_np = y_tr.to_numpy()
    preds = np.empty((n_bootstrap, len(pd_te)))
    for b in range(n_bootstrap):
        bidx = rng.integers(0, X_tr_t.shape[0], X_tr_t.shape[0])
        bm = factory().fit(X_tr_t[bidx], y_tr_np[bidx])
        preds[b] = bm.predict_proba(X_te_t)[:, 1]
    lo = np.percentile(preds, 5, axis=0)
    hi = np.percentile(preds, 95, axis=0)
    width = hi - lo
    confident = (width < 0.15).mean()  # fraction with a narrow interval

    return {
        "split": {"train": int(len(y_tr)), "validation": int(len(y_va)), "test": int(len(y_te))},
        "best_model": best_name,
        "validation_auc": {k: round(v, 3) for k, v in sorted(val_auc.items(), key=lambda kv: -kv[1])},
        "test_auc": round(test_auc, 3),
        "test_ks": round(test_ks, 3),
        "lgd": round(lgd, 3),
        "recovery_rate": round(1 - lgd, 3),
        "backtest": {
            "approve_all_profit": round(approve_all, 0),
            "best_threshold": best_threshold,
            "best_profit": round(best_profit, 0),
            "uplift": round(uplift, 0),
            "uplift_pct": round(uplift / approve_all * 100, 1) if approve_all else 0.0,
            "threshold_curve": curve,
        },
        "synthetic_population": {
            "n": n_pop,
            "expected_profit": round(float(profits.mean()), 0),
            "p5": round(float(np.percentile(profits, 5)), 0),
            "p95": round(float(np.percentile(profits, 95)), 0),
            "expected_default_rate": round(float(pd_synth.mean()), 3),
        },
        "prediction_interval": {
            "mean_width": round(float(width.mean()), 3),
            "confident_fraction": round(float(confident), 3),
        },
    }
