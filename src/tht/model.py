"""PD model training and evaluation.

Benchmarks four model families against the same holdout split:
  * logistic regression  -> interpretable scorecard
  * histogram GBM (sklearn)
  * LightGBM
  * XGBoost

By default the GBMs use fixed, reasonable hyperparameters (fast). Pass
``tune=True`` to run a small grid search (4 combos x 3-fold CV) on the train fold
for an extra lift at a modest time cost.
"""
import warnings

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .eval import auc, ks_stat


def gini(y_true, y_score):
    """Gini coefficient = 2 * AUC - 1."""
    return 2.0 * auc(y_true, y_score) - 1.0


def build_preprocessor(numeric_cols, categorical_cols):
    """Impute + encode + scale the raw feature frame without data leakage."""
    num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num, numeric_cols),
            ("cat", cat, categorical_cols),
        ]
    )


def evaluate(y_true, y_score):
    """Return AUC / KS / Gini for a set of predicted default probabilities."""
    return {
        "auc": auc(y_true, y_score),
        "ks": ks_stat(y_true, y_score),
        "gini": gini(y_true, y_score),
    }


def _make_models(random_state):
    """Fixed, reasonable hyperparameters for a fast baseline comparison."""
    return {
        "logistic": LogisticRegression(max_iter=2000),
        "hist_gbm": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=random_state,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
    }


def _tune_grids():
    """Small deterministic search spaces for the optional tuning pass."""
    return {
        "hist_gbm": {"learning_rate": [0.03, 0.06], "max_leaf_nodes": [31, 63]},
        "lightgbm": {"learning_rate": [0.03, 0.06], "num_leaves": [31, 63]},
        "xgboost": {"learning_rate": [0.03, 0.06], "max_depth": [5, 7]},
    }


def _aggregate_importances(pre, model, X_te_t, y_te, numeric_cols, categorical_cols):
    """Permutation importances folded back to original features."""
    with warnings.catch_warnings():
        # LightGBM warns when it can't verify feature-name order on a numpy
        # input; our pipeline controls ordering, so the warning is just noise.
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names", category=UserWarning
        )
        r = permutation_importance(
            model, X_te_t, y_te, n_repeats=5, scoring="roc_auc", random_state=42
        )
    per_col = dict(zip(pre.get_feature_names_out(), r.importances_mean))

    agg = {}
    for feat in numeric_cols + categorical_cols:
        agg[feat] = sum(
            v
            for name, v in per_col.items()
            if name == f"num__{feat}" or name.startswith(f"cat__{feat}_")
        )
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))


def run_benchmark(X, y, test_size=0.3, tune=False, random_state=42):
    """Fit all models and return test-set metrics + LightGBM importances.

    Returns ``(metrics, importances)`` where ``metrics`` maps model name to
    ``{"auc", "ks", "gini"}`` and ``importances`` maps feature name to its
    aggregated permutation importance.
    """
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pre = build_preprocessor(numeric_cols, categorical_cols)
    X_tr_t = pre.fit_transform(X_tr)
    X_te_t = pre.transform(X_te)

    models = _make_models(random_state)
    grids = _tune_grids()

    metrics = {}
    fitted = {}
    for name, model in models.items():
        if tune and name in grids:
            model = GridSearchCV(
                model, grids[name], cv=3, scoring="roc_auc", n_jobs=None
            ).fit(X_tr_t, y_tr).best_estimator_
        else:
            model.fit(X_tr_t, y_tr)
        fitted[name] = model
        proba = model.predict_proba(X_te_t)[:, 1]
        metrics[name] = evaluate(y_te, proba)

    importances = _aggregate_importances(
        pre, fitted["lightgbm"], X_te_t, y_te, numeric_cols, categorical_cols
    )
    return metrics, importances
