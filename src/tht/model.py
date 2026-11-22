"""PD model training and evaluation.

Benchmarks four model families against the same holdout split:
  * logistic regression  -> interpretable scorecard
  * histogram GBM (sklearn)
  * LightGBM
  * XGBoost

By default the GBMs use fixed, reasonable hyperparameters (fast). Pass
``tune=True`` to run an expanded randomized hyperparameter search (parallelised
with ``n_jobs=-1``) over a broad grid for an extra lift.
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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


def _model_specs(random_state):
    """(estimator, search space) per family. ``None`` search space = no tuning.

    Tree boosters set ``n_jobs=1`` internally so the outer ``RandomizedSearchCV``
    can parallelise across folds/combos without thread oversubscription.
    """
    return {
        "logistic": (
            LogisticRegression(max_iter=2000),
            None,
        ),
        "hist_gbm": (
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "max_leaf_nodes": [15, 31, 63, 127],
                "min_samples_leaf": [10, 20, 50, 100],
                "l2_regularization": [0.0, 0.1, 1.0, 10.0],
                "max_iter": [300, 500, 800],
            },
        ),
        "lightgbm": (
            lgb.LGBMClassifier(n_jobs=1, random_state=random_state, verbose=-1),
            {
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "min_child_samples": [10, 20, 50, 100],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0, 10.0],
                "n_estimators": [300, 500, 800],
            },
        ),
        "xgboost": (
            xgb.XGBClassifier(
                n_jobs=1,
                tree_method="hist",
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            ),
            {
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "min_child_weight": [1, 3, 5, 10],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0, 10.0],
                "n_estimators": [300, 500, 800],
            },
        ),
    }


def _fit_tuned(base, param_dist, X, y, n_iter, cv, n_jobs, random_state):
    """Randomized search over the expanded grid, parallelised with ``n_jobs``."""
    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


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


def run_benchmark(
    X,
    y,
    test_size=0.3,
    tune=False,
    n_iter=48,
    cv=4,
    n_jobs=-1,
    random_state=42,
):
    """Fit all models and return test-set metrics + LightGBM importances.

    ``tune=True`` runs an expanded randomized hyperparameter search (parallelised
    with ``n_jobs``, default ``-1`` = all cores) on the train fold.
    """
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pre = build_preprocessor(numeric_cols, categorical_cols)
    X_tr_t = pre.fit_transform(X_tr)
    X_te_t = pre.transform(X_te)

    specs = _model_specs(random_state)

    metrics = {}
    fitted = {}
    for name, (base, param_dist) in specs.items():
        if tune and param_dist is not None:
            model = _fit_tuned(base, param_dist, X_tr_t, y_tr, n_iter, cv, n_jobs, random_state)
        else:
            model = base.fit(X_tr_t, y_tr)
        fitted[name] = model
        proba = model.predict_proba(X_te_t)[:, 1]
        metrics[name] = evaluate(y_te, proba)

    importances = _aggregate_importances(
        pre, fitted["lightgbm"], X_te_t, y_te, numeric_cols, categorical_cols
    )
    return metrics, importances
