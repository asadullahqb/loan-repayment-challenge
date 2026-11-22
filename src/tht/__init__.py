"""Shared helpers for the THT loan-repayment challenge.

Importable as ``import tht`` from any notebook located in ``nb/src/``.
"""
from .data import DATA_DIR, FILES, load_data
from .eda import fraud_indicator_sparsity, missingness_report, segment_default_rate
from .eval import auc, ks_stat
from .external import STATE_UNEMPLOYMENT, add_monthly_macro, add_state_unemployment
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from .model import build_preprocessor, gini, run_benchmark
from .monitoring import DriftMonitor, SelfMaintainingModel, psi, simulate_covariate_drift
from .target import BAD_STATUSES, GOOD_STATUSES, build_target

__all__ = [
    "DATA_DIR",
    "FILES",
    "load_data",
    "missingness_report",
    "segment_default_rate",
    "fraud_indicator_sparsity",
    "auc",
    "ks_stat",
    "BAD_STATUSES",
    "GOOD_STATUSES",
    "build_target",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "build_features",
    "build_preprocessor",
    "gini",
    "run_benchmark",
    "STATE_UNEMPLOYMENT",
    "add_state_unemployment",
    "add_monthly_macro",
    "DriftMonitor",
    "SelfMaintainingModel",
    "psi",
    "simulate_covariate_drift",
]
