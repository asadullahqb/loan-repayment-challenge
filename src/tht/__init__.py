"""Shared helpers for the THT loan-repayment challenge.

Importable as ``import tht`` from any notebook located in ``nb/src/``.
"""
from .data import DATA_DIR, FILES, load_data
from .eda import fraud_indicator_sparsity, missingness_report, segment_default_rate
from .eval import auc, ks_stat
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
]
