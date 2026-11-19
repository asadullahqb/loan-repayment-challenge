"""Model-evaluation metrics used across EDA and modelling."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_score):
    """ROC AUC, made direction-agnostic (works if higher score => lower risk)."""
    a = roc_auc_score(y_true, y_score)
    return max(a, 1 - a)


def ks_stat(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max separation between good/bad scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bad = np.sort(y_score[y_true == 1])
    good = np.sort(y_score[y_true == 0])
    if len(bad) == 0 or len(good) == 0:
        return float("nan")

    thresholds = np.unique(y_score)
    cdf_bad = np.searchsorted(bad, thresholds, side="right") / len(bad)
    cdf_good = np.searchsorted(good, thresholds, side="right") / len(good)
    return float(np.max(np.abs(cdf_bad - cdf_good)))
