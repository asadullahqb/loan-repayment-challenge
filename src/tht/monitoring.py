"""Low-maintenance drift monitoring for a one-time-deployed PD model.

The pipeline is intentionally tiny so it runs unattended:

* ``DriftMonitor`` compares a live batch to a reference distribution via PSI.
* Mild drift  -> the ``SelfMaintainingModel`` auto-retrains and re-baselines.
* Severe drift -> flagged and escalated to an MLOps engineer (no silent fix).

Nothing here depends on a specific model family; it works on any feature frame.
"""
import numpy as np
import pandas as pd

# Standard PSI severity thresholds.
MILD_PSI = 0.10
SEVERE_PSI = 0.25


def psi(expected, actual, bins=10, min_bucket=1e-6):
    """Population Stability Index between two 1-D numeric arrays.

    ``expected`` is the reference (training) distribution; ``actual`` is the live
    batch. Higher PSI => the live population has drifted away from reference.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_pct = np.clip(exp_pct, min_bucket, None)
    act_pct = np.clip(act_pct, min_bucket, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def simulate_covariate_drift(df, feature_specs, seed=0):
    """Return a synthetically drifted copy of ``df``.

    ``feature_specs`` maps a column name to a dict:
      * numeric columns: ``{"shift": mean_add, "scale": noise_std}``
      * categorical columns: ``{"p_swap": fraction_to_shuffle}``
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for feat, spec in feature_specs.items():
        if feat not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[feat]):
            shift = spec.get("shift", 0.0)
            scale = spec.get("scale", 0.0)
            out[feat] = out[feat] + shift + rng.normal(0, scale, size=len(out))
        else:
            p = spec.get("p_swap", 0.0)
            mask = rng.random(len(out)) < p
            cats = out[feat].dropna().unique()
            if len(cats):
                out.loc[mask, feat] = rng.choice(cats, size=int(mask.sum()))
    return out


def _severity(max_psi):
    if max_psi < MILD_PSI:
        return "healthy"
    if max_psi < SEVERE_PSI:
        return "mild"
    return "severe"


class DriftMonitor:
    """Computes per-feature PSI and classifies overall drift severity."""

    def __init__(self, mild_threshold=MILD_PSI, severe_threshold=SEVERE_PSI):
        self.mild_threshold = mild_threshold
        self.severe_threshold = severe_threshold
        self.reference = None

    def fit(self, X_ref):
        self.reference = X_ref.copy()
        return self

    def check(self, X):
        if self.reference is None:
            raise RuntimeError("DriftMonitor has no reference; call .fit() first.")
        per_feature = {
            col: psi(self.reference[col], X[col])
            for col in self.reference.columns
            if pd.api.types.is_numeric_dtype(self.reference[col])
        }
        max_psi = max(per_feature.values()) if per_feature else 0.0
        return {
            "severity": _severity(max_psi),
            "max_psi": round(max_psi, 4),
            "per_feature": {k: round(v, 4) for k, v in per_feature.items()},
        }


class SelfMaintainingModel:
    """Deploy-once model that self-retrains on mild drift and escalates severe drift.

    ``model_factory`` is a zero-argument callable returning an unfitted estimator.
    Labels for the live batch (``y_batch``) are optional; retraining only happens
    when they are provided (real deployments get labels with a delay).
    """

    def __init__(self, model_factory, monitor=None):
        self.model_factory = model_factory
        self.monitor = monitor or DriftMonitor()
        self.model = None

    def deploy(self, X_ref, y_ref):
        self.monitor.fit(X_ref)
        self.model = self.model_factory().fit(X_ref, y_ref)
        return self

    def serve(self, X_batch, y_batch=None):
        report = self.monitor.check(X_batch)
        severity = report["severity"]

        if severity == "mild" and y_batch is not None:
            self.model = self.model_factory().fit(X_batch, y_batch)
            self.monitor.fit(X_batch)  # re-baseline to the new population
            action = "auto-retrain"
        elif severity == "severe":
            action = "ESCALATE -> MLOps engineer"
        else:
            action = "none"

        report["action"] = action
        return report
