"""Demonstrate the self-maintaining model under synthetic drift.

One-time deploy -> healthy (no action) -> mild drift (auto-retrain) ->
severe drift (escalated to the MLOps engineer).

Usage:
    python scripts/demo_drift.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import tht
from tht.monitoring import SelfMaintainingModel, simulate_covariate_drift


def main():
    data = tht.load_data()
    X, y = tht.build_features(data["loan"], data["payment"], data["underwriting"])

    X_ref, X_pool, y_ref, y_pool = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    def model_factory():
        pre = tht.build_preprocessor(numeric_cols, categorical_cols)
        return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=42))])

    # One-time deploy on the reference (training) population.
    model = SelfMaintainingModel(model_factory).deploy(X_ref, y_ref)
    print(f"Deployed. Reference cohort = {len(X_ref):,} loans.\n")

    cs_std = X_ref["clearfraudscore"].std()
    apr_std = X_ref["apr"].std()

    scenarios = [
        ("no drift", {}),
        (
            "mild: clearfraudscore shifted +0.4 SD",
            {"clearfraudscore": {"shift": 0.4 * cs_std, "scale": 0.0}},
        ),
        (
            "severe: clearfraudscore +1.2 SD, apr +0.8 SD",
            {
                "clearfraudscore": {"shift": 1.2 * cs_std, "scale": 0.2 * cs_std},
                "apr": {"shift": 0.8 * apr_std, "scale": 0.2 * apr_std},
            },
        ),
    ]

    for label, specs in scenarios:
        batch = simulate_covariate_drift(X_pool, specs, seed=0)
        report = model.serve(batch, y_batch=y_pool)
        print(f"{label:<42} maxPSI={report['max_psi']:.3f}  "
              f"{report['severity']:<8} -> {report['action']}")


if __name__ == "__main__":
    main()
