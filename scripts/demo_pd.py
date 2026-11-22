"""End-to-end PD model benchmark.

Usage:
    python scripts/demo_pd.py                 # application-time features (fast)
    python scripts/demo_pd.py --behavioral    # + leading delinquency indicators
    python scripts/demo_pd.py --sample 0.2    # 20% smoke-test pass
    python scripts/demo_pd.py --tune          # add a small grid search
"""
import argparse

import tht


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral", action="store_true", help="add delinquency indicators")
    parser.add_argument("--sample", type=float, default=None, help="fraction of cohort to sample")
    parser.add_argument("--tune", action="store_true", help="run a small grid search")
    args = parser.parse_args()

    data = tht.load_data()
    X, y = tht.build_features(
        data["loan"],
        data["payment"],
        data["underwriting"],
        behavioral=args.behavioral,
        sample_frac=args.sample,
    )

    mode = "behavioral (app-time + delinquency)" if args.behavioral else "application-time"
    print(f"Feature set     : {mode}")
    print(f"Resolved cohort : {len(y):,} loans")
    print(f"Default rate    : {y.mean():.1%}  ({y.sum():,} bad / {(1 - y).sum():,} good)")
    print(f"Features        : {X.shape[1]}  (numeric + categorical)\n")

    metrics, importances = tht.run_benchmark(X, y, tune=args.tune)

    print("Test-set metrics (30% holdout, stratified):")
    print(f"{'model':<14}{'AUC':>8}{'KS':>8}{'Gini':>8}")
    for name, m in metrics.items():
        print(f"{name:<14}{m['auc']:>8.3f}{m['ks']:>8.3f}{m['gini']:>8.3f}")

    print("\nLightGBM permutation importances (top features):")
    for feat, imp in importances.items():
        print(f"  {feat:<24}{imp:>8.4f}")


if __name__ == "__main__":
    main()
