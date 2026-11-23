"""Run the model backtest (economic uplift + synthetic population + intervals)."""
import tht
from tht.backtest import run_backtest


def main():
    data = tht.load_data()
    r = run_backtest(data["loan"], data["payment"], data["underwriting"])

    print("=" * 64)
    print("MODEL BACKTEST (economic uplift on real data)")
    print("=" * 64)
    print("  split:", r["split"])
    print("  best model:", r["best_model"])
    print("  validation AUC:", r["validation_auc"])
    print("  test AUC:", r["test_auc"], " KS:", r["test_ks"])
    print("  LGD:", r["lgd"], " recovery rate:", r["recovery_rate"])

    b = r["backtest"]
    print("\n  backtest:")
    print("    approve-all profit :", f"${b['approve_all_profit']:,.0f}")
    print("    best threshold     :", b["best_threshold"])
    print("    best profit        :", f"${b['best_profit']:,.0f}")
    print("    uplift             :", f"${b['uplift']:,.0f}", f"({b['uplift_pct']}%)")
    print("    threshold curve:")
    for T, p in b["threshold_curve"].items():
        print(f"      PD<{T:<4} profit ${p:,.0f}")

    s = r["synthetic_population"]
    print("\n  synthetic population (from validation set):")
    print("    size             :", s["n"])
    print("    expected profit  :", f"${s['expected_profit']:,.0f}")
    print("    profit 5–95%     :", f"${s['p5']:,.0f} – ${s['p95']:,.0f}")
    print("    expected default :", f"{s['expected_default_rate']*100:.1f}%")

    pi = r["prediction_interval"]
    print("\n  prediction interval (bootstrap):")
    print("    mean PD interval width :", pi["mean_width"])
    print("    fraction confident     :", f"{pi['confident_fraction']*100:.1f}%")


if __name__ == "__main__":
    main()
