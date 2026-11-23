"""Run the three portfolio simulations and print their results.

Usage:
    python scripts/demo_simulations.py
"""
import tht
from tht.simulations import ifrs9_ecl, markov_migration, profitability_mc


def main():
    data = tht.load_data()
    loan, payment, clarity = data["loan"], data["payment"], data["underwriting"]

    print("=" * 64)
    print("1. PROFITABILITY (Monte Carlo expected profit)")
    print("=" * 64)
    r = profitability_mc(loan, payment, clarity)
    for k, v in r.items():
        print(f"  {k:<18} {v}")

    print("\n" + "=" * 64)
    print("2. IFRS-9 TIME-WISE EXPECTED CREDIT LOSS")
    print("=" * 64)
    r = ifrs9_ecl(loan, payment, clarity)
    for k, v in r.items():
        print(f"  {k:<18} {v}")

    print("\n" + "=" * 64)
    print("3. MARKOV MIGRATION (delinquency transition matrix)")
    print("=" * 64)
    r = markov_migration(loan, payment)
    states = r["transition_matrix"]["states"]
    print("  states:", states)
    print("  transition matrix (row -> col):")
    for i, row in enumerate(r["transition_matrix"]["matrix"]):
        print(f"    {states[i]:<10} {row}")
    print("  cumulative P(default) by scenario:")
    for scen, d in r["scenario_default_by_t"].items():
        print(f"    {scen:<10} t12={d['t12']}  t24={d['t24']}")


if __name__ == "__main__":
    main()
