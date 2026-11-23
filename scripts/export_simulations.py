"""Export portfolio-simulation results for the dashboard (docs/simulations.json)."""
import json

import tht
from tht.simulations import ifrs9_ecl, markov_migration, profitability_mc


def main():
    data = tht.load_data()
    loan, payment, clarity = data["loan"], data["payment"], data["underwriting"]

    out = {
        "profitability": profitability_mc(loan, payment, clarity),
        "ifrs9": ifrs9_ecl(loan, payment, clarity),
        "markov": markov_migration(loan, payment),
    }
    with open("docs/simulations.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote docs/simulations.json")


if __name__ == "__main__":
    main()
