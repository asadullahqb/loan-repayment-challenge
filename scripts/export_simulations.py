"""Export portfolio-simulation + backtest results for the dashboard.

Writes:
  * docs/simulations.json   (profitability / IFRS-9 / Markov)
  * docs/backtest.json      (model uplift backtest)
"""
import json

import tht
from tht.backtest import run_backtest
from tht.simulations import ifrs9_ecl, markov_migration, profitability_mc


def main():
    data = tht.load_data()
    loan, payment, clarity = data["loan"], data["payment"], data["underwriting"]

    sims = {
        "profitability": profitability_mc(loan, payment, clarity),
        "ifrs9": ifrs9_ecl(loan, payment, clarity),
        "markov": markov_migration(loan, payment),
    }
    with open("docs/simulations.json", "w", encoding="utf-8") as f:
        json.dump(sims, f, indent=2)

    backtest = run_backtest(loan, payment, clarity)
    with open("docs/backtest.json", "w", encoding="utf-8") as f:
        json.dump(backtest, f, indent=2)

    print("wrote docs/simulations.json and docs/backtest.json")


if __name__ == "__main__":
    main()
