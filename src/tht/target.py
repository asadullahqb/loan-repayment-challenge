"""Target (default label) construction for the THT challenge."""

BAD_STATUSES = {
    "External Collection",
    "Internal Collection",
    "Charged Off",
    "Charged Off Paid Off",
    "Settled Bankruptcy",
}

GOOD_STATUSES = {"Paid Off Loan"}


def build_target(loan):
    """Return resolved loans with a binary ``defaulted`` label.

    Only loans with a definitive repayment outcome are kept; ambiguous/ongoing
    statuses (e.g. "Settlement Paid Off", "Returned Item", "New Loan") are dropped.
    """
    mask = loan["loanStatus"].isin(BAD_STATUSES | GOOD_STATUSES)
    out = loan[mask].copy()
    out["defaulted"] = out["loanStatus"].isin(BAD_STATUSES).astype(int)
    return out
