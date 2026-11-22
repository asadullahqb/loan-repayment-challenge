"""Optional external macro features from FRED (US Federal Reserve).

FRED data is public domain — free for private-sector / corporate use. State
unemployment rates join on ``state``; a national series (e.g. ``UNRATE``) can be
joined on ``applicationDate`` at monthly grain.

Usage (run somewhere with network, e.g. Colab):

    from tht.external import add_state_unemployment
    loan = add_state_unemployment(loan)          # adds "state_unemployment" column
    loan = add_monthly_macro(loan, "UNRATE", "national_unemployment")

Note: this module requires outbound network access. It is key-less by default
(FRED's CSV endpoint); set ``FRED_API_KEY`` to use ``fredapi`` instead.
"""
import os

import pandas as pd

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# State abbreviation -> FRED state unemployment-rate series id (2-letter + "UR").
STATE_UNEMPLOYMENT = {
    "AL": "ALUR", "AK": "AKUR", "AZ": "AZUR", "AR": "ARUR", "CA": "CAUR",
    "CO": "COUR", "CT": "CTUR", "DE": "DEUR", "DC": "DCUR", "FL": "FLUR",
    "GA": "GAUR", "HI": "HIUR", "ID": "IDUR", "IL": "ILUR", "IN": "INUR",
    "IA": "IAUR", "KS": "KSUR", "KY": "KYUR", "LA": "LAUR", "ME": "MEUR",
    "MD": "MDUR", "MA": "MAUR", "MI": "MIUR", "MN": "MNUR", "MS": "MSUR",
    "MO": "MOUR", "MT": "MTUR", "NE": "NEUR", "NV": "NVUR", "NH": "NHUR",
    "NJ": "NJUR", "NM": "NMUR", "NY": "NYUR", "NC": "NCUR", "ND": "NDUR",
    "OH": "OHUR", "OK": "OKUR", "OR": "ORUR", "PA": "PAUR", "RI": "RIUR",
    "SC": "SCUR", "SD": "SDUR", "TN": "TNUR", "TX": "TXUR", "UT": "UTUR",
    "VT": "VTUR", "VA": "VAUR", "WA": "WAUR", "WV": "WVUR", "WI": "WIUR",
    "WY": "WYUR",
}


def _read_series(series_id):
    """Download one FRED series, returning a DataFrame with 'date' and 'value'."""
    if os.environ.get("FRED_API_KEY"):
        from fredapi import Fred  # optional, more reliable path

        s = Fred(api_key=os.environ["FRED_API_KEY"]).get_series(series_id)
        return s.rename("value").reset_index().rename(columns={"index": "date"})

    df = pd.read_csv(_FRED_CSV.format(series_id=series_id))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    val_col = series_id.lower()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )
    return out.dropna()


def series_asof(series_id, as_of=None):
    """Return the value of a FRED series at or before ``as_of`` (latest if None)."""
    as_of = pd.to_datetime(as_of) if as_of is not None else pd.Timestamp.max
    s = _read_series(series_id)
    s = s[s["date"] <= as_of].sort_values("date")
    return float(s.iloc[-1]["value"]) if not s.empty else None


def state_unemployment_asof(states, as_of=None):
    """Return ``{state: unemployment_rate}`` for each resolvable state."""
    return {
        st: series_asof(STATE_UNEMPLOYMENT[st], as_of=as_of)
        for st in states
        if st in STATE_UNEMPLOYMENT
    }


def add_state_unemployment(loan_df, as_of=None):
    """Left-join a ``state_unemployment`` column onto ``loan_df`` (key: ``state``)."""
    states = [s for s in loan_df["state"].dropna().unique() if s in STATE_UNEMPLOYMENT]
    rates = state_unemployment_asof(states, as_of=as_of)
    out = loan_df.copy()
    out["state_unemployment"] = out["state"].map(rates)
    return out


def add_monthly_macro(loan_df, series_id, column, date_col="applicationDate"):
    """Left-join a monthly FRED series onto ``loan_df`` by application month."""
    s = _read_series(series_id)
    s["month"] = s["date"].dt.to_period("M")

    out = loan_df.copy()
    out["month"] = pd.to_datetime(out[date_col], errors="coerce").dt.to_period("M")
    return out.merge(s[["month", "value"]].rename(columns={"value": column}), on="month", how="left").drop(columns="month")
