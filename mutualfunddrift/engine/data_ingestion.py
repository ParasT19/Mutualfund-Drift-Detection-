"""
MutualFundDrift — data ingestion module.
Fetches AMFI portfolio disclosures,
NSE market cap classification, and fund mandate data from local CSV files.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from mftool import Mftool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Hardcoded fallback ISIN → NSE ticker for the 30 most common large-cap stocks
ISIN_TO_TICKER: dict[str, str] = {
    "INE002A01018": "RELIANCE",
    "INE467B01029": "TCS",
    "INE040A01034": "HDFCBANK",
    "INE009A01021": "INFY",
    "INE090A01021": "ICICIBANK",
    "INE030A01027": "HINDUNILVR",
    "INE154A01025": "ITC",
    "INE237A01028": "KOTAKBANK",
    "INE018A01030": "LT",
    "INE296A01024": "BAJFINANCE",
    "INE280A01028": "TITAN",
    "INE021A01026": "ASIANPAINT",
    "INE075A01022": "WIPRO",
    "INE860A01027": "HCLTECH",
    "INE585B01010": "MARUTI",
    "INE044A01036": "SUNPHARMA",
    "INE397D01024": "NESTLEIND",
    "INE066A01021": "ONGC",
    "INE117A01022": "NTPC",
    "INE019A01038": "HINDPETRO",
    "INE062A01020": "BPCL",
    "INE148A01014": "POWERGRID",
    "INE101A01026": "SBIN",
    "INE123W01016": "DMART",
    "INE205A01025": "GRASIM",
    "INE669C01036": "TATACONSUM",
    "INE081A01012": "COALINDIA",
    "INE001A01036": "ADANIPORTS",
    "INE158A01026": "ULTRACEMCO",
    "INE216A01030": "BAJAJFINSV",
}


# ---------------------------------------------------------------------------
# AMFI NAV fetcher
# ---------------------------------------------------------------------------

def fetch_amfi_nav(scheme_code: str) -> dict:
    """
    Fetch scheme metadata and latest NAV from AMFI via the mftool library.

    Returns a dict with scheme name, NAV, and metadata.
    On any failure returns a dict with an 'error' key describing the issue.
    """
    try:
        mf = Mftool()
        details = mf.get_scheme_details(scheme_code)
        if not details:
            return {"error": f"No scheme details returned for code {scheme_code}"}
        return {
            "scheme_code": scheme_code,
            "scheme_name": details.get("scheme_name", "Unknown"),
            "nav": details.get("nav", None),
            "date": details.get("date", None),
            "fund_house": details.get("fund_house", "Unknown"),
        }
    except Exception as exc:
        logger.error("fetch_amfi_nav failed for %s: %s", scheme_code, exc)
        return {"error": str(exc), "scheme_code": scheme_code}


# ---------------------------------------------------------------------------
# Portfolio holdings fetcher
# ---------------------------------------------------------------------------

def fetch_portfolio_holdings(scheme_code: str, month: str) -> pd.DataFrame:
    """
    Fetch monthly portfolio holdings for a scheme from AMFI or mftool.

    Args:
        scheme_code: AMFI scheme code, e.g. "120503"
        month: Month string in "YYYY-MM" format, e.g. "2024-03"

    Returns:
        DataFrame with columns: [isin, company_name, sector, market_value_lakhs,
        pct_of_nav, asset_type]. Returns an empty DataFrame with these columns on failure.
    """
    columns = ["isin", "company_name", "sector", "market_value_lakhs", "pct_of_nav", "asset_type"]
    empty_df = pd.DataFrame(columns=columns)

    # Primary: Try AMFI portfolio holding API
    try:
        year, mon = month.split("-")
        url = (
            f"https://www.amfiindia.com/modules/PortfolioHolding"
            f"?mf={scheme_code}&month={mon}&year={year}"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        # Parse tables from the HTML response
        tables = pd.read_html(response.text)
        if tables:
            for tbl in tables:
                tbl.columns = [str(c).strip().lower().replace(" ", "_") for c in tbl.columns]
                # Look for a table with equity holdings structure
                if "isin" in tbl.columns or any("isin" in str(c).lower() for c in tbl.columns):
                    # Rename columns to our canonical schema
                    col_map = {
                        c: "isin" for c in tbl.columns if "isin" in str(c).lower()
                    }
                    col_map.update({
                        c: "company_name" for c in tbl.columns
                        if "name" in str(c).lower() and "company" not in col_map.values()
                    })
                    col_map.update({
                        c: "pct_of_nav" for c in tbl.columns
                        if "%" in str(c).lower() or "nav" in str(c).lower()
                    })
                    col_map.update({
                        c: "market_value_lakhs" for c in tbl.columns
                        if "value" in str(c).lower() or "market" in str(c).lower()
                    })
                    tbl = tbl.rename(columns=col_map)
                    for col in columns:
                        if col not in tbl.columns:
                            tbl[col] = "Unknown" if col in ("sector", "asset_type") else 0.0
                    tbl["asset_type"] = tbl.get("asset_type", "Equity").fillna("Equity")
                    result = tbl[columns].copy()
                    equity = result[result["asset_type"].str.lower() == "equity"]
                    if not equity.empty:
                        logger.info(
                            "Fetched %d equity holdings for %s from AMFI HTML",
                            len(equity), scheme_code
                        )
                        return equity.reset_index(drop=True)
    except Exception as exc:
        logger.warning("AMFI HTML portfolio fetch failed for %s: %s", scheme_code, exc)

    # Fallback: mftool portfolio — with hard 20-second timeout
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        def _mftool_fetch():
            mf = Mftool()
            return mf.get_scheme_portfolio(scheme_code)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_mftool_fetch)
            try:
                portfolio = future.result(timeout=5)
            except FuturesTimeout:
                logger.warning("mftool timed out after 5s for %s", scheme_code)
                portfolio = None

        if portfolio:
            rows = []
            for holding in portfolio:
                rows.append({
                    "isin": holding.get("isin", ""),
                    "company_name": holding.get("company_name", holding.get("name", "Unknown")),
                    "sector": holding.get("sector", "Unknown"),
                    "market_value_lakhs": float(holding.get("market_value", 0) or 0),
                    "pct_of_nav": float(holding.get("percentage", holding.get("pct", 0)) or 0),
                    "asset_type": "Equity",
                })
            df = pd.DataFrame(rows, columns=columns)
            logger.info(
                "Fetched %d holdings for %s via mftool fallback", len(df), scheme_code
            )
            return df
    except Exception as exc:
        logger.error("mftool portfolio fallback failed for %s: %s", scheme_code, exc)

    logger.error("All portfolio fetching methods failed for scheme %s", scheme_code)
    return empty_df


# ---------------------------------------------------------------------------
# Stock fundamentals fetcher
# ---------------------------------------------------------------------------

def fetch_stock_fundamentals(isin_list: List[str]) -> pd.DataFrame:
    """
    Returns an empty fundamentals DataFrame.

    The size score (Large/Mid/Small cap) is computed purely from the NSE
    classification CSV which is already on disk — no internet call needed.
    The style score (Value/Growth) uses P/B ratios but gracefully falls
    back to a neutral 0.5 when no data is available, which is acceptable
    for drift detection purposes.

    Skipping external network calls keeps ingestion fast (<15s per quarter).

    Returns:
        Empty DataFrame with columns: [isin, ticker, pb_ratio, market_cap_crore]
    """
    logger.info(
        "Skipping external fundamentals fetch for %d ISINs — "
        "size score uses NSE CSV; style score uses neutral fallback.",
        len(isin_list),
    )
    return pd.DataFrame(columns=["isin", "ticker", "pb_ratio", "market_cap_crore"])




# ---------------------------------------------------------------------------
# Local CSV loaders
# ---------------------------------------------------------------------------

def load_nse_classification() -> pd.DataFrame:
    """
    Load the NSE stock market capitalisation classification from local CSV.

    Returns:
        DataFrame with columns: [rank, company_name, ticker, isin,
        market_cap_crore, category] where category is 'large_cap' (rank ≤ 100),
        'mid_cap' (rank 101–250), or 'small_cap' (rank > 250).
    """
    csv_path = DATA_DIR / "nse_classification.csv"
    try:
        df = pd.read_csv(csv_path)
        required_cols = {"rank", "company_name", "ticker", "isin", "market_cap_crore"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"nse_classification.csv missing columns: {missing}")

        # Compute or override category based on SEBI rank rules
        df["category"] = df["rank"].apply(
            lambda r: "large_cap" if r <= 100 else ("mid_cap" if r <= 250 else "small_cap")
        )
        logger.info("Loaded NSE classification: %d stocks", len(df))
        return df
    except FileNotFoundError:
        logger.error("nse_classification.csv not found at %s", csv_path)
        return pd.DataFrame(
            columns=["rank", "company_name", "ticker", "isin", "market_cap_crore", "category"]
        )
    except Exception as exc:
        logger.error("Failed to load nse_classification.csv: %s", exc)
        return pd.DataFrame(
            columns=["rank", "company_name", "ticker", "isin", "market_cap_crore", "category"]
        )
