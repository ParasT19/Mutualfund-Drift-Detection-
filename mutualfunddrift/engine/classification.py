"""
MutualFundDrift — portfolio classification module.
Classifies individual holdings by SEBI market cap category,
computes sector weights, and calculates the Herfindahl-Hirschman Index.
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def classify_holding(isin: str, nse_df: pd.DataFrame) -> str:
    """
    Classify a single stock holding as 'large_cap', 'mid_cap', 'small_cap',
    or 'unclassified' based on the AMFI/NSE market cap rank list.

    Args:
        isin: The ISIN string to look up.
        nse_df: DataFrame from load_nse_classification() with an 'isin' and 'category' column.

    Returns:
        One of 'large_cap', 'mid_cap', 'small_cap', or 'unclassified'.
    """
    if not isin or not isinstance(isin, str):
        return "unclassified"

    isin_clean = isin.strip().upper()

    # Case-insensitive ISIN lookup; handle minor formatting differences
    nse_isin_col = nse_df["isin"].str.strip().str.upper()
    match = nse_df[nse_isin_col == isin_clean]

    if match.empty:
        # Try partial match for ISINs with minor suffixes
        match = nse_df[nse_isin_col.str.startswith(isin_clean[:11])]

    if match.empty:
        return "unclassified"

    category = match.iloc[0]["category"]
    return str(category).lower().replace(" ", "_")


def classify_portfolio(holdings_df: pd.DataFrame, nse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every holding in a portfolio DataFrame and add a 'cap_category' column.

    Args:
        holdings_df: Portfolio DataFrame with at minimum an 'isin' column.
        nse_df: NSE classification DataFrame from load_nse_classification().

    Returns:
        A copy of holdings_df with an additional 'cap_category' column.
    """
    if holdings_df.empty:
        result = holdings_df.copy()
        result["cap_category"] = pd.Series(dtype=str)
        return result

    result = holdings_df.copy()
    result["cap_category"] = result["isin"].apply(
        lambda isin: classify_holding(isin, nse_df)
    )

    unclassified_count = (result["cap_category"] == "unclassified").sum()
    if unclassified_count > 0:
        logger.warning(
            "%d holdings could not be classified using NSE data — "
            "they will be treated as mid_cap (neutral) in style box computation.",
            unclassified_count,
        )

    return result


def get_sector_weights(holdings_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the percentage weight of each GICS sector in the portfolio.

    Normalises weights to sum to 100%. Holdings with missing sector data
    are grouped under 'Other'.

    Args:
        holdings_df: Portfolio DataFrame with 'sector' and 'pct_of_nav' columns.

    Returns:
        Dict mapping sector name to its portfolio weight as a percentage (0–100).
    """
    if holdings_df.empty or "sector" not in holdings_df.columns:
        return {}

    df = holdings_df.copy()
    df["sector"] = df["sector"].fillna("Other").replace("", "Other")
    df["pct_of_nav"] = pd.to_numeric(df["pct_of_nav"], errors="coerce").fillna(0.0)

    sector_weights = df.groupby("sector")["pct_of_nav"].sum()
    total = sector_weights.sum()

    if total == 0:
        logger.warning("Total pct_of_nav is zero; cannot compute sector weights.")
        return {}

    # Normalise to sum to 100%
    normalised = (sector_weights / total * 100).to_dict()
    return {k: round(v, 4) for k, v in normalised.items()}


def compute_hhi(sector_weights: Dict[str, float]) -> float:
    """
    Compute the Herfindahl-Hirschman Index (HHI) for portfolio sector concentration.

    HHI = sum of (w_i)^2 where w_i is each sector's weight expressed as a decimal
    (i.e., 30% → 0.30). A perfectly concentrated single-sector fund scores 1.0.
    A perfectly diversified fund with equal 10-sector allocation scores 0.10.

    Args:
        sector_weights: Dict of {sector_name: percentage_weight} where weights sum to 100.

    Returns:
        HHI as a float between 0.0 and 1.0.
    """
    if not sector_weights:
        return 0.0

    total = sum(sector_weights.values())
    if total == 0:
        return 0.0

    hhi = sum((w / 100.0) ** 2 for w in sector_weights.values())
    return round(hhi, 6)
