"""
MutualFundDrift — style box coordinate computation module.
This is the core mathematical engine of the drift detection system.
Computes (size_score, style_score) coordinates for placement in a
Morningstar-style 3x3 grid and classifies the resulting style box cell.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numeric value assigned to each market cap category for size axis calculation
CAP_CATEGORY_VALUES: dict[str, float] = {
    "large_cap": 1.0,
    "mid_cap": 0.5,
    "small_cap": 0.0,
    "unclassified": 0.5,  # Neutral — treated as mid-cap with a warning logged
}


def compute_size_score(holdings_df: pd.DataFrame) -> float:
    """
    Compute the weighted average size score for the portfolio on the size axis.

    Each holding is assigned a numeric size value based on its cap_category:
      large_cap    → 1.0
      mid_cap      → 0.5
      small_cap    → 0.0
      unclassified → 0.5 (neutral, triggers a warning)

    The weighted average uses pct_of_nav as weights.

    Args:
        holdings_df: Portfolio DataFrame with columns 'cap_category' and 'pct_of_nav'.
                     Must be non-empty.

    Returns:
        Float between 0.0 (pure small cap) and 1.0 (pure large cap).

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if holdings_df is None or holdings_df.empty:
        raise ValueError(
            "compute_size_score requires a non-empty holdings DataFrame. "
            "Ensure portfolio data was loaded successfully before calling this function."
        )

    required = {"cap_category", "pct_of_nav"}
    missing = required - set(holdings_df.columns)
    if missing:
        raise ValueError(f"holdings_df missing required columns: {missing}")

    df = holdings_df.copy()
    df["pct_of_nav"] = pd.to_numeric(df["pct_of_nav"], errors="coerce").fillna(0.0)

    # Log unclassified holdings as warning
    unclassified = df[df["cap_category"] == "unclassified"]
    if not unclassified.empty:
        logger.warning(
            "%d holdings are unclassified (will be treated as mid_cap, weight=%.2f%%).",
            len(unclassified),
            unclassified["pct_of_nav"].sum(),
        )

    df["size_value"] = df["cap_category"].map(CAP_CATEGORY_VALUES).fillna(0.5)

    total_weight = df["pct_of_nav"].sum()
    if total_weight == 0:
        logger.warning("Total pct_of_nav is 0; returning neutral size_score of 0.5")
        return 0.5

    weighted_size = (df["size_value"] * df["pct_of_nav"]).sum() / total_weight
    return float(np.clip(weighted_size, 0.0, 1.0))


def compute_style_score(
    holdings_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> float:
    """
    Compute the weighted average style score for the portfolio on the style axis.

    For each holding, the style score is derived from its P/B ratio relative
    to the median P/B of its market cap category:
      - relative_pb = holding_pb / category_median_pb
      - style_raw   = relative_pb (high PB → growth, low PB → value)
    Scores are then min-max normalised across the portfolio to [0.0, 1.0].
    Holdings without P/B data receive a neutral score of 0.5.

    Args:
        holdings_df: Portfolio DataFrame with 'isin', 'pct_of_nav', 'cap_category'.
        fundamentals_df: DataFrame from fetch_stock_fundamentals() with 'isin', 'pb_ratio'.

    Returns:
        Float between 0.0 (deep value) and 1.0 (pure growth).
    """
    if holdings_df is None or holdings_df.empty:
        logger.warning("Empty holdings_df passed to compute_style_score; returning 0.5")
        return 0.5

    df = holdings_df.copy()
    df["pct_of_nav"] = pd.to_numeric(df["pct_of_nav"], errors="coerce").fillna(0.0)

    # Merge fundamentals on ISIN
    if fundamentals_df is not None and not fundamentals_df.empty:
        fund_df = fundamentals_df[["isin", "pb_ratio"]].copy()
        fund_df["pb_ratio"] = pd.to_numeric(fund_df["pb_ratio"], errors="coerce")
        df = df.merge(fund_df, on="isin", how="left")
    else:
        df["pb_ratio"] = np.nan

    missing_pb = df["pb_ratio"].isna().sum()
    if missing_pb > 0:
        logger.warning(
            "%d holdings have no P/B ratio data — assigned neutral style score 0.5.",
            missing_pb,
        )

    # Compute category median P/B
    df["cap_category"] = df.get("cap_category", "unclassified")
    category_medians: dict[str, float] = {}
    for cat in df["cap_category"].unique():
        cat_df = df[df["cap_category"] == cat]
        median_pb = cat_df["pb_ratio"].dropna().median()
        category_medians[cat] = median_pb if not np.isnan(median_pb) else 2.0

    # Compute relative P/B and raw style score
    def _relative_pb(row: pd.Series) -> float:
        if pd.isna(row["pb_ratio"]) or row["pb_ratio"] <= 0:
            return 1.0  # neutral
        cat_med = category_medians.get(row["cap_category"], 2.0)
        if cat_med <= 0:
            return 1.0
        return float(row["pb_ratio"] / cat_med)

    df["style_raw"] = df.apply(_relative_pb, axis=1)

    # Min-max normalise style_raw across the portfolio
    raw_min = df["style_raw"].min()
    raw_max = df["style_raw"].max()
    if raw_max == raw_min:
        df["style_normalised"] = 0.5
    else:
        df["style_normalised"] = (df["style_raw"] - raw_min) / (raw_max - raw_min)

    # Fill NaN normalised scores with neutral 0.5
    df["style_normalised"] = df["style_normalised"].fillna(0.5)

    total_weight = df["pct_of_nav"].sum()
    if total_weight == 0:
        return 0.5

    weighted_style = (df["style_normalised"] * df["pct_of_nav"]).sum() / total_weight
    return float(np.clip(weighted_style, 0.0, 1.0))


def compute_style_box_coordinate(
    holdings_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    nse_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Orchestrate the computation of both style box axes for a portfolio.

    Adds 'cap_category' via classification if not already present, then
    computes size_score and style_score.

    Args:
        holdings_df: Raw portfolio DataFrame with 'isin' and 'pct_of_nav'.
        fundamentals_df: Stock fundamentals DataFrame with 'isin' and 'pb_ratio'.
        nse_df: NSE market cap classification DataFrame.

    Returns:
        Tuple (size_score, style_score), both floats in [0.0, 1.0].
    """
    from engine.classification import classify_portfolio

    if "cap_category" not in holdings_df.columns:
        holdings_df = classify_portfolio(holdings_df, nse_df)

    size = compute_size_score(holdings_df)
    style = compute_style_score(holdings_df, fundamentals_df)

    logger.info(
        "Style box coordinate computed: size_score=%.4f, style_score=%.4f",
        size, style,
    )
    return (size, style)


def classify_style_box_cell(size_score: float, style_score: float) -> str:
    """
    Map a (size_score, style_score) coordinate to one of the 9 Morningstar-style box cells.

    Size axis:  size_score > 0.67 → Large, 0.33–0.67 → Mid, < 0.33 → Small
    Style axis: style_score < 0.33 → Value, 0.33–0.67 → Blend, > 0.67 → Growth

    Args:
        size_score: Float in [0.0, 1.0] — the fund's position on the size axis.
        style_score: Float in [0.0, 1.0] — the fund's position on the style axis.

    Returns:
        String like 'large_value', 'mid_blend', 'small_growth', etc.
    """
    if size_score > 0.67:
        size_label = "large"
    elif size_score >= 0.33:
        size_label = "mid"
    else:
        size_label = "small"

    if style_score < 0.33:
        style_label = "value"
    elif style_score <= 0.67:
        style_label = "blend"
    else:
        style_label = "growth"

    return f"{size_label}_{style_label}"
