"""
MutualFundDrift — drift scoring module.
Computes Euclidean drift scores, severity classifications, drift velocity,
rolling benchmark correlations, and active share.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_drift_score(
    size_score: float,
    style_score: float,
    mandate_size: float,
    mandate_style: float,
) -> float:
    """
    Compute the Euclidean distance between the current style coordinate
    and the fund's SEBI-mandated target coordinate.

    Formula: sqrt((size_score - mandate_size)^2 + (style_score - mandate_style)^2)
    Maximum possible value is sqrt(2) ≈ 1.414 (when at opposite corners of the grid).
    A score of 0.0 means perfect alignment with mandate.

    Args:
        size_score: Current computed size score in [0.0, 1.0].
        style_score: Current computed style score in [0.0, 1.0].
        mandate_size: Fund's mandated size score in [0.0, 1.0].
        mandate_style: Fund's mandated style score in [0.0, 1.0].

    Returns:
        Float drift score in [0.0, sqrt(2)].
    """
    drift = np.sqrt(
        (size_score - mandate_size) ** 2 + (style_score - mandate_style) ** 2
    )
    return float(round(drift, 6))


def classify_drift_severity(drift_score: float) -> str:
    """
    Classify the drift score into a human-readable severity level.

    Severity bands:
      < 0.15  → 'normal'   (within acceptable mandate range)
      0.15–0.25 → 'watch'  (monitor closely)
      0.25–0.35 → 'amber'  (action recommended)
      > 0.35  → 'red'      (significant mandate violation)

    Args:
        drift_score: Float drift score computed by compute_drift_score().

    Returns:
        Severity string: 'normal', 'watch', 'amber', or 'red'.
    """
    if drift_score < 0.15:
        return "normal"
    elif drift_score < 0.25:
        return "watch"
    elif drift_score < 0.35:
        return "amber"
    else:
        return "red"


def compute_drift_velocity(drift_scores: List[float]) -> float:
    """
    Compute the linear regression slope of the drift score series over time.

    A positive slope indicates worsening drift; negative indicates recovery.
    Uses numpy polyfit with degree 1 on the time index [0, 1, 2, ..., N-1].

    Args:
        drift_scores: Chronologically ordered list of drift scores.
                      Returns 0.0 gracefully if fewer than 3 data points.

    Returns:
        Float slope (velocity). Positive = drift worsening, negative = improving.
    """
    if len(drift_scores) < 3:
        logger.debug(
            "compute_drift_velocity called with fewer than 3 data points; returning 0.0"
        )
        return 0.0

    x = np.arange(len(drift_scores), dtype=float)
    y = np.array(drift_scores, dtype=float)

    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception as exc:
        logger.error("polyfit failed in compute_drift_velocity: %s", exc)
        return 0.0


def compute_rolling_correlation(
    fund_nav_series: pd.Series,
    benchmark_series: pd.Series,
    window: int = 12,
) -> Optional[float]:
    """
    Compute the rolling Pearson correlation of fund NAV returns against
    a benchmark index return series over a given rolling window.

    Both series should be monthly NAV values (or returns). If either series
    has fewer than `window` data points, returns None.

    Args:
        fund_nav_series: Monthly NAV or return series for the fund (pd.Series).
        benchmark_series: Monthly NAV or return series for the benchmark (pd.Series).
        window: Rolling window in months (default 12).

    Returns:
        The most recent rolling correlation value as a float, or None if insufficient data.
    """
    try:
        if len(fund_nav_series) < window or len(benchmark_series) < window:
            logger.debug(
                "Insufficient data for rolling correlation (fund=%d, bench=%d, window=%d)",
                len(fund_nav_series), len(benchmark_series), window,
            )
            return None

        # Compute monthly returns from NAV series
        fund_returns = fund_nav_series.pct_change().dropna()
        bench_returns = benchmark_series.pct_change().dropna()

        # Align on common index
        aligned = pd.concat(
            [fund_returns.rename("fund"), bench_returns.rename("bench")], axis=1
        ).dropna()

        if len(aligned) < window:
            return None

        rolling_corr = aligned["fund"].rolling(window=window).corr(aligned["bench"])
        latest = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else None
        return float(latest) if latest is not None else None

    except Exception as exc:
        logger.error("compute_rolling_correlation failed: %s", exc)
        return None


def compute_active_share(
    holdings_df: pd.DataFrame,
    benchmark_holdings_df: pd.DataFrame,
) -> float:
    """
    Compute active share — the degree to which the portfolio deviates from its benchmark.

    Active Share = 0.5 * sum(|fund_weight_i - benchmark_weight_i|) for all ISINs.
    ISINs present in the fund but not the benchmark have benchmark_weight = 0,
    and vice versa.

    Args:
        holdings_df: Fund portfolio DataFrame with 'isin' and 'pct_of_nav' columns.
        benchmark_holdings_df: Benchmark index constituents with 'isin' and 'pct_of_nav'.

    Returns:
        Active share as a float in [0.0, 1.0]. Returns 1.0 if benchmark data unavailable.
    """
    if benchmark_holdings_df is None or benchmark_holdings_df.empty:
        logger.warning(
            "No benchmark holdings data available; active share defaulting to 1.0 (fully active)."
        )
        return 1.0

    if holdings_df is None or holdings_df.empty:
        return 1.0

    try:
        fund = (
            holdings_df[["isin", "pct_of_nav"]]
            .copy()
            .rename(columns={"pct_of_nav": "fund_weight"})
        )
        fund["fund_weight"] = pd.to_numeric(fund["fund_weight"], errors="coerce").fillna(0.0)
        fund["fund_weight"] /= fund["fund_weight"].sum()  # normalise to 0–1

        bench = (
            benchmark_holdings_df[["isin", "pct_of_nav"]]
            .copy()
            .rename(columns={"pct_of_nav": "bench_weight"})
        )
        bench["bench_weight"] = pd.to_numeric(bench["bench_weight"], errors="coerce").fillna(0.0)
        bench["bench_weight"] /= bench["bench_weight"].sum()

        merged = fund.merge(bench, on="isin", how="outer").fillna(0.0)
        active_share = 0.5 * (merged["fund_weight"] - merged["bench_weight"]).abs().sum()
        return float(np.clip(active_share, 0.0, 1.0))

    except Exception as exc:
        logger.error("compute_active_share failed: %s", exc)
        return 1.0
