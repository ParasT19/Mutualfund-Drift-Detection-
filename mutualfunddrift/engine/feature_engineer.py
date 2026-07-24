"""
MutualFundDrift — feature engineering module for the XGBoost predictor.
Converts time-series portfolio snapshots into a flat feature vector
suitable for drift prediction modelling.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from backend.config import settings
from engine.drift_scorer import compute_drift_velocity

logger = logging.getLogger(__name__)

# The 19 canonical feature names used by the XGBoost model
FEATURE_NAMES: List[str] = [
    "current_drift_score",
    "drift_velocity",
    "drift_acceleration",
    "size_score_current",
    "size_score_delta_4q",
    "style_score_current",
    "style_score_delta_4q",
    "rolling_corr_current",
    "rolling_corr_delta_4q",
    "hhi_sector_current",
    "hhi_sector_delta_4q",
    "top10_holdings_current",
    "large_cap_pct_delta_4q",
    # Lag features at t-1
    "drift_score_lag1",
    "size_score_lag1",
    "style_score_lag1",
    # Lag features at t-2
    "drift_score_lag2",
    "size_score_lag2",
    "style_score_lag2",
]


def _safe_delta(series: List[float], periods: int) -> float:
    """Compute the difference between the last value and the value `periods` steps ago."""
    if len(series) <= periods:
        return 0.0
    return float(series[-1] - series[-(periods + 1)])


def build_feature_vector(snapshots: List[dict]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from the last 8 portfolio snapshots.

    Takes a chronologically ordered list of PortfolioSnapshot dicts and engineers
    13 primary features plus 6 lag features, for a total of 19 named features.

    Feature definitions:
      1.  current_drift_score      — most recent drift_score
      2.  drift_velocity           — linear slope of last 4 drift scores
      3.  drift_acceleration       — slope of 4-window velocities across 8 quarters
      4.  size_score_current       — most recent size_score
      5.  size_score_delta_4q      — change in size_score over last 4 snapshots
      6.  style_score_current      — most recent style_score
      7.  style_score_delta_4q     — change in style_score over last 4 snapshots
      8.  rolling_corr_current     — most recent rolling_corr (1.0 if null)
      9.  rolling_corr_delta_4q    — change in rolling_corr over 4 snapshots
      10. hhi_sector_current       — most recent sector HHI
      11. hhi_sector_delta_4q      — change in HHI over 4 snapshots
      12. top10_holdings_current   — most recent top10_holdings_pct
      13. large_cap_pct_delta_4q   — change in large_cap_pct over 4 snapshots
      14–16. Lag t-1 for drift, size, style
      17–19. Lag t-2 for drift, size, style

    Args:
        snapshots: List of dicts representing PortfolioSnapshot records (chronological).
                   Should ideally contain 8+ snapshots; fewer are handled gracefully.

    Returns:
        Single-row pd.DataFrame with 19 feature columns matching FEATURE_NAMES.
    """
    if not snapshots:
        logger.warning("build_feature_vector called with empty snapshot list; returning zeros.")
        return pd.DataFrame([{f: 0.0 for f in FEATURE_NAMES}])

    drift_scores = [s.get("drift_score", 0.0) for s in snapshots]
    size_scores = [s.get("size_score", 0.5) for s in snapshots]
    style_scores = [s.get("style_score", 0.5) for s in snapshots]
    rolling_corrs = [s.get("rolling_corr") or 1.0 for s in snapshots]
    hhi_values = [s.get("hhi_sector", 0.1) for s in snapshots]
    top10_values = [s.get("top10_holdings_pct", 30.0) for s in snapshots]
    large_cap_pcts = [s.get("large_cap_pct", 33.0) for s in snapshots]

    # Core features
    current_drift = drift_scores[-1]
    drift_velocity = compute_drift_velocity(drift_scores[-4:]) if len(drift_scores) >= 4 else 0.0

    # Drift acceleration: velocity of velocities across rolling 4-windows
    if len(drift_scores) >= 6:
        v1 = compute_drift_velocity(drift_scores[-6:-2])
        v2 = compute_drift_velocity(drift_scores[-4:])
        drift_acceleration = v2 - v1
    else:
        drift_acceleration = 0.0

    features = {
        "current_drift_score": current_drift,
        "drift_velocity": drift_velocity,
        "drift_acceleration": drift_acceleration,
        "size_score_current": size_scores[-1],
        "size_score_delta_4q": _safe_delta(size_scores, 4),
        "style_score_current": style_scores[-1],
        "style_score_delta_4q": _safe_delta(style_scores, 4),
        "rolling_corr_current": rolling_corrs[-1],
        "rolling_corr_delta_4q": _safe_delta(rolling_corrs, 4),
        "hhi_sector_current": hhi_values[-1],
        "hhi_sector_delta_4q": _safe_delta(hhi_values, 4),
        "top10_holdings_current": top10_values[-1],
        "large_cap_pct_delta_4q": _safe_delta(large_cap_pcts, 4),
        # Lag features
        "drift_score_lag1": drift_scores[-2] if len(drift_scores) >= 2 else current_drift,
        "size_score_lag1": size_scores[-2] if len(size_scores) >= 2 else size_scores[-1],
        "style_score_lag1": style_scores[-2] if len(style_scores) >= 2 else style_scores[-1],
        "drift_score_lag2": drift_scores[-3] if len(drift_scores) >= 3 else current_drift,
        "size_score_lag2": size_scores[-3] if len(size_scores) >= 3 else size_scores[-1],
        "style_score_lag2": style_scores[-3] if len(style_scores) >= 3 else style_scores[-1],
    }

    return pd.DataFrame([features])[FEATURE_NAMES]


def prepare_training_data(
    all_snapshots_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build labelled training data for the XGBoost classifier.

    For each fund in the dataset, and for each time point where 8+ historical
    snapshots exist, this function:
      1. Builds a feature vector from the preceding 8 snapshots.
      2. Labels the row as 1 if drift_score exceeded DRIFT_ALERT_THRESHOLD
         within the next 4 snapshots, otherwise 0.

    Args:
        all_snapshots_df: Full DataFrame of PortfolioSnapshot records with columns
                          scheme_code, snapshot_date, drift_score, size_score,
                          style_score, rolling_corr, hhi_sector, top10_holdings_pct,
                          large_cap_pct.

    Returns:
        Tuple (X: pd.DataFrame of feature rows, y: pd.Series of binary labels).
    """
    threshold = settings.drift_alert_threshold

    all_X_rows: List[dict] = []
    all_labels: List[int] = []

    if all_snapshots_df.empty:
        logger.warning("prepare_training_data called with empty DataFrame.")
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    for scheme_code, group in all_snapshots_df.groupby("scheme_code"):
        group = group.sort_values("snapshot_date").reset_index(drop=True)
        snapshots_list = group.to_dict("records")

        # Need at least 4 historical + 2 future snapshots = 6 minimum
        for i in range(4, len(snapshots_list) - 2):
            history = snapshots_list[i - 4 : i]
            future = snapshots_list[i : i + 2]

            fv = build_feature_vector(history)
            label = int(any(s["drift_score"] > threshold for s in future))

            all_X_rows.append(fv.iloc[0].to_dict())
            all_labels.append(label)

    if not all_X_rows:
        logger.warning(
            "No training samples generated — ensure at least 12 snapshots per fund."
        )
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    X = pd.DataFrame(all_X_rows, columns=FEATURE_NAMES)
    y = pd.Series(all_labels, name="label", dtype=int)
    logger.info(
        "Training data prepared: %d samples, class balance: %d positive / %d negative",
        len(y), y.sum(), (y == 0).sum(),
    )
    return X, y
