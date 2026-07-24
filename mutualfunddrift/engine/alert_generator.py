"""
MutualFundDrift — Rule-based investor alert generation module.
Generates detailed, data-driven plain-English investor alerts using
drift metrics, severity classifications, and portfolio composition data.
No external API key required — fully self-contained.
"""

import logging
from typing import Optional

from engine.drift_scorer import classify_drift_severity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Severity-specific alert templates
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_OPENING = {
    "normal": "is performing within its SEBI-mandated investment parameters",
    "watch":  "is showing early signs of style drift that warrant monitoring",
    "amber":  "has drifted meaningfully from its SEBI-mandated investment style",
    "red":    "has significantly deviated from its SEBI-mandated investment mandate",
}

_TREND_PHRASES = {
    "worsening": "Drift has been accelerating over recent quarters",
    "improving": "Drift is improving — the portfolio is moving back toward mandate",
    "stable":    "Drift has remained relatively stable over recent quarters",
}

_SIZE_LABELS = {
    "large_cap": "large cap",
    "mid_cap":   "mid cap",
    "small_cap": "small cap",
}

_CATEGORY_EXPECTED_SIZE = {
    "Large Cap Fund":   "large cap",
    "Mid Cap Fund":     "mid cap",
    "Small Cap Fund":   "small cap",
    "Flexi Cap Fund":   "diversified",
    "Multi Cap Fund":   "diversified",
    "ELSS":             "diversified",
}


def _size_label(size_score: float) -> str:
    """Convert a numeric size score to a human-readable cap label."""
    if size_score > 0.67:
        return "large cap"
    elif size_score >= 0.33:
        return "mid cap"
    return "small cap"


def _trend_word(drift_velocity: float) -> str:
    """Convert a drift velocity to a human-readable trend descriptor."""
    if drift_velocity > 0.005:
        return "worsening"
    elif drift_velocity < -0.005:
        return "improving"
    return "stable"


def generate_investor_alert(
    fund: dict,
    snapshot: dict,
    drift_velocity: float,
    shap_features: dict,
) -> str:
    """
    Generate a detailed, data-driven plain-English investor alert using
    only the drift metrics passed in. No external API or key required.

    Constructs a multi-sentence alert covering:
      - Drift score and severity relative to SEBI mandate
      - Portfolio composition shift (large/mid/small cap breakdown)
      - Rolling benchmark correlation drop (if available)
      - Drift trend direction (worsening/improving/stable)
      - Sector concentration warning (if HHI is elevated)
      - Active share observation (if data is available)
      - SHAP-driven explanation of the top risk factor
      - A clear action recommendation based on severity

    Args:
        fund: Dict with keys: scheme_name, category, amc_name,
              mandate_size_score, mandate_style_score.
        snapshot: Dict with keys: drift_score, size_score, style_score,
                  large_cap_pct, mid_cap_pct, small_cap_pct,
                  rolling_corr, hhi_sector, active_share.
        drift_velocity: Slope of drift score over recent quarters.
                        Positive = worsening, negative = recovering.
        shap_features: Dict of {feature_name: shap_value} for top predictors.

    Returns:
        A plain-English alert string of at most 1000 characters.
    """
    fund_name   = fund.get("scheme_name", "This fund")
    category    = fund.get("category", "its stated category")
    amc         = fund.get("amc_name", "the AMC")
    mandate_sz  = fund.get("mandate_size_score", 0.5)
    mandate_st  = fund.get("mandate_style_score", 0.5)

    drift_score  = snapshot.get("drift_score", 0.0)
    size_score   = snapshot.get("size_score", mandate_sz)
    style_score  = snapshot.get("style_score", mandate_st)
    large_pct    = snapshot.get("large_cap_pct", 0.0)
    mid_pct      = snapshot.get("mid_cap_pct", 0.0)
    small_pct    = snapshot.get("small_cap_pct", 0.0)
    rolling_corr = snapshot.get("rolling_corr")
    hhi          = snapshot.get("hhi_sector", 0.0)
    active_share = snapshot.get("active_share")

    severity    = classify_drift_severity(drift_score)
    trend       = _trend_word(drift_velocity)
    current_sz  = _size_label(size_score)
    expected_sz = _CATEGORY_EXPECTED_SIZE.get(category, "its mandated")

    opening_desc = _SEVERITY_OPENING.get(severity, "is under review")
    trend_phrase = _TREND_PHRASES.get(trend, "")

    # ── Sentence 1: Core drift statement ─────────────────────────────────────
    sentence_1 = (
        f"{fund_name} ({amc}) {opening_desc}, "
        f"with a current style drift score of {drift_score:.3f} "
        f"(Severity: {severity.upper()}). "
    )

    # ── Sentence 2: Portfolio composition vs. mandate ─────────────────────────
    if abs(size_score - mandate_sz) > 0.1:
        sentence_2 = (
            f"The portfolio is positioned as {current_sz} "
            f"({large_pct:.0f}% large / {mid_pct:.0f}% mid / {small_pct:.0f}% small cap) "
            f"while the {category} mandate expects primarily {expected_sz} exposure. "
        )
    else:
        sentence_2 = (
            f"Portfolio composition stands at {large_pct:.0f}% large / "
            f"{mid_pct:.0f}% mid / {small_pct:.0f}% small cap, "
            f"broadly in line with the {category} mandate. "
        )

    # ── Sentence 3: Correlation / trend / HHI signal ─────────────────────────
    signals = []
    if rolling_corr is not None and rolling_corr < 0.85:
        signals.append(
            f"rolling benchmark correlation has weakened to {rolling_corr:.2f} "
            f"(threshold: 0.85), indicating the fund's returns diverge from "
            f"its {category} benchmark"
        )
    if hhi and hhi > 0.18:
        signals.append(
            f"sector concentration (HHI: {hhi:.2f}) is elevated, "
            f"suggesting the portfolio is over-weight in fewer sectors than expected"
        )
    if active_share is not None and active_share < 0.60:
        signals.append(
            f"active share of {active_share:.0%} is below 60%, "
            f"meaning the fund resembles a closet index fund at an active expense ratio"
        )

    if signals:
        sentence_3 = f"{trend_phrase}; {'; '.join(signals[:2])}. "
    else:
        sentence_3 = f"{trend_phrase}. "

    # ── Sentence 4: SHAP-driven key risk driver ───────────────────────────────
    if shap_features:
        top_feature = list(shap_features.keys())[0]
        top_value   = list(shap_features.values())[0]
        direction   = "increasing" if top_value > 0 else "decreasing"
        readable    = top_feature.replace("_", " ")
        sentence_4  = (
            f"The primary risk driver is {readable} ({direction}). "
        )
    else:
        sentence_4 = ""

    # ── Sentence 5: Recommendation based on severity ─────────────────────────
    recommendations = {
        "normal": (
            f"No immediate action required — continue to monitor quarterly disclosures."
        ),
        "watch": (
            f"Investors should monitor the next 1–2 quarterly disclosures closely "
            f"to confirm whether drift is a temporary tactical shift or a structural change."
        ),
        "amber": (
            f"Investors who purchased this fund for {expected_sz} exposure should review "
            f"whether it still matches their intended asset allocation, and consider "
            f"rebalancing if the drift persists into the next quarter."
        ),
        "red": (
            f"Investors seeking pure {expected_sz} exposure should urgently review "
            f"this holding — the fund's current risk-return profile no longer matches "
            f"a standard {category} mandate. Consider switching to a compliant alternative."
        ),
    }
    sentence_5 = recommendations.get(severity, "Please review your investment.")

    full_alert = sentence_1 + sentence_2 + sentence_3 + sentence_4 + sentence_5
    logger.info(
        "Rule-based alert generated for %s (drift=%.3f, severity=%s)",
        fund_name, drift_score, severity,
    )
    return full_alert[:1000]

