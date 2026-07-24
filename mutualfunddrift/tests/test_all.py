"""
MutualFundDrift — All unit tests in one file.
Covers: drift_scorer, style_box, and Pydantic schemas.
Run with: python -m pytest tests/test_all.py -v
"""

import math
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from engine.drift_scorer import (
    classify_drift_severity,
    compute_active_share,
    compute_drift_score,
    compute_drift_velocity,
    compute_rolling_correlation,
)
from engine.style_box import (
    classify_style_box_cell,
    compute_size_score,
    compute_style_box_coordinate,
    compute_style_score,
)
from backend.schemas import (
    DriftAlertCreate,
    DriftSummary,
    FundCreate,
    PortfolioSnapshotCreate,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _holdings(cap_categories, weights, include_pb=True):
    """Build a minimal holdings DataFrame."""
    df = pd.DataFrame({
        "isin":         [f"INE{i:06d}A01000" for i in range(len(cap_categories))],
        "company_name": [f"Company {i}"       for i in range(len(cap_categories))],
        "cap_category": cap_categories,
        "pct_of_nav":   weights,
        "sector":       ["Financial Services"] * len(cap_categories),
    })
    if include_pb:
        df["pb_ratio"] = 2.5
    return df


def _fundamentals(isins, pb_ratios):
    """Build a minimal fundamentals DataFrame."""
    return pd.DataFrame({
        "isin":             isins,
        "ticker":           [f"TEST{i}" for i in range(len(isins))],
        "pb_ratio":         pb_ratios,
        "market_cap_crore": [50000.0] * len(isins),
    })


# ── Drift Scorer Tests ────────────────────────────────────────────────────────

def test_drift_score_at_mandate_is_zero():
    """Fund exactly at mandate coordinate → drift score = 0.0."""
    assert compute_drift_score(0.5, 0.5, 0.5, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_drift_score_maximum_distance():
    """Fund at opposite corner from mandate → drift ≈ sqrt(2) ≈ 1.414."""
    assert compute_drift_score(0.0, 0.0, 1.0, 1.0) == pytest.approx(math.sqrt(2), abs=1e-5)


def test_severity_normal():
    assert classify_drift_severity(0.10) == "normal"

def test_severity_watch():
    assert classify_drift_severity(0.20) == "watch"

def test_severity_amber():
    assert classify_drift_severity(0.30) == "amber"

def test_severity_red():
    assert classify_drift_severity(0.40) == "red"


def test_drift_velocity_increasing():
    """Increasing drift scores → positive velocity slope."""
    assert compute_drift_velocity([0.10, 0.15, 0.20, 0.25, 0.30]) > 0


def test_drift_velocity_decreasing():
    """Decreasing drift scores → negative velocity slope."""
    assert compute_drift_velocity([0.30, 0.25, 0.20, 0.15, 0.10]) < 0


@pytest.mark.parametrize("scores", [[], [0.1], [0.1, 0.2]])
def test_drift_velocity_insufficient_data(scores):
    """Fewer than 3 data points → returns 0.0 gracefully."""
    assert compute_drift_velocity(scores) == 0.0


def test_rolling_correlation_identical_series():
    """Correlation of a series with itself → 1.0."""
    nav = pd.Series([100,102,105,103,108,112,110,115,118,120,122,125,128], dtype=float)
    result = compute_rolling_correlation(nav, nav.copy(), window=12)
    assert result is not None
    assert result == pytest.approx(1.0, abs=1e-6)


# ── Style Box Tests ───────────────────────────────────────────────────────────

def test_size_score_all_large_cap():
    """100% large cap → size_score = 1.0."""
    h = _holdings(["large_cap","large_cap","large_cap"], [40.0,35.0,25.0])
    assert compute_size_score(h) == pytest.approx(1.0, abs=1e-9)


def test_size_score_all_small_cap():
    """100% small cap → size_score = 0.0."""
    h = _holdings(["small_cap","small_cap","small_cap"], [50.0,30.0,20.0])
    assert compute_size_score(h) == pytest.approx(0.0, abs=1e-9)


def test_size_score_equal_split():
    """Equal large/mid/small split → size_score ≈ 0.5."""
    h = _holdings(["large_cap","mid_cap","small_cap"], [33.33,33.33,33.34])
    assert compute_size_score(h) == pytest.approx(0.5, abs=0.01)


def test_style_box_coordinate_valid_range():
    """style_box_coordinate returns (size, style) both in [0.0, 1.0]."""
    h = _holdings(["large_cap","mid_cap","small_cap"], [40.0,40.0,20.0], include_pb=False)
    isins = h["isin"].tolist()
    f = _fundamentals(isins, [3.0, 2.0, 1.5])
    nse = pd.DataFrame({
        "rank": [1,51,260], "isin": isins,
        "category": ["large_cap","mid_cap","small_cap"],
        "company_name": ["Big Co","Mid Co","Small Co"],
        "ticker": ["BIG.NS","MID.NS","SML.NS"],
        "market_cap_crore": [500000,50000,5000],
    })
    size, style = compute_style_box_coordinate(h, f, nse)
    assert 0.0 <= size  <= 1.0
    assert 0.0 <= style <= 1.0


def test_classify_large_growth():
    assert classify_style_box_cell(0.8, 0.8) == "large_growth"

def test_classify_mid_blend():
    assert classify_style_box_cell(0.5, 0.5) == "mid_blend"

def test_classify_small_value():
    assert classify_style_box_cell(0.1, 0.1) == "small_value"


def test_size_score_empty_df_raises():
    """Empty DataFrame → raises ValueError."""
    empty = pd.DataFrame(columns=["isin","cap_category","pct_of_nav"])
    with pytest.raises(ValueError, match="non-empty"):
        compute_size_score(empty)


def test_unclassified_treated_as_mid_cap():
    """Unclassified holdings → size_score = 0.5 (mid-neutral)."""
    h = _holdings(["unclassified","unclassified"], [60.0,40.0])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = compute_size_score(h)
    assert result == pytest.approx(0.5, abs=1e-9)


# ── Schema Tests ──────────────────────────────────────────────────────────────

def test_portfolio_snapshot_valid():
    """Valid PortfolioSnapshotCreate → no errors."""
    snap = PortfolioSnapshotCreate(
        scheme_code="120503", snapshot_date=date(2024,3,1),
        size_score=0.65, style_score=0.55,
        large_cap_pct=35.0, mid_cap_pct=55.0, small_cap_pct=10.0,
        top10_holdings_pct=48.0, hhi_sector=0.12,
        active_share=0.74, turnover_ratio=0.35,
        drift_score=0.21, rolling_corr=0.89,
    )
    assert snap.scheme_code == "120503"
    assert snap.drift_score == pytest.approx(0.21)


def test_portfolio_snapshot_drift_score_too_high():
    """drift_score > 2.0 → ValidationError."""
    with pytest.raises(ValidationError) as exc:
        PortfolioSnapshotCreate(scheme_code="120503", snapshot_date=date(2024,3,1),
                                size_score=0.5, style_score=0.5, drift_score=2.5)
    assert any("drift_score" in loc for loc in [e["loc"] for e in exc.value.errors()])


def test_portfolio_snapshot_size_score_negative():
    """size_score < 0 → ValidationError."""
    with pytest.raises(ValidationError) as exc:
        PortfolioSnapshotCreate(scheme_code="120503", snapshot_date=date(2024,3,1),
                                size_score=-0.1, style_score=0.5, drift_score=0.1)
    assert any("size_score" in loc for loc in [e["loc"] for e in exc.value.errors()])


def test_fund_create_empty_scheme_code():
    """Empty scheme_code → ValidationError."""
    with pytest.raises(ValidationError) as exc:
        FundCreate(scheme_code="", scheme_name="Test Fund", amc_name="Test AMC",
                   category="Mid Cap Fund", benchmark_index="Nifty Midcap 150 TRI",
                   mandate_size_score=0.5, mandate_style_score=0.5)
    assert any("scheme_code" in str(e) for e in exc.value.errors())


def test_drift_alert_invalid_severity():
    """severity not in ('watch','amber','red') → ValidationError."""
    with pytest.raises(ValidationError) as exc:
        DriftAlertCreate(scheme_code="120503", alert_date=date(2024,3,15),
                         alert_type="drift_threshold", drift_score=0.30,
                         previous_drift_score=0.22, alert_message="Test.",
                         severity="critical", acknowledged=False)
    assert len(exc.value.errors()) > 0


def test_drift_summary_serialisation():
    """DriftSummary serialises to dict with all required keys."""
    summary = DriftSummary(
        scheme_code="120503", scheme_name="HDFC Mid-Cap Opportunities Fund",
        category="Mid Cap Fund", current_drift_score=0.312,
        mandate_size_score=0.5, mandate_style_score=0.5,
        drift_trend=[0.05,0.09,0.12,0.17,0.21,0.27,0.29,0.31],
        size_score_trend=[0.50,0.52,0.55,0.58,0.62,0.67,0.71,0.74],
        style_score_trend=[0.55,0.56,0.57,0.58,0.59,0.60,0.60,0.61],
        severity="amber", latest_alert_message="Fund has drifted from mandate.",
    )
    data = summary.model_dump()
    required = {"scheme_code","scheme_name","category","current_drift_score",
                "drift_trend","size_score_trend","style_score_trend",
                "severity","latest_alert_message"}
    assert required.issubset(data.keys())
    assert data["severity"] == "amber"
    assert len(data["drift_trend"]) == 8
