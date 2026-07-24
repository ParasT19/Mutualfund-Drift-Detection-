"""
MutualFundDrift — SQLModel ORM table definitions.
Defines the four core tables: Fund, PortfolioSnapshot, DriftAlert, DriftPrediction.
"""

import json
from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# Fund — registered mutual fund scheme metadata
# ---------------------------------------------------------------------------

class Fund(SQLModel, table=True):
    """
    Represents a mutual fund scheme tracked by the drift detection system.
    The scheme_code is the primary key and corresponds to the AMFI scheme code.
    """

    __tablename__ = "fund"

    scheme_code: str = Field(
        primary_key=True,
        description="AMFI scheme code, e.g. '120503' for HDFC Mid-Cap Opportunities Fund",
    )
    scheme_name: str = Field(description="Full name of the mutual fund scheme")
    amc_name: str = Field(description="Asset Management Company name")
    category: str = Field(description="SEBI category, e.g. 'Mid Cap Fund'")
    sub_category: str = Field(default="", description="SEBI sub-category if applicable")
    benchmark_index: str = Field(
        description="Benchmark index name, e.g. 'Nifty Midcap 150 TRI'"
    )

    # Mandate coordinates derived from fund_mandates.csv and SEBI category rules
    mandate_size_score: float = Field(
        description="Target size score from SEBI mandate: 0.0=small, 0.5=mid, 1.0=large"
    )
    mandate_style_score: float = Field(
        description="Target style score from SEBI mandate: 0.0=value, 0.5=blend, 1.0=growth"
    )

    active: bool = Field(default=True, description="Whether this fund is actively tracked")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# PortfolioSnapshot — one computed snapshot per fund per month
# ---------------------------------------------------------------------------

class PortfolioSnapshot(SQLModel, table=True):
    """
    Stores the computed style box coordinates and drift metrics for a fund
    at a specific month. Dates are always normalised to the first of the month.
    """

    __tablename__ = "portfolio_snapshot"

    id: Optional[int] = Field(default=None, primary_key=True)
    scheme_code: str = Field(foreign_key="fund.scheme_code", index=True)

    # Snapshot date, always stored as first-of-month (e.g. 2024-03-01)
    snapshot_date: date = Field(index=True)

    # Style box computed coordinates
    size_score: float = Field(
        description="Weighted size score 0.0 (small) to 1.0 (large)"
    )
    style_score: float = Field(
        description="Weighted style score 0.0 (value) to 1.0 (growth)"
    )

    # Market cap composition as percentage of NAV
    large_cap_pct: float = Field(default=0.0)
    mid_cap_pct: float = Field(default=0.0)
    small_cap_pct: float = Field(default=0.0)

    # Risk and concentration metrics
    top10_holdings_pct: float = Field(
        default=0.0,
        description="Percentage of NAV held in top 10 stocks",
    )
    hhi_sector: float = Field(
        default=0.0,
        description="Herfindahl-Hirschman Index across GICS sectors (0=diverse, 1=concentrated)",
    )
    active_share: Optional[float] = Field(
        default=None,
        description="Active share vs benchmark index (0.0 to 1.0)",
    )
    turnover_ratio: Optional[float] = Field(
        default=None,
        description="Portfolio turnover ratio from AMFI footnotes",
    )

    # Core drift metrics
    drift_score: float = Field(
        description="Euclidean distance from mandate coordinate (sqrt((s-ms)^2 + (st-mst)^2))"
    )
    rolling_corr: Optional[float] = Field(
        default=None,
        description="12-month rolling Pearson correlation vs category benchmark",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# DriftAlert — one alert per fund per trigger event
# ---------------------------------------------------------------------------

class DriftAlert(SQLModel, table=True):
    """
    Records an alert raised when a fund's drift score, correlation, or
    predicted drift breaches a defined threshold.
    """

    __tablename__ = "drift_alert"

    id: Optional[int] = Field(default=None, primary_key=True)
    scheme_code: str = Field(foreign_key="fund.scheme_code", index=True)
    alert_date: date = Field(index=True)

    # What triggered the alert
    alert_type: str = Field(
        description="One of: 'drift_threshold', 'correlation_drop', 'predicted_drift'"
    )

    drift_score: float = Field(description="Drift score at time of alert")
    previous_drift_score: float = Field(
        description="Drift score from the preceding snapshot for trend context"
    )

    # LLM-generated plain English investor alert message (max 1000 characters)
    alert_message: str = Field(
        max_length=1000,
        description="Plain-English LLM-generated investor warning",
    )

    severity: str = Field(
        description="Alert severity: 'watch', 'amber', or 'red'"
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether a human has reviewed and acknowledged this alert",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# DriftPrediction — XGBoost forward-looking prediction per fund
# ---------------------------------------------------------------------------

class DriftPrediction(SQLModel, table=True):
    """
    Stores the XGBoost model's prediction of whether a fund will exhibit
    significant style drift in the next 4 weeks (one quarter).
    """

    __tablename__ = "drift_prediction"

    id: Optional[int] = Field(default=None, primary_key=True)
    scheme_code: str = Field(foreign_key="fund.scheme_code", index=True)
    prediction_date: date = Field(index=True)

    predicted_drift_score: float = Field(
        description="Model's point estimate of the next-quarter drift score"
    )
    current_drift_score: float = Field(
        description="Actual drift score at time of prediction"
    )
    drift_probability: float = Field(
        description="P(drift_score > threshold) in next 4 weeks, from predict_proba"
    )

    # JSON string of top 5 SHAP feature names and their float values
    top_shap_features: str = Field(
        default="{}",
        description="JSON-serialised dict of top 5 SHAP feature contributions",
    )

    model_version: str = Field(
        default="1.0.0",
        description="Version tag of the XGBoost model that generated this prediction",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_shap_features(self) -> dict:
        """Deserialise the top_shap_features JSON string to a Python dict."""
        try:
            return json.loads(self.top_shap_features)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_shap_features(self, features: dict) -> None:
        """Serialise a Python dict into the top_shap_features JSON string field."""
        self.top_shap_features = json.dumps(features, default=float)
