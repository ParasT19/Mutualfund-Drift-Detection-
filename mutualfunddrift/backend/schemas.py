"""
MutualFundDrift — Pydantic v2 request/response schemas.
Schemas are separate from ORM models to control what data enters and leaves the API.
"""

from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Fund Schemas
# ─────────────────────────────────────────────────────────────────────────────

class FundCreate(BaseModel):
    """Schema for registering a new fund to track."""

    scheme_code: str
    scheme_name: str
    amc_name: str
    category: str
    sub_category: str = ""
    benchmark_index: str
    mandate_size_score: float = Field(ge=0.0, le=1.0)
    mandate_style_score: float = Field(ge=0.0, le=1.0)
    active: bool = True

    @field_validator("scheme_code")
    @classmethod
    def scheme_code_must_not_be_empty(cls, v: str) -> str:
        """Ensure scheme_code is a non-empty string."""
        if not v or not v.strip():
            raise ValueError("scheme_code must be a non-empty string")
        return v.strip()


class FundRead(FundCreate):
    """Schema returned when reading a single fund record."""

    created_at: datetime

    model_config = {"from_attributes": True}


class FundList(BaseModel):
    """Lightweight fund summary for list endpoints."""

    scheme_code: str
    scheme_name: str
    amc_name: str
    category: str
    benchmark_index: str
    active: bool

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioSnapshot Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioSnapshotCreate(BaseModel):
    """Schema for creating or upserting a monthly portfolio snapshot."""

    scheme_code: str
    snapshot_date: date
    size_score: float = Field(ge=0.0, le=1.0)
    style_score: float = Field(ge=0.0, le=1.0)
    large_cap_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    mid_cap_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    small_cap_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    top10_holdings_pct: float = Field(ge=0.0, le=100.0, default=0.0)
    hhi_sector: float = Field(ge=0.0, le=1.0, default=0.0)
    active_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    turnover_ratio: Optional[float] = Field(default=None, ge=0.0)
    drift_score: float = Field(ge=0.0, le=2.0)
    rolling_corr: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("scheme_code")
    @classmethod
    def scheme_code_not_empty(cls, v: str) -> str:
        """Ensure scheme_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("scheme_code must be a non-empty string")
        return v.strip()


class PortfolioSnapshotRead(PortfolioSnapshotCreate):
    """Schema returned when reading a portfolio snapshot record."""

    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# DriftAlert Schemas
# ─────────────────────────────────────────────────────────────────────────────

VALID_ALERT_TYPES = Literal["drift_threshold", "correlation_drop", "predicted_drift"]
VALID_SEVERITIES = Literal["watch", "amber", "red"]


class DriftAlertCreate(BaseModel):
    """Schema for creating a new drift alert."""

    scheme_code: str
    alert_date: date
    alert_type: VALID_ALERT_TYPES
    drift_score: float = Field(ge=0.0, le=2.0)
    previous_drift_score: float = Field(ge=0.0, le=2.0)
    alert_message: str = Field(max_length=1000)
    severity: VALID_SEVERITIES
    acknowledged: bool = False

    @field_validator("scheme_code")
    @classmethod
    def scheme_code_not_empty(cls, v: str) -> str:
        """Ensure scheme_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("scheme_code must be a non-empty string")
        return v.strip()


class DriftAlertRead(DriftAlertCreate):
    """Schema returned when reading a drift alert record."""

    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DriftAlertAcknowledge(BaseModel):
    """Schema for acknowledging an alert via PUT endpoint."""

    acknowledged: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# DriftPrediction Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DriftPredictionCreate(BaseModel):
    """Schema for storing a new XGBoost model prediction."""

    scheme_code: str
    prediction_date: date
    predicted_drift_score: float = Field(ge=0.0, le=2.0)
    current_drift_score: float = Field(ge=0.0, le=2.0)
    drift_probability: float = Field(ge=0.0, le=1.0)
    top_shap_features: str = "{}"
    model_version: str = "1.0.0"

    @field_validator("scheme_code")
    @classmethod
    def scheme_code_not_empty(cls, v: str) -> str:
        """Ensure scheme_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("scheme_code must be a non-empty string")
        return v.strip()


class DriftPredictionRead(DriftPredictionCreate):
    """Schema returned when reading a prediction record."""

    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Composite / Aggregated Schemas
# ─────────────────────────────────────────────────────────────────────────────

class DriftSummary(BaseModel):
    """
    Aggregated drift summary for a fund, including trend arrays and severity.
    Used as the primary response from the fund detail endpoint.
    """

    scheme_code: str
    scheme_name: str
    category: str
    mandate_size_score: float = Field(ge=0.0, le=1.0)
    mandate_style_score: float = Field(ge=0.0, le=1.0)
    current_drift_score: float
    drift_trend: List[float] = Field(
        description="List of drift scores for the last 8 snapshots (oldest first)"
    )
    size_score_trend: List[float] = Field(
        description="List of size scores for the last 8 snapshots"
    )
    style_score_trend: List[float] = Field(
        description="List of style scores for the last 8 snapshots"
    )
    severity: str = Field(
        description="Current drift severity: 'normal', 'watch', 'amber', or 'red'"
    )
    latest_alert_message: Optional[str] = Field(
        default=None,
        description="Most recent LLM-generated alert message, if any",
    )

    model_config = {"from_attributes": True}

