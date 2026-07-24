"""
MutualFundDrift — FastAPI router for drift alert management.
Provides alert retrieval, creation, filtering by severity, and acknowledgement.
"""

import logging
from datetime import date
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlmodel import select

from backend.database import get_db
from backend.models import DriftAlert, Fund, PortfolioSnapshot
from backend.schemas import DriftAlertAcknowledge, DriftAlertCreate, DriftAlertRead
from engine.drift_scorer import classify_drift_severity, compute_drift_velocity

logger = logging.getLogger(__name__)
router = APIRouter(tags=["alerts"])


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[DriftAlertRead])
def list_all_alerts(db: Session = Depends(get_db)) -> List[DriftAlertRead]:
    """Return all unacknowledged alerts ordered by creation date descending (newest first)."""
    stmt = (
        select(DriftAlert)
        .where(DriftAlert.acknowledged == False)
        .order_by(DriftAlert.created_at.desc())
    )
    alerts = db.exec(stmt).all()
    return [DriftAlertRead.model_validate(a) for a in alerts]


@router.get("/severity/{level}", response_model=List[DriftAlertRead])
def list_alerts_by_severity(
    level: str, db: Session = Depends(get_db)
) -> List[DriftAlertRead]:
    """
    Return alerts filtered by severity level.
    Valid levels: 'watch', 'amber', 'red'.
    """
    valid_levels = {"watch", "amber", "red"}
    if level.lower() not in valid_levels:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity level '{level}'. Must be one of: {valid_levels}",
        )
    stmt = (
        select(DriftAlert)
        .where(DriftAlert.severity == level.lower())
        .order_by(DriftAlert.created_at.desc())
    )
    alerts = db.exec(stmt).all()
    return [DriftAlertRead.model_validate(a) for a in alerts]


@router.get("/{code}", response_model=List[DriftAlertRead])
def list_fund_alerts(code: str, db: Session = Depends(get_db)) -> List[DriftAlertRead]:
    """Return all alerts for a specific fund, newest first."""
    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")
    stmt = (
        select(DriftAlert)
        .where(DriftAlert.scheme_code == code)
        .order_by(DriftAlert.created_at.desc())
    )
    alerts = db.exec(stmt).all()
    return [DriftAlertRead.model_validate(a) for a in alerts]


@router.post("/{code}", response_model=DriftAlertRead, status_code=201)
def trigger_alert(code: str, db: Session = Depends(get_db)) -> DriftAlertRead:
    """
    Manually trigger alert generation for a fund.
    Always generates an alert regardless of drift severity.
    Saves and returns the resulting DriftAlert.
    """
    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")

    # Fetch latest snapshots for drift context
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.scheme_code == code)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
        .limit(8)
    )
    snapshots = list(reversed(db.exec(stmt).all()))

    if not snapshots:
        raise HTTPException(
            status_code=422,
            detail=f"No portfolio snapshots found for fund '{code}'.",
        )

    latest_snap = snapshots[-1]
    drift_score = latest_snap.drift_score
    prev_drift = snapshots[-2].drift_score if len(snapshots) >= 2 else drift_score
    severity = classify_drift_severity(drift_score)
    # Always save as at least 'watch' (DB requires non-normal)
    alert_severity = severity if severity != "normal" else "watch"

    # Compute drift velocity from available history
    drift_scores_list = [s.drift_score for s in snapshots]
    velocity = compute_drift_velocity(drift_scores_list)

    # Build snapshot dict for alert generator
    snap_dict = {
        "drift_score": drift_score,
        "size_score": latest_snap.size_score,
        "style_score": latest_snap.style_score,
        "large_cap_pct": latest_snap.large_cap_pct,
        "mid_cap_pct": latest_snap.mid_cap_pct,
        "small_cap_pct": latest_snap.small_cap_pct,
        "rolling_corr": latest_snap.rolling_corr,
        "hhi_sector": latest_snap.hhi_sector,
        "active_share": latest_snap.active_share,
    }
    fund_dict = {
        "scheme_name": fund.scheme_name,
        "category": fund.category,
        "amc_name": fund.amc_name,
        "mandate_size_score": fund.mandate_size_score,
        "mandate_style_score": fund.mandate_style_score,
    }

    try:
        from engine.alert_generator import generate_investor_alert
        alert_message = generate_investor_alert(fund_dict, snap_dict, velocity, {})
    except Exception as exc:
        logger.error("Alert generation failed: %s", exc)
        sev_label = severity.upper()
        alert_message = (
            f"{fund.scheme_name} has a current drift score of {drift_score:.3f} "
            f"(Severity: {sev_label}). Portfolio holds {latest_snap.large_cap_pct:.0f}% large cap / "
            f"{latest_snap.mid_cap_pct:.0f}% mid cap / {latest_snap.small_cap_pct:.0f}% small cap. "
            "Please review portfolio composition."
        )

    alert = DriftAlert(
        scheme_code=code,
        alert_date=date.today(),
        alert_type="drift_threshold",
        drift_score=drift_score,
        previous_drift_score=prev_drift,
        alert_message=alert_message[:1000],
        severity=alert_severity,
        acknowledged=False,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(
        "Alert created for %s: drift=%.3f, severity=%s", code, drift_score, alert.severity
    )
    return DriftAlertRead.model_validate(alert)


@router.put("/{alert_id}/ack", response_model=DriftAlertRead)
def acknowledge_alert(
    alert_id: int,
    payload: DriftAlertAcknowledge,
    db: Session = Depends(get_db),
) -> DriftAlertRead:
    """Acknowledge an alert by its integer ID, marking it as reviewed."""
    alert = db.get(DriftAlert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert with id={alert_id} not found.")
    alert.acknowledged = payload.acknowledged
    db.commit()
    db.refresh(alert)
    logger.info("Alert %d acknowledged=%s", alert_id, payload.acknowledged)
    return DriftAlertRead.model_validate(alert)
