"""
MutualFundDrift — FastAPI router for fund CRUD and chart endpoints.
Handles fund registration, listing, snapshot management, and chart delivery.
"""

import logging
import io
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session
from sqlmodel import select

from backend.database import get_db
from backend.models import DriftAlert, Fund, PortfolioSnapshot
from backend.schemas import (
    DriftSummary,
    FundCreate,
    FundList,
    FundRead,
    PortfolioSnapshotRead,
)
from engine.drift_scorer import classify_drift_severity

logger = logging.getLogger(__name__)
router = APIRouter(tags=["funds"])


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build a DriftSummary from a Fund
# ─────────────────────────────────────────────────────────────────────────────

def _build_drift_summary(fund: Fund, db: Session) -> DriftSummary:
    """Build a DriftSummary response object from a Fund and its last 8 snapshots."""
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.scheme_code == fund.scheme_code)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
        .limit(8)
    )
    snapshots = db.exec(stmt).all()
    snapshots = list(reversed(snapshots))  # oldest first

    drift_trend = [s.drift_score for s in snapshots]
    size_trend = [s.size_score for s in snapshots]
    style_trend = [s.style_score for s in snapshots]
    current_drift = drift_trend[-1] if drift_trend else 0.0
    severity = classify_drift_severity(current_drift)

    # Latest unacknowledged alert message
    alert_stmt = (
        select(DriftAlert)
        .where(DriftAlert.scheme_code == fund.scheme_code)
        .order_by(DriftAlert.created_at.desc())
        .limit(1)
    )
    latest_alert = db.exec(alert_stmt).first()
    alert_message = latest_alert.alert_message if latest_alert else None

    return DriftSummary(
        scheme_code=fund.scheme_code,
        scheme_name=fund.scheme_name,
        category=fund.category,
        mandate_size_score=fund.mandate_size_score,
        mandate_style_score=fund.mandate_style_score,
        current_drift_score=current_drift,
        drift_trend=drift_trend,
        size_score_trend=size_trend,
        style_score_trend=style_trend,
        severity=severity,
        latest_alert_message=alert_message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[FundList])
def list_funds(
    category: Optional[str] = Query(default=None, description="Filter by SEBI category"),
    db: Session = Depends(get_db),
) -> List[FundList]:
    """Return all tracked funds, optionally filtered by SEBI category."""
    stmt = select(Fund).where(Fund.active == True)
    if category:
        stmt = stmt.where(Fund.category == category)
    funds = db.exec(stmt).all()
    return [FundList.model_validate(f) for f in funds]


@router.get("/{code}", response_model=DriftSummary)
def get_fund_detail(code: str, db: Session = Depends(get_db)) -> DriftSummary:
    """Return full fund detail including the last 8 snapshots and drift summary."""
    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")
    return _build_drift_summary(fund, db)


@router.post("/", response_model=FundRead, status_code=201)
def create_fund(payload: FundCreate, db: Session = Depends(get_db)) -> FundRead:
    """Register a new mutual fund scheme for drift tracking."""
    existing = db.get(Fund, payload.scheme_code)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Fund with scheme_code '{payload.scheme_code}' already exists.",
        )
    fund = Fund(**payload.model_dump())
    db.add(fund)
    db.commit()
    db.refresh(fund)
    logger.info("Registered new fund: %s (%s)", fund.scheme_name, fund.scheme_code)
    return FundRead.model_validate(fund)


@router.delete("/{code}", status_code=204)
def deactivate_fund(code: str, db: Session = Depends(get_db)):
    """Deactivate a fund so it no longer appears in tracking."""
    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")
    fund.active = False
    db.commit()
    return None


@router.get("/{code}/snapshots", response_model=List[PortfolioSnapshotRead])
def list_snapshots(code: str, db: Session = Depends(get_db)) -> List[PortfolioSnapshotRead]:
    """Return all PortfolioSnapshot records for a fund, newest first."""
    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")
    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.scheme_code == code)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
    )
    snapshots = db.exec(stmt).all()
    return [PortfolioSnapshotRead.model_validate(s) for s in snapshots]


@router.post("/{code}/ingest")
def ingest_fund_data(
    code: str,
    quarters: int = Query(default=1, ge=1, le=4, description="Number of past quarters to backfill"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Manually trigger data ingestion for a single fund.
    Downloads holdings from AMFI,
    computes style box + drift score, and saves PortfolioSnapshots.
    Supports backfilling up to 4 quarters of history.
    """
    import math
    from datetime import date
    from dateutil.relativedelta import relativedelta

    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")

    try:
        from engine.data_ingestion import (
            fetch_portfolio_holdings,
            fetch_stock_fundamentals,
            load_nse_classification,
        )
        from engine.classification import classify_portfolio, compute_hhi, get_sector_weights
        from engine.style_box import compute_style_box_coordinate
        from engine.drift_scorer import compute_drift_score

        nse_df = load_nse_classification()

        # Build list of target months: current month + past quarters (3 months apart)
        today = date.today().replace(day=1)
        target_dates = [today - relativedelta(months=3 * i) for i in range(quarters)]

        last_result = None
        fetched_count = 0

        for target_date in target_dates:
            month_str = target_date.strftime("%Y-%m")

            # Step 1: Fetch holdings
            holdings = fetch_portfolio_holdings(code, month_str)
            if holdings.empty:
                logger.warning("No holdings for %s on %s, skipping.", code, month_str)
                continue

            # Step 2: Classify holdings
            holdings = classify_portfolio(holdings, nse_df)

            # Step 3: Fetch fundamentals — top 30 holdings only (covers ~85%+ of NAV)
            top_holdings = holdings.nlargest(30, "pct_of_nav") if len(holdings) > 30 else holdings
            isin_list = top_holdings["isin"].dropna().unique().tolist()
            fundamentals = fetch_stock_fundamentals(isin_list)

            # Step 4: Compute style box coordinate
            size_score, style_score = compute_style_box_coordinate(holdings, fundamentals, nse_df)

            # Step 5: Cap composition
            large_pct = float(holdings[holdings["cap_category"] == "large_cap"]["pct_of_nav"].sum())
            mid_pct   = float(holdings[holdings["cap_category"] == "mid_cap"]["pct_of_nav"].sum())
            small_pct = float(holdings[holdings["cap_category"] == "small_cap"]["pct_of_nav"].sum())

            # Step 6: Sector HHI
            sector_weights = get_sector_weights(holdings)
            hhi = compute_hhi(sector_weights)

            # Step 7: Top 10 concentration
            top10_pct = float(
                holdings.nlargest(10, "pct_of_nav")["pct_of_nav"].sum()
                if len(holdings) >= 10
                else holdings["pct_of_nav"].sum()
            )

            # Step 8: Drift score
            drift_score = compute_drift_score(
                size_score, style_score,
                fund.mandate_size_score, fund.mandate_style_score,
            )

            # Step 9: Upsert snapshot
            existing_stmt = select(PortfolioSnapshot).where(
                PortfolioSnapshot.scheme_code == code,
                PortfolioSnapshot.snapshot_date == target_date,
            )
            existing = db.exec(existing_stmt).first()

            if existing:
                existing.size_score       = size_score
                existing.style_score      = style_score
                existing.large_cap_pct    = large_pct
                existing.mid_cap_pct      = mid_pct
                existing.small_cap_pct    = small_pct
                existing.top10_holdings_pct = top10_pct
                existing.hhi_sector       = hhi
                existing.drift_score      = drift_score
            else:
                snap = PortfolioSnapshot(
                    scheme_code=code,
                    snapshot_date=target_date,
                    size_score=size_score,
                    style_score=style_score,
                    large_cap_pct=large_pct,
                    mid_cap_pct=mid_pct,
                    small_cap_pct=small_pct,
                    top10_holdings_pct=top10_pct,
                    hhi_sector=hhi,
                    drift_score=drift_score,
                )
                db.add(snap)

            fetched_count += 1

            # Save results from the most recent (first) quarter for the response
            if last_result is None:
                last_result = {
                    "drift_score": drift_score,
                    "size_score": size_score,
                    "style_score": style_score,
                    "large_cap_pct": large_pct,
                    "mid_cap_pct": mid_pct,
                    "small_cap_pct": small_pct,
                }

        db.commit()

        if fetched_count == 0:
            return {
                "status": "warning",
                "message": f"No holdings data found for {fund.scheme_name}. "
                           f"AMFI may not have published these portfolios yet.",
            }

        severity = classify_drift_severity(last_result["drift_score"])
        return {
            "status": "success",
            "message": f"Data ingested for {fund.scheme_name} ({fetched_count} quarter(s) backfilled).",
            "quarters_fetched": fetched_count,
            "drift_score":   round(last_result["drift_score"], 3),
            "size_score":    round(last_result["size_score"], 3),
            "style_score":   round(last_result["style_score"], 3),
            "large_cap_pct": round(last_result["large_cap_pct"], 1),
            "mid_cap_pct":   round(last_result["mid_cap_pct"], 1),
            "small_cap_pct": round(last_result["small_cap_pct"], 1),
            "severity": severity,
        }

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", code, exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Ingestion failed: {str(exc)}",
        }


@router.post("/{code}/upload")
def upload_fund_file(
    code: str,
    file: UploadFile = File(...),
    snapshot_date: Optional[str] = Query(default=None, description="Snapshot date in YYYY-MM-01 format"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Ingest portfolio data for a fund via CSV or Excel file upload.
    Processes data locally and updates the database instantly without external network calls.
    """
    from datetime import date
    from engine.classification import classify_portfolio, compute_hhi, get_sector_weights
    from engine.data_ingestion import load_nse_classification
    from engine.drift_scorer import compute_drift_score, classify_drift_severity

    fund = db.get(Fund, code)
    if not fund:
        raise HTTPException(status_code=404, detail=f"Fund '{code}' not found.")

    filename = file.filename.lower()
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data.")

        # Standardize column names
        df.columns = [str(c).strip().lower().replace(" ", "_").replace("%", "pct") for c in df.columns]

        target_date = date.today().replace(day=1)
        if snapshot_date:
            try:
                target_date = date.fromisoformat(snapshot_date).replace(day=1)
            except ValueError:
                pass
        elif "snapshot_date" in df.columns and not pd.isna(df["snapshot_date"].iloc[0]):
            try:
                target_date = date.fromisoformat(str(df["snapshot_date"].iloc[0])[:10]).replace(day=1)
            except ValueError:
                pass

        # Case A: Direct Snapshot Metrics File
        if "drift_score" in df.columns or ("large_cap_pct" in df.columns and "mid_cap_pct" in df.columns):
            row = df.iloc[0]
            size_score = float(row.get("size_score", fund.mandate_size_score))
            style_score = float(row.get("style_score", fund.mandate_style_score))
            large_pct = float(row.get("large_cap_pct", 0.0))
            mid_pct = float(row.get("mid_cap_pct", 0.0))
            small_pct = float(row.get("small_cap_pct", 0.0))
            top10_pct = float(row.get("top10_holdings_pct", 30.0))
            hhi = float(row.get("hhi_sector", 0.15))
            drift_score = float(row.get("drift_score", compute_drift_score(size_score, style_score, fund.mandate_size_score, fund.mandate_style_score)))

        # Case B: Portfolio Holdings File
        else:
            nse_df = load_nse_classification()

            pct_col = next((c for c in ["pct_of_nav", "pct_nav", "weight", "weight_pct", "market_value_lakhs"] if c in df.columns), None)
            if not pct_col:
                raise HTTPException(status_code=400, detail="Missing weight/NAV column (e.g., 'pct_of_nav' or 'weight').")

            if "isin" not in df.columns and "company_name" in df.columns:
                df["isin"] = df["company_name"]

            if "isin" not in df.columns:
                raise HTTPException(status_code=400, detail="Missing stock identification column (e.g. 'isin' or 'company_name').")

            df["pct_of_nav"] = pd.to_numeric(df[pct_col], errors="coerce").fillna(0.0)

            # Classify
            df = classify_portfolio(df, nse_df)

            large_pct = float(df[df["cap_category"] == "large_cap"]["pct_of_nav"].sum())
            mid_pct   = float(df[df["cap_category"] == "mid_cap"]["pct_of_nav"].sum())
            small_pct = float(df[df["cap_category"] == "small_cap"]["pct_of_nav"].sum())
            total_cap = large_pct + mid_pct + small_pct
            if total_cap > 0:
                large_pct = (large_pct / total_cap) * 100
                mid_pct   = (mid_pct / total_cap) * 100
                small_pct = (small_pct / total_cap) * 100
            else:
                large_pct, mid_pct, small_pct = 70.0, 20.0, 10.0

            top10_pct = float(df.nlargest(10, "pct_of_nav")["pct_of_nav"].sum() if len(df) >= 10 else df["pct_of_nav"].sum())
            sector_weights = get_sector_weights(df)
            hhi = compute_hhi(sector_weights)

            size_score = (large_pct * 1.0 + mid_pct * 0.5 + small_pct * 0.0) / 100.0
            style_score = float(df.iloc[0].get("style_score", fund.mandate_style_score)) if "style_score" in df.columns else fund.mandate_style_score

            drift_score = compute_drift_score(size_score, style_score, fund.mandate_size_score, fund.mandate_style_score)

        # Upsert Snapshot
        existing_stmt = select(PortfolioSnapshot).where(
            PortfolioSnapshot.scheme_code == code,
            PortfolioSnapshot.snapshot_date == target_date,
        )
        existing = db.exec(existing_stmt).first()

        if existing:
            existing.size_score = size_score
            existing.style_score = style_score
            existing.large_cap_pct = large_pct
            existing.mid_cap_pct = mid_pct
            existing.small_cap_pct = small_pct
            existing.top10_holdings_pct = top10_pct
            existing.hhi_sector = hhi
            existing.drift_score = drift_score
        else:
            snap = PortfolioSnapshot(
                scheme_code=code,
                snapshot_date=target_date,
                size_score=size_score,
                style_score=style_score,
                large_cap_pct=large_pct,
                mid_cap_pct=mid_pct,
                small_cap_pct=small_pct,
                top10_holdings_pct=top10_pct,
                hhi_sector=hhi,
                drift_score=drift_score,
            )
            db.add(snap)

        db.commit()

        severity = classify_drift_severity(drift_score)
        return {
            "status": "success",
            "message": f"Successfully ingested file '{file.filename}' for {fund.scheme_name} on {target_date}.",
            "snapshot_date": str(target_date),
            "drift_score": round(drift_score, 4),
            "size_score": round(size_score, 4),
            "style_score": round(style_score, 4),
            "large_cap_pct": round(large_pct, 1),
            "mid_cap_pct": round(mid_pct, 1),
            "small_cap_pct": round(small_pct, 1),
            "severity": severity,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload ingestion failed for %s: %s", code, exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to process file: {str(exc)}",
        }
