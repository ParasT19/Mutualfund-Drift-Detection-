"""
MutualFundDrift — Local seed script.
Populates the SQLite database with demo funds and 12 months of
realistic portfolio snapshots to power the full dashboard.
Run once after starting the backend: python scripts/seed_local.py
"""

import os
import sys
import math
import json
import random
from datetime import date, datetime, timedelta

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env before importing backend modules
from dotenv import load_dotenv
load_dotenv()

from backend.database import create_db_and_tables, get_db_context
from backend.models import Fund, PortfolioSnapshot, DriftAlert

# ── Demo fund definitions ────────────────────────────────────────────────────
DEMO_FUNDS = [
    {
        "scheme_code": "120503",
        "scheme_name": "HDFC Mid-Cap Opportunities Fund",
        "amc_name": "HDFC AMC",
        "category": "Mid Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty Midcap 150 TRI",
        "mandate_size_score": 0.50,
        "mandate_style_score": 0.55,
        # Drift pattern: gradually moves into large cap territory
        "drift_pattern": "drifting_up",
    },
    {
        "scheme_code": "125354",
        "scheme_name": "Axis Bluechip Fund",
        "amc_name": "Axis AMC",
        "category": "Large Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty 50 TRI",
        "mandate_size_score": 0.85,
        "mandate_style_score": 0.70,
        # Stable: stays close to mandate
        "drift_pattern": "stable",
    },
    {
        "scheme_code": "119551",
        "scheme_name": "Nippon India Small Cap Fund",
        "amc_name": "Nippon India AMC",
        "category": "Small Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty Smallcap 250 TRI",
        "mandate_size_score": 0.15,
        "mandate_style_score": 0.60,
        # Stable: stays close to mandate to match real-world ET money data
        "drift_pattern": "stable",
    },
    {
        "scheme_code": "119598",
        "scheme_name": "SBI Magnum Midcap Fund",
        "amc_name": "SBI Funds Management",
        "category": "Mid Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty Midcap 150 TRI",
        "mandate_size_score": 0.50,
        "mandate_style_score": 0.50,
        "drift_pattern": "volatile",
    },
    {
        "scheme_code": "135781",
        "scheme_name": "Parag Parikh Flexi Cap Fund",
        "amc_name": "PPFAS AMC",
        "category": "Flexi Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty 500 TRI",
        "mandate_size_score": 0.60,
        "mandate_style_score": 0.55,
        "drift_pattern": "stable",
    },
    {
        "scheme_code": "118825",
        "scheme_name": "Mirae Asset Large Cap Fund",
        "amc_name": "Mirae Asset AMC",
        "category": "Large Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty 100 TRI",
        "mandate_size_score": 0.85,
        "mandate_style_score": 0.65,
        # Stable: closely follows large cap mandate
        "drift_pattern": "stable",
    },
    {
        "scheme_code": "128155",
        "scheme_name": "Motilal Oswal Midcap Fund",
        "amc_name": "Motilal Oswal AMC",
        "category": "Mid Cap Fund",
        "sub_category": "",
        "benchmark_index": "Nifty Midcap 150 TRI",
        "mandate_size_score": 0.50,
        "mandate_style_score": 0.60,
        # Drifting: gradually creeping into large cap territory
        "drift_pattern": "drifting_up",
    },
]


def _generate_snapshots(fund: dict, n_months: int = 36):
    """Generate realistic monthly snapshot data for a fund based on its drift pattern."""
    pattern = fund["drift_pattern"]
    ms = fund["mandate_size_score"]   # mandate size
    mst = fund["mandate_style_score"] # mandate style

    snapshots = []
    today = date.today().replace(day=1)

    for i in range(n_months, 0, -1):
        snap_date = (today - timedelta(days=30 * i)).replace(day=1)
        progress = i / n_months  # 1.0 = oldest, 0.0 = newest

        # Size score evolution
        if pattern == "drifting_up":
            size_score = ms + (0.35 * (1 - progress))  # drifts away from mandate
            style_score = mst + (0.10 * (1 - progress))
        elif pattern == "volatile":
            # Oscillates around mandate — sometimes drifts, sometimes recovers
            size_score = ms + 0.25 * math.sin(2 * math.pi * (1 - progress) * 2)
            style_score = mst + 0.10 * math.cos(2 * math.pi * (1 - progress) * 1.5)
        elif pattern == "stable":
            noise = (random.random() - 0.5) * 0.04
            size_score = ms + noise
            style_score = mst + (random.random() - 0.5) * 0.03
        elif pattern == "recovering":
            size_score = ms + (0.30 * progress)  # was drifted, now recovering
            style_score = mst + (0.08 * progress)
        else:
            size_score = ms
            style_score = mst

        size_score = max(0.0, min(1.0, size_score))
        style_score = max(0.0, min(1.0, style_score))

        # Drift score = Euclidean distance from mandate
        drift_score = math.sqrt(
            (size_score - ms) ** 2 + (style_score - mst) ** 2
        )

        # Cap composition based on size score
        if size_score > 0.67:
            large_pct = 75 + (size_score - 0.67) * 50
            mid_pct = 100 - large_pct - 8
            small_pct = 8
        elif size_score >= 0.33:
            large_pct = max(5, (size_score - 0.33) * 100)
            small_pct = max(5, (0.67 - size_score) * 50)
            mid_pct = 100 - large_pct - small_pct
        else:
            small_pct = 70 + (0.33 - size_score) * 100
            mid_pct = min(30, 100 - small_pct - 5)
            large_pct = max(0, 100 - small_pct - mid_pct)

        # Clamp
        total = large_pct + mid_pct + small_pct
        large_pct = round(large_pct / total * 100, 1)
        mid_pct = round(mid_pct / total * 100, 1)
        small_pct = round(100 - large_pct - mid_pct, 1)

        rolling_corr = max(0.6, min(0.98, 0.95 - drift_score * 0.8))
        hhi = round(0.10 + drift_score * 0.15, 4)
        top10 = round(40 + drift_score * 30, 1)
        active_share = round(max(0.3, 0.82 - drift_score * 0.3), 3)

        snapshots.append({
            "scheme_code": fund["scheme_code"],
            "snapshot_date": snap_date,
            "size_score": round(size_score, 4),
            "style_score": round(style_score, 4),
            "large_cap_pct": large_pct,
            "mid_cap_pct": mid_pct,
            "small_cap_pct": small_pct,
            "top10_holdings_pct": top10,
            "hhi_sector": hhi,
            "active_share": active_share,
            "drift_score": round(drift_score, 6),
            "rolling_corr": round(rolling_corr, 4),
        })

    return snapshots


def _severity(drift_score: float) -> str:
    if drift_score < 0.15: return "normal"
    elif drift_score < 0.25: return "watch"
    elif drift_score < 0.35: return "amber"
    return "red"


def seed():
    """Create all tables and insert demo funds + snapshots + alerts into SQLite."""
    print("[*] Initialising database tables...")
    create_db_and_tables()
    print("[OK] Tables created.\n")

    with get_db_context() as db:
        for fund_data in DEMO_FUNDS:
            pattern = fund_data.pop("drift_pattern")
            fund_data["active"] = True
            fund_data["created_at"] = datetime.utcnow()

            # Upsert fund
            existing = db.get(Fund, fund_data["scheme_code"])
            if existing:
                print(f"   [SKIP] Fund already exists: {fund_data['scheme_name']}")
                fund_data["drift_pattern"] = pattern
                continue

            fund = Fund(**fund_data)
            db.add(fund)
            print(f"   [+] Added fund: {fund_data['scheme_name']}")

            # Generate and insert snapshots
            fund_data["drift_pattern"] = pattern
            snaps = _generate_snapshots(fund_data, n_months=12)
            for s in snaps:
                snap = PortfolioSnapshot(**s)
                db.add(snap)
            print(f"      [DATA] Inserted {len(snaps)} monthly snapshots")

            # Add alert if latest drift is above threshold
            latest = snaps[-1]
            latest_drift = latest["drift_score"]
            sev = _severity(latest_drift)
            if sev in ("watch", "amber", "red"):
                alert_msg = (
                    f"{fund_data['scheme_name']} has drifted {latest_drift:.3f} points "
                    f"from its {fund_data['category']} mandate (Severity: {sev.upper()}). "
                    f"Portfolio now holds {latest['large_cap_pct']:.0f}% large cap / "
                    f"{latest['mid_cap_pct']:.0f}% mid cap / {latest['small_cap_pct']:.0f}% small cap. "
                    f"Rolling benchmark correlation at {latest['rolling_corr']:.2f}. "
                    f"Investors should review allocation."
                ) if sev != "normal" else ""

                if alert_msg:
                    prev_drift = snaps[-2]["drift_score"] if len(snaps) >= 2 else latest_drift
                    alert = DriftAlert(
                        scheme_code=fund_data["scheme_code"],
                        alert_date=date.today(),
                        alert_type="drift_threshold",
                        drift_score=latest_drift,
                        previous_drift_score=prev_drift,
                        alert_message=alert_msg[:1000],
                        severity=sev,
                        acknowledged=False,
                        created_at=datetime.utcnow(),
                    )
                    db.add(alert)
                    print(f"      [ALERT] Created {sev.upper()} alert")

    print("\n[DONE] Seed complete! Database is ready.")
    print("   Now run: python -m uvicorn backend.main:app --reload")


if __name__ == "__main__":
    random.seed(42)
    seed()
