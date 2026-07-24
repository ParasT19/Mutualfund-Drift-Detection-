"""
Train the XGBoost drift prediction model on all current database snapshots.
Run this once: python engine/train_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from backend.database import get_db_context, create_db_and_tables
from backend.models import PortfolioSnapshot
from sqlmodel import select
from engine.feature_engineer import prepare_training_data
from engine.predictor import train_model

print("[*] Loading snapshots from database...")
create_db_and_tables()

with get_db_context() as db:
    snaps = db.exec(select(PortfolioSnapshot)).all()
    df = pd.DataFrame([s.__dict__ for s in snaps])
    df = df.drop(columns=["_sa_instance_state"], errors="ignore")

print(f"[OK] Loaded {len(df)} snapshots across {df['scheme_code'].nunique()} funds.\n")

print("[*] Engineering features...")
X, y = prepare_training_data(df)
print(f"[OK] Training samples: {len(X)} | Positive (will drift): {y.sum()} | Negative: {(y==0).sum()}\n")

if len(X) < 5:
    print("[!] Not enough samples to train. Need at least 12 snapshots per fund.")
    sys.exit(1)

print("[*] Training XGBoost classifier with TimeSeriesSplit CV...")
model, mean_auc = train_model(X, y)
print(f"\n[OK] Training complete!")
print(f"     Mean AUC across folds: {mean_auc:.4f}")
print(f"     Model features: {X.shape[1]}")
print(f"     Model saved successfully.\n")
print("[DONE] You can now use the ML prediction features in the dashboard.")
