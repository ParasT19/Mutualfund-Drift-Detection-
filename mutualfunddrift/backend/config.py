"""
MutualFundDrift — backend configuration module.
Loads all environment variables from .env using pydantic-settings.
This is the ONLY place settings are loaded; all other modules import from here.
"""

import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alert generator is fully rule-based — no external API key required

    # SQLAlchemy database connection URL (SQLite by default — no install needed)
    database_url: str = "sqlite:///./mfdrift.db"

    # Drift score threshold above which a DriftAlert is created
    drift_alert_threshold: float = 0.25

    # Rolling correlation value below which a correlation drop alert is raised
    correlation_drop_threshold: float = 0.80

    # Filesystem path to save/load the XGBoost model artifact
    model_save_path: str = "./models_saved/xgb_drift_model.joblib"

    # Python logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    log_level: str = "INFO"




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()


# Module-level singleton — import `settings` everywhere else in the project
settings: Settings = get_settings()

# Configure root logger based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
