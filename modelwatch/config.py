import os
from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RuleOverride(BaseModel):
    enabled: bool = True
    severity: Literal["info", "warning", "critical"] | None = None
    min_count: int | None = Field(default=None, ge=1)
    persistence: int | None = Field(default=None, ge=1)
    recovery: int | None = Field(default=None, ge=1)
    cooldown_seconds: int | None = Field(default=None, ge=0)
    destination_env: str = Field(default="MODELWATCH_WEBHOOK_URL", pattern=r"^[A-Z][A-Z0-9_]{0,127}$")


class Config(BaseModel):
    model_name: str
    model_version: str
    schema_version: str
    preprocessing_version: str
    task: str
    feature_types: dict[str, str]
    numeric_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    categorical_values: dict[str, list[str]] = Field(default_factory=dict)
    allow_missing: list[str] = Field(default_factory=list)
    classes: list[int | str] = Field(default_factory=list)
    reference_version: str
    window_seconds: int = Field(default=3600, ge=60)
    min_events: int = Field(default=30, ge=2)
    min_labels: int = Field(default=20, ge=2)
    drift_method: str = "psi"
    drift_threshold: float = Field(default=0.2, gt=0)
    drift_share_threshold: float = Field(default=0.5, ge=0, le=1)
    latency_limit_ms: float = Field(default=200, gt=0)
    error_rate_limit: float = Field(default=0.05, ge=0, le=1)
    ingestion_failure_limit: int = Field(default=3, ge=0)
    max_missing_rates: dict[str, float] = Field(default_factory=dict)
    requests_per_minute: int = Field(default=1200, ge=1)
    max_body_bytes: int = Field(default=1048576, ge=1024)
    max_batch_events: int = Field(default=500, ge=1, le=10000)
    retention_days: int = Field(default=30, ge=1)
    alert_persistence: int = Field(default=2, ge=1)
    alert_rules: dict[str, RuleOverride] = Field(default_factory=dict)
    alert_recovery: int = Field(default=2, ge=1)
    alert_cooldown_seconds: int = Field(default=3600, ge=0)
    performance_accuracy_drop: float = Field(default=0.1, ge=0)
    performance_rmse_increase: float = Field(default=0.25, ge=0)
    webhook_timeout_seconds: int = Field(default=5, ge=1, le=30)

    @model_validator(mode="after")
    def validate_contract(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if not self.feature_types or any(t not in {"numeric", "categorical"} for t in self.feature_types.values()):
            raise ValueError("nonempty feature_types with numeric or categorical types required")
        if self.task == "classification" and not self.classes:
            raise ValueError("classification classes required")
        if self.drift_method not in {"psi", "wasserstein", "ks"}:
            raise ValueError("method must be a supported Evidently test")
        return self


@lru_cache
def load_config() -> Config:
    path = Path(os.getenv("MODELWATCH_CONFIG", "config.json"))
    return Config.model_validate_json(path.read_text())


def database_path() -> str:
    return os.getenv("MODELWATCH_DB", "state/modelwatch.db")


def report_dir() -> Path:
    return Path(os.getenv("MODELWATCH_REPORTS", "reports"))


def reference_dir() -> Path:
    return Path(os.getenv("MODELWATCH_REFERENCE_DIR", "data"))

