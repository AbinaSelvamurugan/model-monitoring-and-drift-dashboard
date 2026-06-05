import math
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, max_length=128)
    timestamp: datetime
    model_name: str
    model_version: str
    schema_version: str
    features: dict[str, Any]
    prediction: float | int | str | None = None
    probabilities: list[float] | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    status: Literal["ok", "error"] = "ok"
    source: str | None = Field(default=None, max_length=60)
    segment: str | None = Field(default=None, max_length=60)
    ground_truth: float | int | str | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("timezone-aware timestamp required")
        if v > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("timestamp too far in future")
        return v.astimezone(timezone.utc)

    @field_validator("probabilities")
    @classmethod
    def probabilities_valid(cls, v):
        if v is not None and (not v or any(not math.isfinite(x) or x < 0 or x > 1 for x in v)
                              or not math.isclose(sum(v), 1, abs_tol=1e-4)):
            raise ValueError("probabilities must be finite, within [0,1], and sum to 1")
        return v


class EventBatch(BaseModel):
    events: list[PredictionEvent] = Field(min_length=1)


class GroundTruth(BaseModel):
    event_id: str
    label: float | int | str
    labeled_at: datetime

    @field_validator("labeled_at")
    @classmethod
    def timestamp_aware(cls, v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("labeled_at requires timezone")
        return v.astimezone(timezone.utc)


class WindowRequest(BaseModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def aware(cls, v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("timezone required")
        return v.astimezone(timezone.utc)


class IncidentChange(BaseModel):
    note: str = Field(min_length=3, max_length=2000)


class ReferenceApproval(BaseModel):
    version: str = Field(pattern=r"^[a-zA-Z0-9._-]{1,64}$")
    source_file: str = Field(pattern=r"^[a-zA-Z0-9._-]+\.csv$")
    reason: str = Field(min_length=3, max_length=2000)
