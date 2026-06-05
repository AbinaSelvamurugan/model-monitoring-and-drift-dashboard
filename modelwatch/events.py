"""Event contract: strict feature validation, idempotent immutable prediction records."""
import math
from datetime import datetime, timezone

from modelwatch.config import Config
from modelwatch.schemas import GroundTruth, PredictionEvent
from modelwatch.store import connect, current_reference, dumps, utcnow


class ContractError(ValueError):
    pass


def valid_label(label, cfg):
    if label is None:
        return
    if cfg.task == "classification":
        if label not in cfg.classes or isinstance(label, bool):
            raise ContractError("label/prediction not in declared classes")
    elif isinstance(label, bool) or not isinstance(label, (float, int)) or not math.isfinite(label):
        raise ContractError("regression label/prediction must be finite numeric")


def validate_event(ev: PredictionEvent, cfg: Config):
    if (ev.model_name != cfg.model_name or ev.model_version != cfg.model_version
            or ev.schema_version != cfg.schema_version):
        raise ContractError("model or schema version incompatible with approved reference")
    if set(ev.features) != set(cfg.feature_types):
        raise ContractError("features do not match approved schema")
    for name, kind in cfg.feature_types.items():
        v = ev.features[name]
        if v is None:
            if name not in cfg.allow_missing:
                raise ContractError(f"{name}: missing value forbidden")
            continue
        if kind == "numeric":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ContractError(f"{name}: finite number required")
            bounds = cfg.numeric_ranges.get(name)
            if bounds and not bounds[0] <= v <= bounds[1]:
                raise ContractError(f"{name}: outside approved range")
        elif not isinstance(v, str) or (name in cfg.categorical_values
                                    and v not in cfg.categorical_values[name]):
            raise ContractError(f"{name}: invalid category")
    if ev.status == "ok":
        if ev.prediction is None:
            raise ContractError("successful event requires prediction")
        valid_label(ev.prediction, cfg)
    elif ev.prediction is not None:
        raise ContractError("failed event must not claim a prediction")
    if ev.probabilities is not None:
        if cfg.task != "classification" or len(ev.probabilities) != len(cfg.classes):
            raise ContractError("probabilities incompatible with class order")
    valid_label(ev.ground_truth, cfg)
    if ev.timestamp < datetime.now(timezone.utc).replace(year=2000):
        raise ContractError("timestamp outside supported retention horizon")


def add_events(events: list[PredictionEvent], cfg: Config):
    if len(events) > cfg.max_batch_events:
        raise ContractError("batch too large")
    ref = current_reference(cfg)
    # Validate complete batch first: atomic failure on any malformed member.
    for event in events:
        validate_event(event, cfg)
    inserted, duplicates, late = 0, 0, 0
    with connect(write=True) as db:
        for event in events:
            payload = event.model_dump(mode="json")
            existing = db.execute("SELECT payload FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if existing:
                if existing["payload"] != dumps(payload):
                    raise ContractError("event_id exists with different content")
                duplicates += 1
                continue
            closed = db.execute("""SELECT 1 FROM windows WHERE model_version=?
                AND reference_version=? AND start<=? AND end>? AND status IN ('running','completed','failed')
                LIMIT 1""", (event.model_version, ref["version"], event.timestamp.isoformat(),
                              event.timestamp.isoformat())).fetchone()
            is_late = int(closed is not None)
            db.execute("""INSERT INTO events
                (event_id,request_id,ts,model_name,model_version,schema_version,reference_version,
                 payload,label_json,labeled_at,created_at,late_after_evaluation)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    event.event_id, event.request_id, event.timestamp.isoformat(), event.model_name,
                    event.model_version, event.schema_version, ref["version"], dumps(payload),
                    dumps(event.ground_truth) if event.ground_truth is not None else None,
                    utcnow() if event.ground_truth is not None else None, utcnow(), is_late))
            inserted += 1
            late += is_late
    return {"inserted": inserted, "duplicates": duplicates, "late_after_evaluation": late}


def add_label(label: GroundTruth, cfg: Config):
    valid_label(label.label, cfg)
    with connect(write=True) as db:
        record = db.execute("SELECT label_json FROM events WHERE event_id=?", (label.event_id,)).fetchone()
        if record is None:
            raise KeyError("unknown event_id")
        desired = dumps(label.label)
        if record["label_json"] is not None:
            if record["label_json"] != desired:
                raise ContractError("conflicting ground truth; requires separate correction procedure")
            return {"updated": False, "duplicate": True}
        db.execute("UPDATE events SET label_json=?,labeled_at=? WHERE event_id=?",
                   (desired, label.labeled_at.isoformat(), label.event_id))
    return {"updated": True, "duplicate": False}
