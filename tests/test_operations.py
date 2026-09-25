"""Synthetic operational fixture tests."""
import json
from datetime import timedelta
from pathlib import Path

import pytest

from modelwatch.config import reference_dir
from modelwatch.evaluate import evaluate_availability, evaluate_window
from modelwatch.events import add_events
from modelwatch.schemas import PredictionEvent
from modelwatch.store import current_reference, many, one, purge_expired
from modelwatch.windows import create_window


def test_local_model_predicts_and_persists(client, headers):
    response = client.post("/v1/predict", headers=headers, json={
        "sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4, "petal width (cm)": 0.2})
    assert response.status_code == 201
    body = response.json()
    assert body["prediction"] in [0, 1, 2]
    assert abs(sum(body["probabilities"]) - 1) < 1e-4
    assert one("SELECT * FROM events WHERE event_id=?", (body["event_id"],))


def test_late_event_not_counted_again(demo, events, monkeypatch):
    import modelwatch.evaluate as engine
    def intentional_failure(*args, **kwargs):
        raise RuntimeError("synthetic evaluator failure fixture")
    monkeypatch.setattr(engine, "evidently_report", intentional_failure)
    batch, start, end = events
    add_events([PredictionEvent.model_validate(e) for e in batch], demo)
    win = create_window(demo, start, end)
    assert evaluate_window(demo, win["id"])["status"] == "failed"  # injected failure
    later = dict(batch[0], event_id="after-window-evaluated")
    result = add_events([PredictionEvent.model_validate(later)], demo)
    assert result["late_after_evaluation"] == 1
    assert one("SELECT late_after_evaluation FROM events WHERE event_id=?",
               (later["event_id"],))["late_after_evaluation"] == 1
    assert one("SELECT event_count FROM windows WHERE id=?", (win["id"],))["event_count"] == len(batch)


def test_missing_monitoring_window_alert(demo, events):
    batch, start, end = events
    early_start = start - timedelta(hours=3)
    earlier = dict(batch[0], timestamp=(early_start + timedelta(minutes=10)).isoformat())
    add_events([PredictionEvent.model_validate(earlier)], demo)
    zero_one = create_window(demo, start - timedelta(hours=2), start - timedelta(hours=1))
    zero_two = create_window(demo, start - timedelta(hours=1), start)
    assert evaluate_availability(demo, zero_one["id"])["summary"]["event_count"] == 0
    assert evaluate_availability(demo, zero_two["id"])["status"] == "completed"
    incident = one("SELECT * FROM incidents WHERE rule_id='missing_monitoring_window'")
    assert incident and incident["state"] == "new"
    assert len(many("SELECT * FROM evaluations WHERE kind='availability'")) == 2


def test_immutable_reference_checksum(demo):
    record = current_reference(demo)
    p = Path(record["path"])
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        current_reference(demo)


def test_approve_new_reference_from_admin(client, demo, admin_headers):
    original = Path(current_reference(demo)["path"])
    staged = reference_dir() / "staged-iris.csv"
    staged.write_bytes(original.read_bytes())
    body = {"version": "iris-holdout-2", "source_file": staged.name,
            "reason": "Review of model holdout records"}
    result = client.post("/v1/references/approve", json=body, headers=admin_headers)
    assert result.status_code == 201, result.text
    assert result.json()["version"] == "iris-holdout-2"
    assert demo.reference_version == "iris-holdout-1"
    assert client.post("/v1/references/approve", json=body, headers=admin_headers).status_code == 409
    assert client.post("/v1/references/approve", json=body).status_code == 401


def test_retention_preserves_recent_events(demo, events):
    first = PredictionEvent.model_validate(events[0][0])
    add_events([first], demo)
    deleted = purge_expired(30)
    assert deleted["deleted_events"] == 0
    assert one("SELECT event_id FROM events WHERE event_id=?", (first.event_id,))


def test_per_process_rate_limit(client, demo, headers, monkeypatch):
    from modelwatch.api import _rate_buckets
    from modelwatch.config import load_config
    path = Path(__import__("os").environ["MODELWATCH_CONFIG"])
    doc = json.loads(path.read_text())
    doc["requests_per_minute"] = 2
    path.write_text(json.dumps(doc))
    load_config.cache_clear()
    _rate_buckets.clear()
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.get("/v1/models", headers=headers).status_code == 429


def test_reference_belongs_to_same_model_version(demo):
    from modelwatch.store import current_reference
    with pytest.raises(ValueError, match="differs"):
        current_reference(demo.model_copy(update={"preprocessing_version": "different-prep"}))


def test_interrupted_claim_marked_failed(demo, events):
    from modelwatch.store import connect, flag_interrupted, utcnow
    _, start, end = events
    win = create_window(demo, start, end)
    with connect(write=True) as db:
        db.execute("""INSERT INTO evaluations(window_id,kind,label_count,status,created_at)
                      VALUES (?, 'drift', 0, 'running', ?)""", (win["id"], utcnow()))
    assert flag_interrupted() == 1
    assert one("SELECT * FROM evaluations")["status"] == "failed"
    assert one("SELECT * FROM windows WHERE id=?", (win["id"],))["status"] == "failed"
    assert flag_interrupted() == 0


def test_concurrent_duplicate_ingestion_serialized(demo, events):
    from concurrent.futures import ThreadPoolExecutor
    ev = PredictionEvent.model_validate(events[0][0])
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: add_events([ev], demo), range(12)))
    assert sum(x["inserted"] for x in results) == 1
    assert sum(x["duplicates"] for x in results) == 11
    assert len(many("SELECT event_id FROM events")) == 1
