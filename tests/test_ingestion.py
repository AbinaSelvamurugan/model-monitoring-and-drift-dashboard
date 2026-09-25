from datetime import datetime, timezone

import pytest

from modelwatch.store import many, one


def test_create_and_idempotent(client, events, headers):
    e = events[0][0]
    assert client.post("/v1/events", json=e, headers=headers).status_code == 201
    assert client.post("/v1/events", json=e, headers=headers).json()["duplicates"] == 1
    assert len(many("SELECT * FROM events")) == 1


def test_conflicting_duplicates(client, events, headers):
    e = events[0][0]
    client.post("/v1/events", json=e, headers=headers)
    conflict = dict(e, latency_ms=16)
    assert client.post("/v1/events", json=conflict, headers=headers).status_code == 422
    assert len(many("SELECT * FROM events")) == 1


@pytest.mark.parametrize("field,value", [
    ("features", {}), ("schema_version", "other"), ("prediction", 99),
    ("timestamp", "not-a-timestamp"), ("timestamp", "2026-01-01T00:00:00"),
    ("probabilities", [0.4, 0.4, 0.4]),
])
def test_validation(client, events, headers, field, value):
    e = dict(events[0][0], **{field: value})
    assert client.post("/v1/events", json=e, headers=headers).status_code == 422


def test_invalid_feature_values(client, events, headers):
    e = dict(events[0][0])
    e["features"] = dict(e["features"], **{"sepal width (cm)": "N/A"})
    assert client.post("/v1/events", json=e, headers=headers).status_code == 422


def test_atomic_batch(client, events, headers):
    a, b = events[0][:2]
    b = dict(b, model_version="wrong")
    response = client.post("/v1/events/batch", json={"events": [a, b]}, headers=headers)
    assert response.status_code == 422
    assert not many("SELECT * FROM events")


def test_out_of_order(client, events, headers):
    a, b = events[0][:2]
    assert client.post("/v1/events", json=b, headers=headers).status_code == 201
    assert client.post("/v1/events", json=a, headers=headers).status_code == 201
    assert len(many("SELECT * FROM events")) == 2


def test_label_delayed_duplicate_and_conflict(client, events, headers):
    e = events[0][0]
    client.post("/v1/events", json=e, headers=headers)
    payload = {"event_id": e["event_id"], "label": 0,
               "labeled_at": datetime.now(timezone.utc).isoformat()}
    assert client.post("/v1/labels", json=payload, headers=headers).json()["updated"]
    assert client.post("/v1/labels", json=payload, headers=headers).json()["duplicate"]
    assert client.post("/v1/labels", json=dict(payload, label=1), headers=headers).status_code == 422
    assert one("SELECT label_json FROM events LIMIT 1")["label_json"] == "0"


def test_auth_and_limits(client, events, headers):
    assert client.post("/v1/events", json=events[0][0]).status_code == 401
    assert client.post("/v1/events", headers=headers, data="X" * 1050000).status_code == 413
    assert client.get("/health/live").json()["status"] == "alive"
