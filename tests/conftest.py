"""All created monitoring events in tests are synthetic fixtures."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelwatch.config import load_config
from modelwatch.prepare import prepare
from modelwatch.store import init_db


@pytest.fixture
def demo(tmp_path, monkeypatch):
    config = json.loads(Path("config.json").read_text())
    config.update(min_events=5, min_labels=2, alert_persistence=2, alert_recovery=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("MODELWATCH_CONFIG", str(path))
    monkeypatch.setenv("MODELWATCH_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("MODELWATCH_REFERENCE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELWATCH_REPORTS", str(tmp_path / "reports"))
    monkeypatch.setenv("MODELWATCH_MODEL", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("MODELWATCH_API_TOKEN", "ingest-secret")
    monkeypatch.setenv("MODELWATCH_ADMIN_TOKEN", "admin-secret")
    load_config.cache_clear()
    cfg = load_config()
    init_db()
    prepare(cfg)
    yield cfg
    load_config.cache_clear()


@pytest.fixture
def events(demo):
    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=1)
    out = []
    for i in range(8):
        out.append({"event_id": f"evt-{i}", "timestamp": (start + timedelta(minutes=i+1)).isoformat(),
                    "model_name": demo.model_name, "model_version": demo.model_version,
                    "schema_version": demo.schema_version,
                    "features": {"sepal length (cm)": 5.1 + i * 0.01,
                                 "sepal width (cm)": 3.5,
                                 "petal length (cm)": 1.4,
                                 "petal width (cm)": 0.2},
                    "prediction": 0, "probabilities": [0.9, 0.05, 0.05],
                    "latency_ms": 14.0, "source": "SYNTHETIC_TEST_FIXTURE"})
    return out, start, end


@pytest.fixture
def client(demo):
    from modelwatch.api import app
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def headers():
    return {"Authorization": "Bearer ingest-secret"}


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer admin-secret"}
