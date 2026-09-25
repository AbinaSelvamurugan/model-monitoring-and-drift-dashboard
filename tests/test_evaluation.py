import sys
from types import SimpleNamespace

from modelwatch.evaluate import (compute_performance, evaluate_performance, evaluate_window,
                                 extract_drift)
from modelwatch.schemas import PredictionEvent
from modelwatch.events import add_events, add_label
from modelwatch.schemas import GroundTruth
from modelwatch.store import many, one
from modelwatch.windows import create_window


def make_window(demo, events):
    batch, start, end = events
    add_events([PredictionEvent.model_validate(e) for e in batch], demo)
    return create_window(demo, start, end)


def test_insufficient_events(demo, events):
    e, start, end = events
    add_events([PredictionEvent.model_validate(e[0])], demo)
    win = create_window(demo, start, end)
    assert win["status"] == "insufficient"
    assert evaluate_window(demo, win["id"])["status"] == "insufficient"


def test_drift_parser_from_synthetic_fixture():
    # Synthetic mock representation of documented Evidently 0.7 snapshot.dict() format.
    fixture = {"metrics": [
        {"metric_name": "ValueDrift(column=a,method=psi)",
         "config": {"column": "a"}, "value": 0.24},
        {"metric_name": "ValueDrift(column=b,method=psi)",
         "config": {"column": "b"}, "value": 0.02}]}
    got = extract_drift(fixture, ["a", "b"], 0.2)
    assert got["a"]["drifted"] is True
    assert got["b"]["drifted"] is False


def test_drift_parser_fails_closed():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        extract_drift({"metrics": []}, ["a"], 0.2)


def test_failed_evidently_persisted(demo, events, monkeypatch):
    import modelwatch.evaluate as engine
    def intentional_failure(*args, **kwargs):
        raise RuntimeError("synthetic evaluator failure fixture")
    monkeypatch.setattr(engine, "evidently_report", intentional_failure)
    win = make_window(demo, events)
    result = evaluate_window(demo, win["id"])
    # Intentional test-only exception: record error, never declare success.
    assert result["status"] == "failed"
    record = one("SELECT * FROM evaluations")
    assert record["status"] == "failed"
    assert record["summary_json"] is None
    assert evaluate_window(demo, win["id"])["status"] == "already_evaluated"


def test_stored_drift_workflow_with_explicit_mock(demo, events, monkeypatch):
    import modelwatch.evaluate as engine
    win = make_window(demo, events)
    def fake_report(current, ref, cols, config, path, quality=False):
        # This is a test stub ONLY. Never used for production metrics.
        if set(cols) - set(ref):
            raise ValueError("missing reference columns")
        path.parent.mkdir(exist_ok=True, parents=True)
        path.with_suffix(".json").write_text("{}")
        path.with_suffix(".html").write_text("<p>SYNTHETIC TEST FIXTURE</p>")
        return {"tests": []}, ({} if quality else {
            col: {"score": 0.5, "threshold": config.drift_threshold, "drifted": True}
            for col in cols})
    monkeypatch.setattr(engine, "evidently_report", fake_report)
    monkeypatch.setitem(sys.modules, "evidently", SimpleNamespace(__version__="MOCK-TEST-ONLY"))
    result = evaluate_window(demo, win["id"])
    assert result["status"] == "completed"
    assert result["summary"]["feature_drift_count"] == 4
    assert result["summary"]["prediction_drift_count"] == 1
    assert one("SELECT * FROM evaluations")["artifact_html"] is not None
    assert len(many("SELECT * FROM incidents")) == 0  # debounce = two windows


def test_classification_metrics(demo):
    labeled = [{"label": x, "prediction": y, "probabilities": None}
               for x, y in [(0, 0), (0, 1), (1, 1), (2, 2)]]
    metrics = compute_performance(demo, labeled)
    assert metrics["accuracy"] == 0.75
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 4


def test_regression_metrics(demo):
    c = demo.model_copy(update={"task": "regression", "classes": []})
    m = compute_performance(c, [{"label": 2, "prediction": 1},
                                {"label": 4, "prediction": 3},
                                {"label": 6, "prediction": 5}])
    assert m["rmse"] == 1.0
    assert m["mae"] == 1.0


def test_ground_truth_insufficient_then_valid(demo, events):
    batch, start, end = events
    win = make_window(demo, events)
    assert evaluate_performance(demo, win["id"])["status"] == "awaiting_ground_truth"
    from datetime import datetime, timezone
    for i in range(2):
        add_label(GroundTruth(event_id=batch[i]["event_id"], label=0,
                              labeled_at=datetime.now(timezone.utc)), demo)
    result = evaluate_performance(demo, win["id"])
    assert result["status"] == "completed"
    assert result["summary"]["labeled_count"] == 2
    assert evaluate_performance(demo, win["id"])["status"] == "already_evaluated"
