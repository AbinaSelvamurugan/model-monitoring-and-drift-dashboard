"""All scores in these alert unit tests are SYNTHETIC TEST FIXTURES."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modelwatch.alerts import change_incident, dispatch, evaluate_rules
from modelwatch.store import many, one


def summary(drift_share=1.0, prediction=True, error=0.0, latency=100.0):
    return {"feature_drift_share": drift_share, "prediction_drift_count": int(prediction),
            "quality": {"missing_rates": {"a": 0}, "error_rate": error,
                        "latency_p95_ms": latency, "tests": []}}


def win(i):
    return {"id": i, "reference_version": "iris-holdout-1"}


def test_persistence_and_grouping(demo):
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    assert many("SELECT * FROM incidents") == []
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    incidents = many("SELECT * FROM incidents")
    assert len(incidents) == 2  # feature and prediction are different rules
    assert {i["rule_id"] for i in incidents} == {"feature_drift_share", "prediction_drift"}
    evaluate_rules(demo, win(3), 12, summary(), "drift")
    assert len(many("SELECT * FROM incidents")) == 2
    assert len(many("SELECT * FROM incident_events")) == 4


def test_same_window_cannot_satisfy_debounce(demo):
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(1), 11, summary(), "drift")
    assert not many("SELECT * FROM incidents")


def test_failed_evaluation_cannot_resolve(demo):
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    # No evaluate_rules invocation on failed/unexecuted windows.
    assert all(i["state"] != "resolved" for i in many("SELECT * FROM incidents"))
    evaluate_rules(demo, win(3), 12, summary(0, False), "drift")
    assert all(i["state"] != "resolved" for i in many("SELECT * FROM incidents"))
    evaluate_rules(demo, win(4), 13, summary(0, False), "drift")
    assert all(i["state"] == "resolved" for i in many("SELECT * FROM incidents"))
    evaluate_rules(demo, win(5), 14, summary(), "drift")
    evaluate_rules(demo, win(6), 15, summary(), "drift")
    assert all(i["state"] == "reopened" for i in many("SELECT * FROM incidents"))


def test_acknowledge_resolve_audited(demo):
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    item = one("SELECT * FROM incidents ORDER BY id")
    assert change_incident(item["id"], "acknowledge", "Investigating batch")['state'] == "acknowledged"
    assert change_incident(item["id"], "resolve", "Mitigated upstream input")['state'] == "resolved"
    assert {a["change_type"] for a in many("SELECT * FROM incident_events WHERE incident_id=?",
                                             (item["id"],))} >= {"acknowledge", "resolve"}
    with pytest.raises(ValueError):
        change_incident(item["id"], "resolve", "already resolved")


def test_performance_regression_rule(demo):
    value = {"performance": {"accuracy": 0.2}, "baseline": {"accuracy": 0.95}}
    evaluate_rules(demo, win(1), 10, value, "performance")
    evaluate_rules(demo, win(2), 11, value, "performance")
    incident = one("SELECT * FROM incidents")
    assert incident["rule_id"] == "performance_accuracy"
    assert json.loads(incident["observed_json"]) == 0.2


def test_unconfigured_webhook_never_delivered(demo, monkeypatch):
    monkeypatch.delenv("MODELWATCH_WEBHOOK_URL", raising=False)
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    statuses = dispatch(demo)
    assert all(row["status"] == "unconfigured" for row in statuses)
    assert not many("SELECT * FROM delivery_attempts")
    assert all(i["delivery_state"] == "unconfigured" for i in many("SELECT * FROM incidents"))


class Webhook(BaseHTTPRequestHandler):
    code = 204
    arrivals = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.__class__.arrivals.append(json.loads(self.rfile.read(length)))
        self.send_response(self.code)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def webhook():
    server = HTTPServer(("127.0.0.1", 0), Webhook)
    server.timeout = 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_real_http_webhook_success(demo, monkeypatch, webhook):
    monkeypatch.setenv("MODELWATCH_WEBHOOK_URL", f"http://127.0.0.1:{webhook.server_port}/")
    Webhook.code, Webhook.arrivals = 204, []
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    outcomes = dispatch(demo)
    assert len(outcomes) == 2
    assert all(x["status"] == "delivered" for x in outcomes)
    assert len(Webhook.arrivals) == 2
    assert all(r["status"] == "delivered" and r["response_code"] == 204
               for r in many("SELECT * FROM delivery_attempts"))
    assert dispatch(demo) == []  # no repeat once acknowledged


def test_webhook_failure_retained_with_retry(demo, monkeypatch, webhook):
    monkeypatch.setenv("MODELWATCH_WEBHOOK_URL", f"http://127.0.0.1:{webhook.server_port}/")
    Webhook.code, Webhook.arrivals = 503, []
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    outcomes = dispatch(demo)
    assert all(x["status"] == "failed" for x in outcomes)
    attempts = many("SELECT * FROM delivery_attempts")
    assert all(x["next_attempt_at"] and x["response_code"] == 503 for x in attempts)
    assert dispatch(demo) == []  # enforces backoff
    assert all(i["delivery_state"] == "failed" for i in many("SELECT * FROM incidents"))


def test_retry_after_failure_can_succeed(demo, monkeypatch, webhook):
    from modelwatch.store import connect
    monkeypatch.setenv("MODELWATCH_WEBHOOK_URL", f"http://127.0.0.1:{webhook.server_port}/")
    Webhook.code, Webhook.arrivals = 503, []
    evaluate_rules(demo, win(1), 10, summary(), "drift")
    evaluate_rules(demo, win(2), 11, summary(), "drift")
    assert all(result["status"] == "failed" for result in dispatch(demo))
    with connect(write=True) as db:
        db.execute("UPDATE delivery_attempts SET next_attempt_at='2000-01-01T00:00:00+00:00'")
    Webhook.code = 200
    assert all(result["status"] == "delivered" for result in dispatch(demo))
    assert len(many("SELECT * FROM delivery_attempts")) == 4
    assert all(x["delivery_state"] == "delivered" for x in many("SELECT * FROM incidents"))


def test_rule_overrides_disable_or_customize(demo):
    from modelwatch.config import RuleOverride
    from modelwatch.alerts import policy_version
    c = demo.model_copy(update={"alert_rules": {
        "feature_drift_share": RuleOverride(enabled=False),
        "prediction_drift": RuleOverride(persistence=1, severity="critical", cooldown_seconds=0)}})
    assert policy_version(c) != policy_version(demo)
    evaluate_rules(c, win(1), 10, summary(), "drift")
    found = many("SELECT * FROM incidents")
    assert len(found) == 1
    assert found[0]["rule_id"] == "prediction_drift"
    assert found[0]["severity"] == "critical"
    assert found[0]["cooldown_seconds"] == 0
