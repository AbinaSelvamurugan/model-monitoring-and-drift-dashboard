"""Versioned threshold policy, durable lifecycle, and webhook delivery with retry."""
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modelwatch.config import Config
from modelwatch.store import connect, dumps, many, one, utcnow

logger = logging.getLogger(__name__)
RULE_VERSION = 1


def policy_version(cfg: Config) -> int:
    # Changes to any configured alert threshold start a new incident generation.
    fields = ("drift_threshold", "drift_share_threshold", "latency_limit_ms",
              "error_rate_limit", "ingestion_failure_limit", "max_missing_rates",
              "performance_accuracy_drop", "performance_rmse_increase",
              "alert_persistence", "alert_recovery", "alert_rules")
    values = cfg.model_dump(mode="json")
    policy = {key: values[key] for key in fields}
    digest = hashlib.blake2s(dumps({"engine_version": RULE_VERSION, "policy": policy}).encode(),
                             digest_size=4).digest()
    return int.from_bytes(digest, "big")


def checks(cfg: Config, summary: dict, kind: str):
    """Only completed evaluations call this method. (rule, observed, threshold, failed)."""
    if kind == "drift":
        q = summary["quality"]
        highest = max(q["missing_rates"].values(), default=0)
        return [
            ("feature_drift_share", summary["feature_drift_share"], cfg.drift_share_threshold,
             summary["feature_drift_share"] >= cfg.drift_share_threshold),
            ("prediction_drift", summary["prediction_drift_count"], 1,
             summary["prediction_drift_count"] >= 1),
            ("inference_error_rate", q["error_rate"], cfg.error_rate_limit,
             q["error_rate"] > cfg.error_rate_limit),
            ("inference_latency_p95", q["latency_p95_ms"], cfg.latency_limit_ms,
             q["latency_p95_ms"] is not None and q["latency_p95_ms"] > cfg.latency_limit_ms),
            ("feature_null_rate", highest, 0,
             any(t["passed"] is False for t in q["tests"])),
        ]
    if kind == "availability":
        return [
            ("missing_monitoring_window", summary["event_count"], 1, summary["event_count"] == 0),
            ("ingestion_failures", summary["rejected_requests"], cfg.ingestion_failure_limit,
             summary["rejected_requests"] > cfg.ingestion_failure_limit)
        ]
    perf, baseline = summary["performance"], summary["baseline"]
    if cfg.task == "classification":
        if "accuracy" not in baseline:
            raise ValueError("approved baseline has no accuracy")
        threshold = baseline["accuracy"] - cfg.performance_accuracy_drop
        return [("performance_accuracy", perf["accuracy"], threshold,
                 perf["accuracy"] < threshold)]
    if "rmse" not in baseline:
        raise ValueError("approved baseline has no rmse")
    threshold = baseline["rmse"] + cfg.performance_rmse_increase
    return [("performance_rmse", perf["rmse"], threshold, perf["rmse"] > threshold)]


def audit(db, incident_id, evaluation_id, change, details):
    db.execute("""INSERT INTO incident_events (incident_id,evaluation_id,change_type,detail_json,created_at)
                  VALUES (?,?,?,?,?)""", (incident_id, evaluation_id, change, dumps(details), utcnow()))


def evaluate_rules(cfg: Config, win: dict, evaluation_id: int, summary: dict, kind: str):
    version = policy_version(cfg)
    for rule, observed, threshold, failed in checks(cfg, summary, kind):
        override = cfg.alert_rules.get(rule)
        if override and not override.enabled:
            continue
        min_count = override.min_count if override else None
        sample_count = summary.get("labeled_count", win.get("event_count", 0))
        if min_count and sample_count < min_count:
            continue
        # group by exact approved reference and versioned rule policy.
        key = ":".join([cfg.model_name, cfg.model_version, win["reference_version"],
                        str(version), rule])
        with connect(write=True) as db:
            progress = db.execute("SELECT * FROM rule_progress WHERE grouping_key=?", (key,)).fetchone()
            if progress and evaluation_id <= progress["last_evaluation_id"]:
                continue
            if progress and progress["last_window_id"] == win["id"]:
                # Re-evaluating a growing delayed-label cohort in the same window cannot
                # satisfy cross-window debounce or recovery requirements.
                continue
            failures = (progress["failures"] if progress else 0) + 1 if failed else 0
            healthy = 0 if failed else (progress["healthy"] if progress else 0) + 1
            db.execute("""INSERT INTO rule_progress VALUES (?,?,?,?,?)
                          ON CONFLICT(grouping_key) DO UPDATE SET failures=excluded.failures,
                            healthy=excluded.healthy,last_evaluation_id=excluded.last_evaluation_id,
                            last_window_id=excluded.last_window_id""",
                       (key, failures, healthy, evaluation_id, win["id"]))
            incident = db.execute("SELECT * FROM incidents WHERE grouping_key=?", (key,)).fetchone()
            # null/data-integrity conditions bypass debounce; performance needs consecutive windows.
            debounce = 1 if rule == "feature_null_rate" else (
                override.persistence if override and override.persistence else cfg.alert_persistence)
            if failed and failures >= debounce:
                if incident is None:
                    cur = db.execute("""INSERT INTO incidents
                        (grouping_key,rule_id,rule_version,state,severity,first_evaluation_id,
                         last_evaluation_id,observed_json,threshold_json,failures,healthy,
                         delivery_state,created_at,updated_at,destination_env,cooldown_seconds)
                         VALUES (?,?,?,'new',?,?,?,?,?,?,0,'pending',?,?,?,?)""",
                        (key, rule, version,
                         (override.severity if override and override.severity else
                          ("critical" if rule.startswith("performance") else "warning")),
                         evaluation_id, evaluation_id, dumps(observed), dumps(threshold), failures,
                         utcnow(), utcnow(),
                         override.destination_env if override else "MODELWATCH_WEBHOOK_URL",
                         override.cooldown_seconds if override and override.cooldown_seconds is not None
                         else cfg.alert_cooldown_seconds))
                    audit(db, cur.lastrowid, evaluation_id, "new", {"observed": observed, "threshold": threshold})
                elif incident["state"] == "resolved":
                    db.execute("""UPDATE incidents SET state='reopened',failures=?,healthy=0,
                              last_evaluation_id=?,observed_json=?,delivery_state='pending',updated_at=?
                              WHERE id=?""", (failures, evaluation_id, dumps(observed), utcnow(), incident["id"]))
                    audit(db, incident["id"], evaluation_id, "reopened", {"observed": observed})
                else:
                    db.execute("""UPDATE incidents SET failures=?,healthy=0,last_evaluation_id=?,
                                observed_json=?,state=CASE WHEN state='new' THEN 'active' ELSE state END,
                                updated_at=? WHERE id=?""",
                               (failures, evaluation_id, dumps(observed), utcnow(), incident["id"]))
                    audit(db, incident["id"], evaluation_id, "observed", {"observed": observed})
            elif not failed and incident and incident["state"] not in {"resolved"}:
                required_recovery = (override.recovery if override and override.recovery
                                     else cfg.alert_recovery)
                if healthy >= required_recovery:
                    db.execute("""UPDATE incidents SET state='resolved',healthy=?,note=?,updated_at=?
                                   WHERE id=?""", (healthy, "auto-recovered after eligible healthy windows",
                                                    utcnow(), incident["id"]))
                    audit(db, incident["id"], evaluation_id, "resolved", {"healthy_windows": healthy})
                else:
                    db.execute("UPDATE incidents SET healthy=?,updated_at=? WHERE id=?",
                               (healthy, utcnow(), incident["id"]))


def change_incident(incident_id: int, action: str, note: str):
    target = {"acknowledge": "acknowledged", "resolve": "resolved"}.get(action)
    if not target:
        raise ValueError("invalid transition")
    with connect(write=True) as db:
        old = db.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not old:
            raise KeyError("incident not found")
        if old["state"] == "resolved" or (old["state"] == "acknowledged" and action == "acknowledge"):
            raise ValueError("incident already transitioned")
        db.execute("UPDATE incidents SET state=?,note=?,updated_at=? WHERE id=?",
                   (target, note, utcnow(), incident_id))
        audit(db, incident_id, old["last_evaluation_id"], action, {"note": note})
    return one("SELECT * FROM incidents WHERE id=?", (incident_id,))


def dispatch(cfg: Config):
    """Webhook delivery acknowledgement == HTTP 2xx. Failed attempts durable and retryable."""
    incidents = many("""SELECT * FROM incidents WHERE state IN ('new','active','reopened','acknowledged')
                        AND delivery_state != 'delivered' ORDER BY id LIMIT 50""")
    results = []
    for incident in incidents:
        webhook = os.getenv(incident["destination_env"], "").strip()
        attempts = many("SELECT * FROM delivery_attempts WHERE incident_id=? ORDER BY id DESC LIMIT 1",
                        (incident["id"],))
        if attempts and attempts[0]["next_attempt_at"] and attempts[0]["next_attempt_at"] > utcnow():
            continue
        if incident["last_sent_at"]:
            last_sent = datetime.fromisoformat(incident["last_sent_at"])
            if (datetime.now(timezone.utc) - last_sent).total_seconds() < incident["cooldown_seconds"]:
                continue
        if not webhook:
            # Not successful. Operator can configure a URL and retry later.
            with connect(write=True) as db:
                db.execute("UPDATE incidents SET delivery_state='unconfigured' WHERE id=?", (incident["id"],))
            results.append({"incident_id": incident["id"], "status": "unconfigured"})
            continue
        payload = dumps({"incident_id": incident["id"], "rule_id": incident["rule_id"],
                         "severity": incident["severity"], "state": incident["state"],
                         "observed": json.loads(incident["observed_json"]),
                         "threshold": json.loads(incident["threshold_json"]),
                         "evaluation_id": incident["last_evaluation_id"]}).encode()
        status, code, error = "failed", None, None
        try:
            request = Request(webhook, data=payload, headers={"Content-Type": "application/json",
                "Idempotency-Key": f"modelwatch-{incident['id']}-{incident['last_evaluation_id']}"},
                method="POST")
            with urlopen(request, timeout=cfg.webhook_timeout_seconds) as response:
                code = response.status
                if 200 <= code < 300:
                    status = "delivered"
                else:
                    error = f"HTTP {code}"
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            code = e.code if isinstance(e, HTTPError) else None
            error = str(e)[:300]
            logger.warning("webhook delivery failed incident=%s: %s", incident["id"], error)
        retry_seconds = min(3600, 60 * 2 ** min(len(attempts), 6))
        next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)).isoformat() if status == "failed" else None
        with connect(write=True) as db:
            db.execute("""INSERT INTO delivery_attempts
                (incident_id,destination,attempted_at,status,response_code,error,next_attempt_at)
                 VALUES (?,?,?,?,?,?,?)""", (incident["id"], webhook, utcnow(), status, code,
                                              error, next_attempt))
            db.execute("UPDATE incidents SET delivery_state=?,last_sent_at=?,updated_at=? WHERE id=?",
                       (status, utcnow() if status == "delivered" else incident["last_sent_at"],
                        utcnow(), incident["id"]))
            audit(db, incident["id"], incident["last_evaluation_id"], "delivery", {"status": status,
                                  "http_code": code, "error": error})
        results.append({"incident_id": incident["id"], "status": status, "http_code": code})
    return results
