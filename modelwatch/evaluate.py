"""Persist Evidently snapshots; isolate metric extraction from alert policy."""
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)

from modelwatch.config import Config, report_dir
from modelwatch.store import connect, current_reference, dumps, one, utcnow
from modelwatch.windows import refresh_eligibility

log = logging.getLogger(__name__)


def extract_drift(snapshot_result: dict, columns: list[str], threshold: float):
    """0.7 Snapshot.dict() metrics: config.column + scalar value for ValueDrift.

    Do not assume a dataset-level failure means any model-performance degradation.
    Fail closed on API output changes instead of creating zero-drift dashboards.
    """
    scores = {}
    for metric in snapshot_result.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        cfg = metric.get("config", {})
        name = str(metric.get("metric_name", ""))
        col = cfg.get("column") if isinstance(cfg, dict) else None
        if name.startswith("ValueDrift(") and col in columns:
            value = metric.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid Evidently ValueDrift result for {col}")
            scores[col] = {"score": float(value), "threshold": threshold,
                           "drifted": bool(value >= threshold)}
    if set(scores) != set(columns):
        raise ValueError(f"Evidently ValueDrift results missing for: {sorted(set(columns)-set(scores))}")
    return scores


def evidently_report(current: pd.DataFrame, reference: pd.DataFrame, columns: list[str],
                      config: Config, path_base: Path, quality=False):
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    if quality:
        report = Report([DataSummaryPreset()], include_tests=True)
        snapshot = report.run(current[columns], reference[columns])
    else:
        report = Report([DataDriftPreset(columns=columns, method=config.drift_method,
                                         threshold=config.drift_threshold,
                                         drift_share=config.drift_share_threshold)], include_tests=True)
        snapshot = report.run(current[columns], reference[columns])
    path_base.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_json(str(path_base.with_suffix(".json")))
    snapshot.save_html(str(path_base.with_suffix(".html")))
    # .json() uses Evidently's own encoder for numpy values/enums.
    result = json.loads(snapshot.json())
    scores = {} if quality else extract_drift(result, columns, config.drift_threshold)
    return result, scores


def window_data(cfg: Config, win: dict):
    with connect() as db:
        records = [dict(r) for r in db.execute("""SELECT * FROM events WHERE model_version=?
             AND reference_version=? AND ts>=? AND ts<? AND late_after_evaluation=0
             ORDER BY ts,event_id""",
             (win["model_version"], win["reference_version"], win["start"], win["end"]))]
    rows = [json.loads(r["payload"]) for r in records]
    return records, rows


def report_quality(rows, cfg):
    n = len(rows)
    missing = {col: sum(row["features"].get(col) is None for row in rows) / n
               for col in cfg.feature_types}
    errors = sum(r["status"] == "error" for r in rows)
    latency = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    return {"missing_rates": missing, "error_count": errors, "error_rate": errors/n,
            "latency_count": len(latency),
            "latency_p50_ms": float(np.percentile(latency, 50)) if latency else None,
            "latency_p90_ms": float(np.percentile(latency, 90)) if len(latency) >= 10 else None,
            "latency_p95_ms": float(np.percentile(latency, 95)) if len(latency) >= 20 else None,
            "throughput_per_second": n/cfg.window_seconds,
            "tests": [{"check": f"missing:{col}", "observed": value,
                       "expected": cfg.max_missing_rates.get(col, 0.0 if col not in cfg.allow_missing else 1.0),
                       "passed": value <= cfg.max_missing_rates.get(col, 0.0 if col not in cfg.allow_missing else 1.0)}
                      for col, value in missing.items()]}


def _insert_evaluation(db, window_id, kind, label_count):
    cur = db.execute("""INSERT INTO evaluations(window_id,kind,label_count,status,created_at)
                     VALUES (?,?,?,?,?)""", (window_id, kind, label_count, "running", utcnow()))
    return cur.lastrowid


def _finalize(eval_id, status, summary=None, tests=None, artifacts=None, error=None):
    with connect(write=True) as db:
        db.execute("""UPDATE evaluations SET status=?,summary_json=?,tests_json=?,
            artifact_json=?,artifact_html=?,error=?,completed_at=? WHERE id=?""",
            (status, dumps(summary) if summary is not None else None,
             dumps(tests) if tests is not None else None,
             artifacts.get("json") if artifacts else None,
             artifacts.get("html") if artifacts else None,
             error, utcnow(), eval_id))


def evaluate_window(cfg: Config, window_id: int):
    """Explicit claim; never silently reprocess an evaluation (even after failure)."""
    win = refresh_eligibility(cfg, window_id)
    if not win["eligible"]:
        return {"status": win["status"], "window_id": window_id, "event_count": win["event_count"]}
    with connect(write=True) as db:
        existing = db.execute("SELECT id,status FROM evaluations WHERE window_id=? AND kind='drift'",
                              (window_id,)).fetchone()
        if existing:
            return {"status": "already_evaluated", "evaluation_id": existing["id"],
                    "original_status": existing["status"]}
        eval_id = _insert_evaluation(db, window_id, "drift", 0)
        db.execute("UPDATE windows SET status='running',updated_at=? WHERE id=?", (utcnow(), window_id))
    artifacts = None
    try:
        ref = current_reference(cfg)
        if ref["version"] != win["reference_version"]:
            raise ValueError("reference changed during window evaluation")
        reference = pd.read_csv(ref["path"])
        expected = list(cfg.feature_types) + ["prediction"]
        if set(expected) - set(reference):
            raise ValueError("reference columns missing")
        _, rows = window_data(cfg, win)
        # Failed inference events contribute to error rates but not prediction/feature drift.
        successes = [r for r in rows if r["status"] == "ok"]
        if len(successes) < cfg.min_events:
            raise ValueError("insufficient successful predictions for drift test")
        current = pd.DataFrame([dict(r["features"], prediction=r["prediction"]) for r in successes])
        # Optional classification probability-vector drift is measured only when
        # reference + every successful event supply the same ordered scores.
        score_cols = [f"score_{i}" for i in range(len(cfg.classes))]
        score_status = "not_applicable" if cfg.task != "classification" else "awaiting_compatible_scores"
        if cfg.task == "classification" and all(r.get("probabilities") is not None for r in successes):
            if all(col in reference for col in score_cols):
                for i, col in enumerate(score_cols):
                    current[col] = [r["probabilities"][i] for r in successes]
                score_status = "ready"
        for col in cfg.feature_types:
            if cfg.feature_types[col] == "categorical":
                current[col] = current[col].astype("category")
                reference[col] = reference[col].astype("category")
        if cfg.task == "classification":
            current["prediction"] = current["prediction"].astype("category")
            reference["prediction"] = reference["prediction"].astype("category")
        base = report_dir() / f"evaluation-{eval_id}"
        quality_raw, _ = evidently_report(current, reference, list(cfg.feature_types), cfg,
                                          base.with_name(base.name + "-quality"), quality=True)
        feature_raw, features = evidently_report(current, reference, list(cfg.feature_types), cfg,
                                                base.with_name(base.name + "-features"))
        pred_raw, predictions = evidently_report(current, reference, ["prediction"], cfg,
                                                base.with_name(base.name + "-prediction"))
        score_raw, scores = ({}, {})
        if score_status == "ready":
            score_raw, scores = evidently_report(current, reference, score_cols, cfg,
                                                base.with_name(base.name + "-scores"))
            score_status = "evaluated"
        quality = report_quality(rows, cfg)
        tests = {"data_quality": quality["tests"], "evidently_quality": quality_raw.get("tests", []),
                 "feature": feature_raw.get("tests", []), "prediction": pred_raw.get("tests", []),
                 "scores": score_raw.get("tests", [])}
        from modelwatch.alerts import policy_version
        ref_counts = reference["prediction"].value_counts(normalize=True).to_dict()
        cur_counts = current["prediction"].value_counts(normalize=True).to_dict()
        summary = {"event_count": len(rows), "successful_predictions": len(successes),
                   "rule_policy_version": policy_version(cfg),
                   "prediction_shares": {"reference": {str(k): float(v) for k, v in ref_counts.items()},
                                         "current": {str(k): float(v) for k, v in cur_counts.items()}},
                   "reference_version": ref["version"], "method": cfg.drift_method,
                   "threshold": cfg.drift_threshold, "quality": quality,
                   "features": features, "feature_drift_count": sum(v["drifted"] for v in features.values()),
                   "feature_drift_share": sum(v["drifted"] for v in features.values())/len(features),
                   "prediction": predictions["prediction"],
                   "prediction_drift_count": int(predictions["prediction"]["drifted"]),
                   "probability_score_status": score_status, "probability_scores": scores,
                   "evidently_version": __import__("evidently").__version__,
                   "report_artifacts": {
                       kind: {"json": str(base.with_name(base.name + f"-{kind}.json")),
                              "html": str(base.with_name(base.name + f"-{kind}.html"))}
                       for kind in (("quality", "features", "prediction", "scores")
                                    if score_status == "evaluated" else ("quality", "features", "prediction"))}}
        artifacts = {"json": str(base.with_name(base.name + "-features.json")),
                     "html": str(base.with_name(base.name + "-features.html"))}
        _finalize(eval_id, "completed", summary, tests, artifacts)
        with connect(write=True) as db:
            db.execute("UPDATE windows SET status='completed',updated_at=? WHERE id=?", (utcnow(), window_id))
        from modelwatch.alerts import evaluate_rules
        evaluate_rules(cfg, win, eval_id, summary, kind="drift")
        return {"status": "completed", "evaluation_id": eval_id, "summary": summary}
    except Exception as e:
        log.exception("drift evaluation failed for window %s", window_id)
        _finalize(eval_id, "failed", error=f"{type(e).__name__}: {e}")
        with connect(write=True) as db:
            db.execute("UPDATE windows SET status='failed',updated_at=? WHERE id=?", (utcnow(), window_id))
        return {"status": "failed", "evaluation_id": eval_id, "error": f"{type(e).__name__}: {e}"}


def compute_performance(cfg: Config, labeled_rows: list[dict]):
    y_true = [r["label"] for r in labeled_rows]
    y_pred = [r["prediction"] for r in labeled_rows]
    if cfg.task == "classification":
        result = {"accuracy": float(accuracy_score(y_true, y_pred)),
                  "precision_macro": float(precision_score(y_true, y_pred, labels=cfg.classes,
                                                           zero_division=0, average="macro")),
                  "recall_macro": float(recall_score(y_true, y_pred, labels=cfg.classes,
                                                     zero_division=0, average="macro")),
                  "f1_macro": float(f1_score(y_true, y_pred, labels=cfg.classes,
                                             zero_division=0, average="macro")),
                  "confusion_matrix": confusion_matrix(y_true, y_pred, labels=cfg.classes).tolist()}
        if len(cfg.classes) == 2 and all(r.get("probabilities") for r in labeled_rows):
            try:
                scores = [r["probabilities"][1] for r in labeled_rows]
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
            except ValueError:
                result["roc_auc"] = None
        return result
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()), "count": len(y_true)}


def evaluate_performance(cfg: Config, window_id: int):
    win = one("SELECT * FROM windows WHERE id=?", (window_id,))
    if not win:
        raise KeyError("window not found")
    if not win["eligible"]:
        return {"status": "insufficient", "window_id": window_id}
    records, rows = window_data(cfg, win)
    labeled = []
    for rec, row in zip(records, rows):
        if rec["label_json"] is not None and row["status"] == "ok":
            labeled.append({"label": json.loads(rec["label_json"]),
                            "prediction": row["prediction"], "probabilities": row.get("probabilities")})
    if len(labeled) < cfg.min_labels:
        return {"status": "awaiting_ground_truth", "labeled": len(labeled), "required": cfg.min_labels}
    with connect(write=True) as db:
        existing = db.execute("""SELECT id,status FROM evaluations WHERE window_id=?
                            AND kind='performance' AND label_count=?""", (window_id, len(labeled))).fetchone()
        if existing:
            return {"status": "already_evaluated", "evaluation_id": existing["id"]}
        eval_id = _insert_evaluation(db, window_id, "performance", len(labeled))
    try:
        perf = compute_performance(cfg, labeled)
        baseline = json.loads(current_reference(cfg)["baseline_json"])
        from modelwatch.alerts import policy_version
        summary = {"labeled_count": len(labeled), "total_event_count": win["event_count"],
                   "rule_policy_version": policy_version(cfg),
                   "label_coverage": len(labeled)/win["event_count"], "task": cfg.task,
                   "reference_version": win["reference_version"], "baseline": baseline,
                   "performance": perf, "selection": "labeled successful events only"}
        _finalize(eval_id, "completed", summary, [])
        from modelwatch.alerts import evaluate_rules
        evaluate_rules(cfg, win, eval_id, summary, kind="performance")
        return {"status": "completed", "evaluation_id": eval_id, "summary": summary}
    except Exception as e:
        log.exception("performance evaluation failed")
        _finalize(eval_id, "failed", error=f"{type(e).__name__}: {e}")
        return {"status": "failed", "evaluation_id": eval_id, "error": str(e)}


def evaluate_availability(cfg: Config, window_id: int):
    """Separate service-health evidence; empty windows NEVER imply a no-drift result."""
    win = refresh_eligibility(cfg, window_id)
    with connect(write=True) as db:
        previous = db.execute("SELECT COUNT(*) FROM events WHERE ts<? AND model_version=?",
                              (win["start"], cfg.model_version)).fetchone()[0]
        if not previous and win["event_count"] == 0:
            return {"status": "not_monitoring_yet"}
        existing = db.execute("SELECT id FROM evaluations WHERE window_id=? AND kind='availability'",
                              (window_id,)).fetchone()
        if existing:
            return {"status": "already_evaluated", "evaluation_id": existing["id"]}
        eval_id = _insert_evaluation(db, window_id, "availability", 0)
        failures = db.execute("""SELECT COUNT(*) FROM ingest_errors WHERE ts>=? AND ts<?""",
                              (win["start"], win["end"])).fetchone()[0]
    from modelwatch.alerts import policy_version
    summary = {"event_count": win["event_count"], "rejected_requests": failures,
               "rule_policy_version": policy_version(cfg),
               "reference_version": win["reference_version"],
               "status": "events_observed" if win["event_count"] else "no_events",
               "evidence": "event counts and ingestion rejection records; no drift inference"}
    _finalize(eval_id, "completed", summary, [])
    from modelwatch.alerts import evaluate_rules
    evaluate_rules(cfg, win, eval_id, summary, kind="availability")
    return {"status": "completed", "evaluation_id": eval_id, "summary": summary}
