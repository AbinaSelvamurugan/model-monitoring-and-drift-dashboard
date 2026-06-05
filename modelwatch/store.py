"""SQLite transactions; single-writer-safe local/reference deployment."""
import contextlib
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from modelwatch.config import database_path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@contextlib.contextmanager
def connect(write=False):
    path = Path(database_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        if write:
            connection.execute("BEGIN IMMEDIATE")
        yield connection
        if write:
            connection.commit()
    except Exception:
        if write:
            connection.rollback()
        raise
    finally:
        connection.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS references_catalog (
 version TEXT PRIMARY KEY, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
 schema_version TEXT NOT NULL, preprocessing_version TEXT NOT NULL, path TEXT NOT NULL,
 sha256 TEXT NOT NULL, baseline_json TEXT NOT NULL, approved_at TEXT NOT NULL,
 approval_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
 event_id TEXT PRIMARY KEY, request_id TEXT, ts TEXT NOT NULL, model_name TEXT NOT NULL,
 model_version TEXT NOT NULL, schema_version TEXT NOT NULL, reference_version TEXT NOT NULL,
 payload TEXT NOT NULL, label_json TEXT, labeled_at TEXT, created_at TEXT NOT NULL,
 late_after_evaluation INTEGER NOT NULL DEFAULT 0,
 FOREIGN KEY(reference_version) REFERENCES references_catalog(version)
);
CREATE INDEX IF NOT EXISTS events_window_idx ON events(model_version,ts);
CREATE TABLE IF NOT EXISTS ingest_errors (
 id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, status INTEGER NOT NULL,
 reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS windows (
 id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, model_version TEXT NOT NULL,
 reference_version TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
 event_count INTEGER NOT NULL, status TEXT NOT NULL, eligible INTEGER NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 UNIQUE(model_name,model_version,reference_version,start,end)
);
CREATE TABLE IF NOT EXISTS evaluations (
 id INTEGER PRIMARY KEY AUTOINCREMENT, window_id INTEGER NOT NULL, kind TEXT NOT NULL,
 label_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, summary_json TEXT,
 tests_json TEXT, artifact_json TEXT, artifact_html TEXT,
 error TEXT, created_at TEXT NOT NULL, completed_at TEXT,
 FOREIGN KEY(window_id) REFERENCES windows(id), UNIQUE(window_id,kind,label_count)
);
CREATE TABLE IF NOT EXISTS incidents (
 id INTEGER PRIMARY KEY AUTOINCREMENT, grouping_key TEXT UNIQUE NOT NULL,
 rule_id TEXT NOT NULL, rule_version INTEGER NOT NULL,
 state TEXT NOT NULL, severity TEXT NOT NULL, first_evaluation_id INTEGER NOT NULL,
 last_evaluation_id INTEGER NOT NULL, observed_json TEXT NOT NULL,
 threshold_json TEXT NOT NULL, failures INTEGER NOT NULL DEFAULT 0,
 healthy INTEGER NOT NULL DEFAULT 0, last_sent_at TEXT, delivery_state TEXT NOT NULL,
 note TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 destination_env TEXT NOT NULL DEFAULT 'MODELWATCH_WEBHOOK_URL',
 cooldown_seconds INTEGER NOT NULL DEFAULT 3600
);
CREATE TABLE IF NOT EXISTS incident_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER NOT NULL,
 evaluation_id INTEGER, change_type TEXT NOT NULL, detail_json TEXT NOT NULL,
 created_at TEXT NOT NULL, FOREIGN KEY(incident_id) REFERENCES incidents(id)
);
CREATE TABLE IF NOT EXISTS rule_progress (
 grouping_key TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0,
 healthy INTEGER NOT NULL DEFAULT 0, last_evaluation_id INTEGER NOT NULL DEFAULT 0, last_window_id INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS delivery_attempts (
 id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER NOT NULL,
 destination TEXT NOT NULL, attempted_at TEXT NOT NULL, status TEXT NOT NULL,
 response_code INTEGER, error TEXT, next_attempt_at TEXT,
 FOREIGN KEY(incident_id) REFERENCES incidents(id)
);
"""


def init_db():
    with connect(write=True) as db:
        db.executescript(SCHEMA)
        if "last_window_id" not in [r[1] for r in db.execute("PRAGMA table_info(rule_progress)")]:
            db.execute("ALTER TABLE rule_progress ADD COLUMN last_window_id INTEGER NOT NULL DEFAULT 0")
        if "late_after_evaluation" not in [r[1] for r in db.execute("PRAGMA table_info(events)")]:
            db.execute("ALTER TABLE events ADD COLUMN late_after_evaluation INTEGER NOT NULL DEFAULT 0")
        fields = [r[1] for r in db.execute("PRAGMA table_info(incidents)")]
        if "destination_env" not in fields:
            db.execute("ALTER TABLE incidents ADD COLUMN destination_env TEXT NOT NULL DEFAULT 'MODELWATCH_WEBHOOK_URL'")
        if "cooldown_seconds" not in fields:
            db.execute("ALTER TABLE incidents ADD COLUMN cooldown_seconds INTEGER NOT NULL DEFAULT 3600")


def one(sql, params=()):
    with connect() as db:
        row = db.execute(sql, params).fetchone()
        return dict(row) if row else None


def many(sql, params=()):
    with connect() as db:
        return [dict(r) for r in db.execute(sql, params).fetchall()]


def register_reference(version, file_path, config, reason, baseline):
    p = Path(file_path).resolve()
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    with connect(write=True) as db:
        existing = db.execute("SELECT * FROM references_catalog WHERE version=?", (version,)).fetchone()
        if existing:
            if existing["sha256"] != digest:
                raise ValueError("reference version is immutable; use a new version")
            return dict(existing)
        db.execute("""INSERT INTO references_catalog VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            version, config.model_name, config.model_version, config.schema_version,
            config.preprocessing_version, str(p), digest, dumps(baseline), utcnow(), reason))
    return one("SELECT * FROM references_catalog WHERE version=?", (version,))


def current_reference(config):
    ref = one("SELECT * FROM references_catalog WHERE version=?", (config.reference_version,))
    if ref is None:
        raise ValueError("approved reference missing; run `modelwatch prepare`")
    if (ref["model_version"] != config.model_version or
        ref["schema_version"] != config.schema_version or
        ref["preprocessing_version"] != config.preprocessing_version):
        raise ValueError("configured model/schema/preprocessing differs from reference")
    path = Path(ref["path"])
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("approved reference file absent or modified")
    return ref


def purge_expired(retention_days: int):
    """Delete old raw events/labels. Completed aggregate evaluations stay for audit."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with connect(write=True) as db:
        count = db.execute("DELETE FROM events WHERE ts<? AND created_at<?", (cutoff, cutoff)).rowcount
        errors = db.execute("DELETE FROM ingest_errors WHERE ts<?", (cutoff,)).rowcount
    return {"deleted_events": count, "deleted_rejections": errors}


def flag_interrupted():
    """On worker restart, make abandoned claims visible; never reclassify as healthy."""
    with connect(write=True) as db:
        rows = db.execute("SELECT id,window_id FROM evaluations WHERE status='running'").fetchall()
        for row in rows:
            db.execute("""UPDATE evaluations SET status='failed',error='interrupted evaluation; operator review required',
                          completed_at=? WHERE id=?""", (utcnow(), row["id"]))
            db.execute("UPDATE windows SET status='failed',updated_at=? WHERE id=?",
                       (utcnow(), row["window_id"]))
    return len(rows)
