"""Fixed UTC event-time windows, transactionally claimed by worker/evaluator."""
from datetime import datetime, timezone

from modelwatch.config import Config
from modelwatch.store import connect, current_reference, one, utcnow


def create_window(cfg: Config, start: datetime, end: datetime):
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("valid aware window boundaries required")
    if end > datetime.now(timezone.utc):
        raise ValueError("window must be closed before evaluation")
    ref = current_reference(cfg)
    start, end = start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()
    with connect(write=True) as db:
        existing = db.execute("""SELECT id FROM windows WHERE model_name=? AND model_version=?
                        AND reference_version=? AND start=? AND end=?""",
                        (cfg.model_name, cfg.model_version, ref["version"], start, end)).fetchone()
        if existing:
            return one_window(db, existing["id"])
        n = db.execute("""SELECT COUNT(*) FROM events WHERE model_name=? AND model_version=?
                    AND reference_version=? AND ts>=? AND ts<? AND late_after_evaluation=0""",
                    (cfg.model_name, cfg.model_version, ref["version"], start, end)).fetchone()[0]
        status = "pending" if n >= cfg.min_events else ("insufficient" if n else "empty")
        cursor = db.execute("""INSERT INTO windows
            (model_name,model_version,reference_version,start,end,event_count,status,eligible,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (cfg.model_name, cfg.model_version, ref["version"], start, end, n, status,
             int(n >= cfg.min_events), utcnow(), utcnow()))
        return one_window(db, cursor.lastrowid)


def one_window(db, id):
    result = db.execute("SELECT * FROM windows WHERE id=?", (id,)).fetchone()
    return dict(result) if result else None


def scheduled_window(cfg):
    """Last fully closed UTC fixed-duration window; replaying is safe."""
    now = int(datetime.now(timezone.utc).timestamp())
    end_ts = now // cfg.window_seconds * cfg.window_seconds
    start = datetime.fromtimestamp(end_ts - cfg.window_seconds, timezone.utc)
    end = datetime.fromtimestamp(end_ts, timezone.utc)
    return create_window(cfg, start, end)


def refresh_eligibility(cfg, id):
    """Permit late arrivals to upgrade an incomplete window, never replace completed evaluations."""
    with connect(write=True) as db:
        win = one_window(db, id)
        if not win:
            raise KeyError("window not found")
        if win["model_version"] != cfg.model_version or win["reference_version"] != cfg.reference_version:
            raise ValueError("window/reference mismatch")
        n = db.execute("""SELECT COUNT(*) FROM events WHERE model_version=? AND
                 reference_version=? AND ts>=? AND ts<? AND late_after_evaluation=0""", (
                 win["model_version"], win["reference_version"], win["start"], win["end"])).fetchone()[0]
        status = win["status"] if win["status"] in {"completed", "running", "failed"} else (
            "pending" if n >= cfg.min_events else ("insufficient" if n else "empty"))
        db.execute("UPDATE windows SET event_count=?,eligible=?,status=?,updated_at=? WHERE id=?",
                   (n, int(n >= cfg.min_events), status, utcnow(), id))
        return one_window(db, id)


def backfill_windows(cfg: Config, max_recent=48):
    """Create recent closed fixed windows, including gaps after first observed event.

    Older event-time windows can always be created explicitly by an operator.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    last_end = now // cfg.window_seconds * cfg.window_seconds
    first = one("""SELECT MIN(ts) AS ts FROM events WHERE model_version=? AND reference_version=?""",
                (cfg.model_version, cfg.reference_version))
    if not first or not first["ts"]:
        return [scheduled_window(cfg)]
    earliest = int(datetime.fromisoformat(first["ts"]).timestamp())
    begin = max(earliest // cfg.window_seconds * cfg.window_seconds,
                last_end - max_recent * cfg.window_seconds)
    out = []
    for ts in range(begin, last_end, cfg.window_seconds):
        out.append(create_window(cfg, datetime.fromtimestamp(ts, timezone.utc),
                                 datetime.fromtimestamp(ts + cfg.window_seconds, timezone.utc)))
    return out
