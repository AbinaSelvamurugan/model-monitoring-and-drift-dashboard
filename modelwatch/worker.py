"""Separate restart-safe evaluation process; no drift computation in ingestion path."""
import logging
import os
import signal
import time

from modelwatch.alerts import dispatch
from modelwatch.config import load_config
from modelwatch.evaluate import evaluate_availability, evaluate_performance, evaluate_window
from modelwatch.store import flag_interrupted, init_db, many, purge_expired
from modelwatch.windows import backfill_windows, refresh_eligibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
stop = False


def shutdown(_sig, _frame):
    global stop
    stop = True


def tick():
    c = load_config()
    init_db()
    purge_expired(c.retention_days)
    for win in backfill_windows(c):
        evaluate_availability(c, win["id"])
    for win in many("SELECT * FROM windows WHERE status IN ('pending','empty','insufficient') ORDER BY id LIMIT 50"):
        updated = refresh_eligibility(c, win["id"])
        if updated["status"] == "pending":
            logging.info("drift window=%s", win["id"])
            evaluate_window(c, win["id"])
    # A late label is evaluated at a new label_count; unchanged label-count is idempotent.
    for win in many("SELECT * FROM windows WHERE status='completed' ORDER BY id DESC LIMIT 100"):
        evaluate_performance(c, win["id"])
    dispatch(c)


def main():
    init_db()
    abandoned = flag_interrupted()
    if abandoned:
        logging.warning("marked %s interrupted claims as failed (operator review)", abandoned)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while not stop:
        try:
            tick()
        except Exception:
            logging.exception("worker tick failed, will retry")
        for _ in range(int(os.getenv("MODELWATCH_POLL_SECONDS", "60"))):
            if stop:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
