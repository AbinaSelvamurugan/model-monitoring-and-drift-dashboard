"""FastAPI endpoints; privileged calls never execute heavy drift checks in ingestion."""
import logging
import threading
from contextlib import asynccontextmanager
import os
import secrets
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

from modelwatch.admin import approve_reference
from modelwatch.alerts import change_incident, dispatch
from modelwatch.config import Config, load_config, report_dir
from modelwatch.evaluate import evaluate_performance, evaluate_window
from modelwatch.events import ContractError, add_events, add_label, validate_event
from modelwatch.model import infer, model_path
from modelwatch.schemas import (EventBatch, GroundTruth, IncidentChange, PredictionEvent,
                                ReferenceApproval, WindowRequest)
from modelwatch.store import connect, current_reference, init_db, many, one, utcnow
from modelwatch.windows import create_window

log = logging.getLogger(__name__)
_rate_lock = threading.Lock()
_rate_buckets = {}

@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="ModelWatch", version="0.1.0", lifespan=lifespan)


def cfg() -> Config:
    return load_config()


def require_token(token: str | None, name: str):
    expected = os.getenv(name, "")
    if not expected or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "valid bearer token required")


def ingest_auth(authorization: str | None = Header(default=None)):
    require_token(authorization.removeprefix("Bearer ") if authorization else None,
                  "MODELWATCH_API_TOKEN")


def admin_auth(authorization: str | None = Header(default=None)):
    require_token(authorization.removeprefix("Bearer ") if authorization else None,
                  "MODELWATCH_ADMIN_TOKEN")


@app.middleware("http")
async def limitations(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    config = cfg()
    limit = config.max_body_bytes
    # Demo/local per-process rate limit. For multi-replica deployments use a shared gateway.
    peer = request.client.host if request.client else "unknown"
    period = int(time.time() // 60)
    with _rate_lock:
        for k in list(_rate_buckets):
            if k[1] != period:
                del _rate_buckets[k]
        bucket = (peer, period)
        count = _rate_buckets.get(bucket, 0) + 1
        _rate_buckets[bucket] = count
    if count > config.requests_per_minute:
        return JSONResponse(status_code=429, content={"detail": "request rate limit exceeded"})
    length = request.headers.get("content-length")
    if length and length.isdecimal() and int(length) > limit:
        return JSONResponse(status_code=413, content={"detail": "body exceeds byte limit"})
    # Bounded read rejects chunked/streamed payloads too; re-inject consumed bytes.
    body = await request.body()
    if len(body) > limit:
        return JSONResponse(status_code=413, content={"detail": "body exceeds byte limit"})
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    request._receive = receive
    started = time.perf_counter()
    result = await call_next(request)
    result.headers["X-API-Response-Ms"] = str(round((time.perf_counter() - started) * 1000, 3))
    if request.url.path.startswith("/v1/events") and result.status_code >= 400:
        with connect(write=True) as db:
            db.execute("INSERT INTO ingest_errors(ts,status,reason) VALUES(?,?,?)",
                       (utcnow(), result.status_code, "request rejected"))
    log.info("method=%s path=%s status=%s", request.method, request.url.path, result.status_code)
    return result


@app.exception_handler(ContractError)
def bad_contract(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def bad_config(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    try:
        init_db()
        ref = current_reference(cfg())
        if cfg().model_name == "iris-logreg-demo" and not model_path().exists():
            raise HTTPException(503, "demonstration model is missing")
        return {"status": "ready" if model_path().exists() else "monitor_only", "reference_version": ref["version"]}
    except (ValueError, OSError) as exc:
        raise HTTPException(503, "reference not prepared or inaccessible") from exc


@app.get("/v1/models", dependencies=[Depends(ingest_auth)])
def model_config():
    c = cfg()
    ref = current_reference(c)
    return {"model_name": c.model_name, "model_version": c.model_version, "task": c.task,
            "schema_version": c.schema_version, "feature_types": c.feature_types,
            "reference_version": ref["version"], "approved_at": ref["approved_at"],
            "baseline": __import__("json").loads(ref["baseline_json"])}


@app.post("/v1/events", status_code=201, dependencies=[Depends(ingest_auth)])
def ingest(ev: PredictionEvent):
    result = add_events([ev], cfg())
    return result


@app.post("/v1/events/batch", status_code=201, dependencies=[Depends(ingest_auth)])
def batch(batch: EventBatch):
    return add_events(batch.events, cfg())


@app.post("/v1/labels", dependencies=[Depends(ingest_auth)])
def labels(label: GroundTruth):
    try:
        return add_label(label, cfg())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/v1/predict", status_code=201, dependencies=[Depends(ingest_auth)])
def predict(features: dict[str, float | str]):
    c = cfg()
    if c.model_name != "iris-logreg-demo":
        raise HTTPException(501, "replace the inference adapter for an external model")
    now = datetime.now(timezone.utc)
    ev = PredictionEvent(event_id=str(uuid4()), timestamp=now, model_name=c.model_name,
                         model_version=c.model_version, schema_version=c.schema_version,
                         features=features, prediction=0)
    validate_event(ev, c)
    started = time.perf_counter()
    try:
        prediction, probabilities = infer(features, c)
    except FileNotFoundError as exc:
        raise HTTPException(503, "model artifact unavailable") from exc
    ev.prediction = prediction
    ev.probabilities = probabilities
    ev.latency_ms = round((time.perf_counter() - started) * 1000, 4)
    ev.source = "local-reference-inference"
    add_events([ev], c)
    return {"event_id": ev.event_id, "prediction": prediction,
            "probabilities": probabilities, "latency_ms": ev.latency_ms,
            "model_version": c.model_version}


@app.post("/v1/windows", dependencies=[Depends(admin_auth)])
def create(req: WindowRequest):
    return create_window(cfg(), req.start, req.end)


@app.post("/v1/windows/{window_id}/evaluate", dependencies=[Depends(admin_auth)])
def evaluate(window_id: int):
    # Admin-only manual trigger; production ingestion must use worker.
    if not one("SELECT id FROM windows WHERE id=?", (window_id,)):
        raise HTTPException(404, "window not found")
    return evaluate_window(cfg(), window_id)


@app.post("/v1/windows/{window_id}/performance", dependencies=[Depends(admin_auth)])
def performance(window_id: int):
    try:
        return evaluate_performance(cfg(), window_id)
    except KeyError as exc:
        raise HTTPException(404, "window not found") from exc


@app.get("/v1/windows", dependencies=[Depends(ingest_auth)])
def windows(limit: int = 100):
    return many("SELECT * FROM windows ORDER BY id DESC LIMIT ?", (max(1, min(500, limit)),))


@app.get("/v1/evaluations", dependencies=[Depends(ingest_auth)])
def evaluations(limit: int = 100):
    return many("SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (max(1, min(500, limit)),))


@app.get("/v1/evaluations/{evaluation_id}", dependencies=[Depends(ingest_auth)])
def evaluation(evaluation_id: int):
    e = one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
    if not e:
        raise HTTPException(404, "evaluation not found")
    return e


@app.get("/v1/evaluations/{evaluation_id}/report/{format}", dependencies=[Depends(ingest_auth)])
def report(evaluation_id: int, format: str):
    if format not in {"json", "html"}:
        raise HTTPException(400, "format must be json or html")
    e = one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
    if not e or not e[f"artifact_{format}"]:
        raise HTTPException(404, "report unavailable")
    # File path is DB-managed and never supplied by the caller.
    return FileResponse(e[f"artifact_{format}"], filename=f"evaluation-{evaluation_id}.{format}")


@app.get("/v1/metrics/history", dependencies=[Depends(ingest_auth)])
def history(limit: int = 100):
    return many("""SELECT e.id,e.kind,e.status,e.summary_json,e.created_at,w.start,w.end,
                   w.reference_version,w.model_version FROM evaluations e JOIN windows w ON
                   w.id=e.window_id ORDER BY e.id DESC LIMIT ?""", (max(1, min(500, limit)),))


@app.get("/v1/incidents", dependencies=[Depends(ingest_auth)])
def incidents():
    return many("SELECT * FROM incidents ORDER BY id DESC LIMIT 500")


@app.get("/v1/incidents/{incident_id}/history", dependencies=[Depends(ingest_auth)])
def incident_history(incident_id: int):
    return {"events": many("SELECT * FROM incident_events WHERE incident_id=? ORDER BY id",
                           (incident_id,)),
            "deliveries": many("SELECT * FROM delivery_attempts WHERE incident_id=? ORDER BY id",
                               (incident_id,))}


@app.post("/v1/incidents/{incident_id}/acknowledge", dependencies=[Depends(admin_auth)])
def acknowledge(incident_id: int, request: IncidentChange):
    try:
        return change_incident(incident_id, "acknowledge", request.note)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/v1/incidents/{incident_id}/resolve", dependencies=[Depends(admin_auth)])
def resolve(incident_id: int, request: IncidentChange):
    try:
        return change_incident(incident_id, "resolve", request.note)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/v1/notifications/dispatch", dependencies=[Depends(admin_auth)])
def notifications():
    return dispatch(cfg())


@app.get("/v1/references", dependencies=[Depends(ingest_auth)])
def references():
    return many("""SELECT version,model_name,model_version,schema_version,preprocessing_version,
                      sha256,approved_at,approval_reason,baseline_json FROM references_catalog ORDER BY approved_at DESC""")


@app.get("/v1/evaluations/{evaluation_id}/report/{kind}/{format}", dependencies=[Depends(ingest_auth)])
def report_type(evaluation_id: int, kind: str, format: str):
    if format not in {"json", "html"} or kind not in {"quality", "features", "prediction", "scores"}:
        raise HTTPException(400, "invalid report kind or format")
    e = one("SELECT * FROM evaluations WHERE id=?", (evaluation_id,))
    if not e or e["status"] != "completed" or e["kind"] != "drift":
        raise HTTPException(404, "completed drift evaluation not found")
    root = report_dir() / f"evaluation-{evaluation_id}-{kind}.{format}"
    if not root.is_file():
        raise HTTPException(404, "report artifact not available")
    return FileResponse(root, filename=f"evaluation-{evaluation_id}-{kind}.{format}")


@app.post("/v1/references/approve", dependencies=[Depends(admin_auth)], status_code=201)
def approve(data: ReferenceApproval):
    approved = approve_reference(cfg(), data.version, data.source_file, data.reason)
    return {k: approved[k] for k in ("version", "sha256", "approved_at", "approval_reason")}
