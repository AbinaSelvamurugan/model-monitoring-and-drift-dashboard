# Model Monitoring and Drift Dashboard

**Model monitoring, drift detection and regression alerting**

This is an event-driven, SQLite-backed FastAPI service that associates deployed model predictions with fixed UTC monitoring windows, runs Evidently AI quality and drift reports, evaluates delayed-label performance, and raises auditable alert incidents with HTTP webhook delivery. A Streamlit/Plotly dashboard reads the persisted records. The package ships with a demonstration Iris model and a simulated traffic generator so the whole pipeline can be exercised locally. Empty states and evaluation errors are shown explicitly rather than replaced with hardcoded health scores.

## Features and logical components

1. **Prediction / label ingestion:** strict Pydantic event contract; atomic batches; duplicate ID idempotence; schema/version/range checks; separate privilege tokens; size bounds and demo per-process rate limits; optional correlation/source metadata; bounded error telemetry.
2. **Reference registry and windows:** SHA-256 immutable approved CSV versions, schema/preprocessing binding, atomic SQLite WAL writes, fixed UTC windows with empty/insufficient/pending/running/completed/failed states; the worker backfills recent closed gaps and detects interruptions.
3. **Evaluator:** Evidently 0.7 API `Report`, `DataSummaryPreset`, `DataDriftPreset(method='psi')`, raw JSON/HTML artifacts and fail-closed `ValueDrift` score extraction. Separate feature and prediction reports, typed quality checks and labeled classification/regression metrics.
4. **Alert dispatcher:** configuration-versioned rule groups, per-distinct-window debounce and recovery, event audit trail, acknowledgement/resolution/reopening, webhook delivery with 2xx-only success and retry/backoff.
5. **Dashboard:** overview, data quality, feature drift, prediction/performance, latency and errors, incidents, and downloadable evaluation reports. Charts stay empty until completed evaluations exist.

[Architecture and event-flow diagrams](docs/ARCHITECTURE.md) · [Project report outline](docs/PROJECT_REPORT_OUTLINE.md) · [Findings](docs/FINAL_FINDINGS.md)

## Repository layout

```text
modelwatch/
├── README.md
├── pyproject.toml
├── config.json / config.example.json / .env.example
├── Dockerfile / compose.yaml / Makefile
├── modelwatch/
│   ├── api.py           # health, inference, REST ingestion/query, admin
│   ├── schemas.py       # Pydantic event, label, window and admin inputs
│   ├── config.py        # typed policy and schema configuration
│   ├── events.py        # validation, idempotent persistence, label linking
│   ├── store.py         # SQLite DDL, transactions, retention, reference registry
│   ├── windows.py       # UTC windows, safe claim/refresh/backfill
│   ├── evaluate.py      # Evidently, data quality, drift and performance
│   ├── alerts.py        # incident engine, webhook with retries/audit
│   ├── admin.py         # immutable new reference approval
│   ├── model.py         # local Iris model adapter
│   ├── prepare.py       # public Iris train/stratified holdout
│   ├── worker.py        # independent periodic monitoring process
│   ├── dashboard.py     # local operator UI
│   └── cli.py           # prepare / simulated HTTP traffic
├── tests/               # synthetic fixtures, integration and transport tests
├── docs/                # architecture, findings, project outline
├── data/                # generated holdout reference, model artifact
└── reports/             # generated Evidently JSON and HTML (at runtime)
```

## Installation (Python 3.11)

```bash
cd modelwatch
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
# Set distinct, unpredictable MODELWATCH_API_TOKEN and MODELWATCH_ADMIN_TOKEN in .env
set -a; . ./.env; set +a
python -m modelwatch.cli prepare
```

The package pins `evidently==0.7.23` and other key dependencies in `pyproject.toml`.

The `prepare` command uses `sklearn.datasets.load_iris` (bundled locally by scikit-learn) and a deterministic stratified 65/35 split with `random_state=42`. It trains `Pipeline(StandardScaler(), LogisticRegression(max_iter=500))`, exports `data/model.joblib`, creates `data/reference-iris-holdout-1.csv`, computes the holdout baseline and registers a SHA-256 checksum in `state/modelwatch.db`. Repeating `prepare` is idempotent for matching data; it will not overwrite a changed reference CSV. Do not load untrusted pickle/joblib artifacts.

### Local services, in three terminals

Source `.env` in each terminal and activate `.venv`:

```bash
uvicorn modelwatch.api:app --host 127.0.0.1 --port 8000 --workers 1
python -m modelwatch.worker
streamlit run modelwatch/dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) for the read-only dashboard and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for FastAPI's OpenAPI documentation. The operator dashboard has **no login** and must not be exposed to the internet.

### Docker Compose

From the repository root:

```bash
cp .env.example .env
# Edit .env: replace both bearer tokens, configure webhook URL if required
# All .env paths are relative to WORKDIR (/app inside containers).
docker compose up --build -d
docker compose ps
docker compose logs -f api worker
docker compose down
# Reset *all* local data only when intentionally destroying demonstration state:
docker compose down -v
```

Compose runs `prepare` once before the API; `api`, `worker`, and `dashboard` share three named persistent volumes (`data`, `state`, `reports`). The API is bound to `127.0.0.1:8000` and the dashboard to `127.0.0.1:8501`, both with declared health checks. On a first run the UI shows no production observations. For real traffic volumes, use a durable distributed queue and PostgreSQL rather than scaling SQLite workers horizontally.

## Monitored event contract

```json
{
  "event_id": "unique-stable-id-123",
  "request_id": "optional-upstream-correlation-456",
  "timestamp": "2026-09-24T04:10:00+00:00",
  "model_name": "iris-logreg-demo",
  "model_version": "1.0.0",
  "schema_version": "iris-1",
  "features": {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  },
  "prediction": 0,
  "probabilities": [0.98, 0.01, 0.01],
  "latency_ms": 15.5,
  "status": "ok",
  "source": "example-schema-only",
  "segment": null
}
```

The example above is the request schema. Event and correlation IDs can be strings up to 128 chars. Required feature fields must exactly match the configured names. Numeric values must be finite numbers, declared ranges enforced, categorical values checked when constrained; successful predictions must match the task/schema. Probability vectors are optional but, when present, must sum to 1 and match the declared class order. `status='error'` must not provide a prediction. All event/label/window timestamps must be timezone-aware. Duplicate IDs with identical content return a duplicate count; conflicting duplicates return HTTP 422. Batches validate all rows before committing, in a single transaction. Old/out-of-order observations are accepted within the supported timestamp contract. The same closed window is not silently re-evaluated when late observations arrive.

### End-to-end local demonstration (synthetic monitoring traffic)

With the API running, a trained model artifact and tokens sourced from `.env`:

```bash
python -m modelwatch.cli simulate --count 120 --shift 1.5 --labels --evaluate
```

The command samples public Iris holdout records (seed 42), synthetically shifts `sepal length (cm)` by +1.5, runs the local trained adapter for each prediction, sends 120 events through `/v1/events/batch`, associates holdout labels through `/v1/labels`, creates the last closed event-time window and triggers separate drift and performance checks through FastAPI. The synthetically shifted Iris data exists to exercise the observability pipeline; the drift it produces is not a real-world regression.

Without `--labels`, performance is reported as `awaiting_ground_truth`; this is the recommended first demonstration. Without `--evaluate`, the independent worker evaluates a closed eligible window. A default configured 1-hour window must contain at least 30 events and 20 labels for the two evaluation types respectively.

### Explicit API examples

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS -H "Authorization: Bearer $MODELWATCH_API_TOKEN" http://127.0.0.1:8000/v1/models

# Local model prediction plus persistent event ingestion:
curl -fsS -X POST http://127.0.0.1:8000/v1/predict \
  -H "Authorization: Bearer $MODELWATCH_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}'

# Manual closed window (UTC); replace timestamps with the interval your events occupy:
curl -fsS -X POST http://127.0.0.1:8000/v1/windows \
  -H "Authorization: Bearer $MODELWATCH_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"start":"2026-09-24T04:00:00+00:00","end":"2026-09-24T05:00:00+00:00"}'

# Substitute the returned ID for 1:
curl -fsS -X POST -H "Authorization: Bearer $MODELWATCH_ADMIN_TOKEN" http://127.0.0.1:8000/v1/windows/1/evaluate
curl -fsS -X POST -H "Authorization: Bearer $MODELWATCH_ADMIN_TOKEN" http://127.0.0.1:8000/v1/windows/1/performance
curl -fsS -H "Authorization: Bearer $MODELWATCH_API_TOKEN" http://127.0.0.1:8000/v1/evaluations
curl -fsS -H "Authorization: Bearer $MODELWATCH_API_TOKEN" http://127.0.0.1:8000/v1/metrics/history
curl -fsS -H "Authorization: Bearer $MODELWATCH_API_TOKEN" http://127.0.0.1:8000/v1/incidents
curl -fsS -H "Authorization: Bearer $MODELWATCH_API_TOKEN" \
  http://127.0.0.1:8000/v1/evaluations/1/report/features/json -o feature-report.json
```

The timestamp example only works once matching events have been ingested and the interval has closed. For sample event ingestion and delayed labels, use `/v1/events`, `/v1/events/batch` and `/v1/labels` with the documented Pydantic schemas; the simulator calls those exact endpoints. Privileged operations require a **different admin bearer token**. For an acknowledgement or operator-provided resolution note:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/incidents/1/acknowledge \
  -H "Authorization: Bearer $MODELWATCH_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"note":"Investigating upstream input distribution"}'
```

### Principal API endpoints

| Method and endpoint | Action | Auth |
|---|---|---|
| GET `/health/live`, `/health/ready` | Process liveness and reference readiness | No |
| GET `/v1/models` | Active schema/version/baseline | Ingest |
| POST `/v1/predict` | Local Iris inference and event | Ingest |
| POST `/v1/events`, `/v1/events/batch` | Idempotent events and atomic batches | Ingest |
| POST `/v1/labels` | Link delayed label by stable ID | Ingest |
| POST `/v1/windows` | Create a closed evaluation interval | Admin |
| POST `/v1/windows/{id}/evaluate`, `/performance` | Manual drift / delayed-label checks | Admin |
| GET `/v1/windows`, `/v1/evaluations`, `/v1/evaluations/{id}` | Window and evaluation states | Ingest |
| GET `/v1/metrics/history` | Stored summary trend series | Ingest |
| GET `/v1/evaluations/{id}/report/{kind}/{format}` | Export quality/features/prediction/scores JSON or HTML | Ingest |
| GET `/v1/incidents`, `/v1/incidents/{id}/history` | Incidents, audit and delivery attempts | Ingest |
| POST `/v1/incidents/{id}/acknowledge`, `/resolve` | State transition with note | Admin |
| POST `/v1/notifications/dispatch` | Webhook dispatch/retry | Admin |
| GET `/v1/references` | Approved versions and baseline metadata | Ingest |
| POST `/v1/references/approve` | Approve another locally staged CSV under new version | Admin |

Errors use HTTP 401 for denied tokens, 422 for event contract errors, 404 for missing IDs, 409 for incompatible configuration/state, 413 for oversized requests, 429 for per-process rate limits, 503 for missing demo model/reference readiness. Evaluation computation errors are persisted with `status=failed` and an error string.

## Configuration and approved references

Edit `config.json` before preparation and restart all services after changing it. Schema contract fields include `feature_types`, `numeric_ranges`, `categorical_values`, `classes`, `allow_missing`, `schema_version`, `preprocessing_version`, `task` and `reference_version`. Monitoring parameters include `window_seconds`, `min_events`, `min_labels`, `drift_method`, `drift_threshold`, `drift_share_threshold`, `max_missing_rates`, `error_rate_limit`, `latency_limit_ms`, `ingestion_failure_limit`, `requests_per_minute`, and retention and alert settings. `config.example.json` mirrors the default demonstration configuration.

PSI is used as a reproducible demonstration comparison for both numerical and categorical distributions. The default threshold `0.2`, share rule `0.5`, performance accuracy drop `0.1`, persistence `2` and recovery `2` are explicit engineering defaults that require calibration on real production reference data. Evidently reports measure distributional changes, not model quality deterioration on their own.

To register another reference, stage a CSV in configured `data/` with all feature columns, `prediction` and `target`, and call:

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/references/approve \
  -H "Authorization: Bearer $MODELWATCH_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"version":"iris-holdout-2","source_file":"staged-iris.csv","reason":"Approved controlled evaluation population"}'
```

The service computes baseline metrics from that CSV, copies it to an immutable versioned filename and saves its SHA-256. It does not automatically make it active. A controlled operator must update `reference_version` and restart/redeploy the services, after checking model/schema/preprocessing compatibility. Reference records cannot overwrite existing versions. The demo `prepare` command only knows the Iris schema/model; integrating your own model requires preparing and approving an appropriate reference and providing an inference adapter or emitting events from your existing deployed endpoint.

## Integrating a real model

- Emit a prediction event from your inference service with stable `event_id`, exact `model_name`, `model_version`, `schema_version`, typed input feature fields, prediction, optional probabilities, inference latency and status. Never put raw PII or credentials into `source`/`segment` metadata.
- Create a new `config.json` schema and versioned reference with the *same preprocessing path, units, population, and model version* used in production. Keep its approval evidence. Use stable output/class mappings. Regressors must use numeric predictions and targets; classifiers must declare class order.
- Connect ground truth later via `/v1/labels`. All metrics use only the labels linked through that endpoint, and the UI shows label-coverage fraction. Watch for delayed/selected labels bias before drawing quality conclusions.
- Replace the Iris-only `modelwatch/model.py` adapter for `/v1/predict`, or do not use this endpoint at all: monitoring ingestion is model-agnostic within its configured task contract.
- Production-volume integrations require stronger operational controls: source-side request budgets, retry queues, authentication/key rotation, TLS, PostgreSQL, retention/erasure, deployment health instrumentation and a calibrated testing policy.

## Continuous monitoring, alert lifecycle and delivery

The independent worker wakes every `MODELWATCH_POLL_SECONDS` (default 60), creates recent fixed-duration closed windows, updates empty/underfilled statuses, runs eligible drift evaluations, evaluates labeled metrics when enough labels arrive, and independently evaluates availability/ingestion rejection evidence. It dispatches unresolved incident notifications. A worker restart flags abandoned `running` evaluations as failed. The same window is never implicitly run twice, even after a failed evaluation; an explicit, audited rerun feature is future work.

A rule groups by model, version, immutable reference and versioned policy hash. Consecutive **distinct** failed windows trigger a new incident; a subsequent failure updates its last evidence; acknowledgment requires operator action; consecutive successful windows resolve it; recurrence reopens it. Silence, an underfilled window or a failed Evidently report cannot resolve an incident. Webhook 2xx means `delivered`; 503/timeout means `failed` with recorded backoff; blank URL means `unconfigured`. The dispatcher marks a delivery successful only on an HTTP 2xx response.

The dashboard can investigate completed evaluation summaries, raw Evidently reports and the persisted alert audit timeline. Its metrics are stored in SQLite and artifacts on a shared volume, surviving process restarts. Inference latency from events is not the same as API response latency; FastAPI exposes `X-API-Response-Ms` on requests but does not persist a separate response-time series.

## Testing and QA

```bash
python -m compileall -q modelwatch
python -m pytest -q
ruff check .
ruff format --check .
```

Tests cover schema/type mismatches, duplicate IDs, out-of-order events, atomic batches, delayed/conflicting labels, insufficient data, a synthetic substitute for Evidently snapshot extraction, failure persistence, binary/multiclass/regression performance, per-window debounce, incident recovery/reopen/audit, and HTTP 204/503 webhook responses. Further detail is in [docs/FINAL_FINDINGS.md](docs/FINAL_FINDINGS.md).

The dashboard has no login and SQLite suits modest volumes; larger deployments need access controls and PostgreSQL.

The supported operational model intentionally separates input data drift, prediction distribution drift, labeled performance degradation, possible concept drift, and infrastructure failures. None of them is automatically inferred from the others.

### Rule-level override example

The optional `alert_rules` object supports `enabled`, `severity`, `min_count`, `persistence`, `recovery`, `cooldown_seconds` and `destination_env` per rule. For example, add this to the top level of `config.json`:

```json
"alert_rules": {
  "prediction_drift": {
    "enabled": true, "severity": "warning", "min_count": 100,
    "persistence": 3, "recovery": 2, "cooldown_seconds": 7200,
    "destination_env": "MODELWATCH_WEBHOOK_URL"
  }
}
```

This is a JSON *fragment*, not a complete independent config file. Rule thresholds remain configured through the relevant metric fields. Every policy change creates a different hash/version in incident grouping. Notification destinations are operator-controlled environment variable names, not caller-supplied URLs; the dispatcher reads the corresponding URL when sending. With no URL configured, the incident's delivery state is `unconfigured`, not successful.

For version-interface verification after dependency installation:

```bash
python scripts/verify_evidently.py
```

The script runs `DataDriftPreset` against disjoint offline public Iris test subsets, verifies Evidently JSON/HTML export and `ValueDrift` score extraction, and prints the observed values. It works on offline test data and does not touch monitoring storage. The included GitHub Actions workflow runs this smoke test on Python 3.11.

## License and attribution

No license is imposed by this repository; add one before granting reuse rights. ModelWatch depends on Evidently AI, scikit-learn, Pandas, FastAPI, Streamlit and Plotly, each distributed under its own license, and the Iris dataset, which comes from scikit-learn.
