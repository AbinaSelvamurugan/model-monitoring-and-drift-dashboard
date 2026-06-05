"""Read-only local SQLite operator dashboard."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modelwatch.config import load_config
from modelwatch.store import init_db, many

st.set_page_config(page_title="ModelWatch", page_icon="📈", layout="wide")
init_db()
cfg = load_config()
st.title("ModelWatch | ML Observability")
st.caption("All values derive from stored events and completed evaluations. Iris demo data is simulated monitoring traffic.")
windows = many("SELECT * FROM windows ORDER BY id DESC LIMIT 500")
evals = many("""SELECT e.*,w.start,w.end,w.model_version,w.reference_version FROM evaluations e
                    JOIN windows w ON w.id=e.window_id ORDER BY e.id DESC LIMIT 500""")
incidents = many("SELECT * FROM incidents ORDER BY id DESC LIMIT 500")
errors = many("SELECT * FROM ingest_errors ORDER BY id DESC LIMIT 100")
complete = [e for e in evals if e["status"] == "completed"]
drift = [e for e in complete if e["kind"] == "drift"]
performance = [e for e in complete if e["kind"] == "performance"]


def summary(e):
    return json.loads(e["summary_json"]) if e and e["summary_json"] else {}


def plot_history(rows, get_value, label):
    values = []
    for e in reversed(rows):
        v = get_value(summary(e))
        if v is not None:
            values.append({"window_end": e["end"], label: v, "evaluation": e["id"]})
    if values:
        st.plotly_chart(px.line(pd.DataFrame(values), x="window_end", y=label,
                                hover_data=["evaluation"], markers=True), use_container_width=True)
    else:
        st.info("No completed eligible evaluations for this chart.")


pages = st.tabs(["Overview", "Data quality", "Feature drift", "Predictions & performance",
                 "Latency & health", "Alerts & incidents", "Reports"])
with pages[0]:
    st.subheader("Current monitoring state")
    cols = st.columns(4)
    cols[0].metric("Model", cfg.model_name)
    cols[1].metric("Version", cfg.model_version)
    cols[2].metric("Completed windows", len(drift))
    cols[3].metric("Unresolved incidents", sum(i["state"] != "resolved" for i in incidents))
    st.write("Approved reference:", cfg.reference_version)
    if not windows:
        st.info("No observation windows yet. Prepare a reference, ingest events and close a window.")
    else:
        st.dataframe(pd.DataFrame(windows)[["id", "start", "end", "event_count", "status", "eligible"]],
                     hide_index=True, use_container_width=True)
    if drift:
        latest = summary(drift[0])
        st.write("Latest successful evaluation:", drift[0]["end"])
        st.write("Feature drift share:", latest.get("feature_drift_share"))
        st.write("Prediction drift:", latest.get("prediction", {}).get("drifted"))
    else:
        st.warning("No completed drift evaluation. Status is unknown, not healthy.")
    st.write("Ground-truth performance:", "available" if performance else "awaiting ground truth")
with pages[1]:
    st.subheader("Completeness and data quality")
    if drift:
        quality = summary(drift[0])["quality"]
        st.dataframe(pd.DataFrame(quality["tests"]), hide_index=True)
        st.write("Evidently quality test results", json.loads(drift[0]["tests_json"]).get("evidently_quality", []))
        plot_history(drift, lambda s: max(s.get("quality", {}).get("missing_rates", {}).values(), default=None),
                     "Max missing-feature fraction")
    else:
        st.info("No completed quality check.")
    st.write("Rejected ingestion requests (metadata only):", len(errors))
    if errors:
        st.dataframe(pd.DataFrame(errors), hide_index=True)
with pages[2]:
    st.subheader("Reference vs observed feature drift")
    if drift:
        e = drift[0]
        s = summary(e)
        st.caption(f"Reference {e['reference_version']}; UTC [{e['start']}, {e['end']}); method {s['method']}; threshold {s['threshold']}")
        st.dataframe(pd.DataFrame.from_dict(s["features"], orient="index"), use_container_width=True)
        plot_history(drift, lambda x: x.get("feature_drift_share"), "Drifted feature share")
        st.caption("Open the original Evidently report in Reports for reference/current distributions.")
    else:
        st.info("No completed reference/current feature comparison.")
with pages[3]:
    st.subheader("Prediction distribution and labeled model quality")
    if drift:
        st.write("Latest prediction drift", summary(drift[0]).get("prediction"))
        proportions = summary(drift[0]).get("prediction_shares")
        if proportions:
            st.write("Observed vs reference class proportions", proportions)
        st.write("Probability-score drift status", summary(drift[0]).get("probability_score_status"))
        if summary(drift[0]).get("probability_scores"):
            st.dataframe(pd.DataFrame.from_dict(summary(drift[0])["probability_scores"], orient="index"))
        plot_history(drift, lambda x: x.get("prediction", {}).get("score"), "Prediction drift score")
    if performance:
        p = summary(performance[0])
        st.write("Latest labeled subset size", p["labeled_count"], "of", p["total_event_count"])
        st.write("Observed metrics", p["performance"])
        st.write("Approved baseline", p["baseline"])
        metric = "accuracy" if cfg.task == "classification" else "rmse"
        plot_history(performance, lambda s: s.get("performance", {}).get(metric), metric)
    else:
        st.info("Awaiting sufficient linked ground truth; no accuracy/error rate is inferred from drift.")
with pages[4]:
    st.subheader("Inference service health (from observed prediction events)")
    if drift:
        plot_history(drift, lambda x: x.get("quality", {}).get("error_rate"), "Inference error rate")
        plot_history(drift, lambda x: x.get("quality", {}).get("latency_p95_ms"), "Inference p95 latency (ms)")
        plot_history(drift, lambda x: x.get("quality", {}).get("throughput_per_second"),
                     "Observed throughput (events/sec)")
    else:
        st.info("No instrumented window to chart.")
    st.caption("Inference latency excludes API overhead; that overhead is returned as X-API-Response-Ms and is not retained as a time series.")
with pages[5]:
    st.subheader("Incident register")
    if incidents:
        st.dataframe(pd.DataFrame(incidents), hide_index=True, use_container_width=True)
        selected = st.selectbox("Incident ID", [x["id"] for x in incidents])
        events = many("SELECT * FROM incident_events WHERE incident_id=? ORDER BY id", (selected,))
        deliveries = many("SELECT * FROM delivery_attempts WHERE incident_id=? ORDER BY id", (selected,))
        st.write("Audit trail")
        st.dataframe(pd.DataFrame(events), use_container_width=True)
        st.write("Delivery attempts")
        if deliveries:
            st.dataframe(pd.DataFrame(deliveries), use_container_width=True)
        else:
            st.info("No attempted delivery; never assume delivered.")
    else:
        st.info("No incidents recorded. Without evaluations this does not imply health.")
with pages[6]:
    st.subheader("Persisted Evidently evaluations")
    if evals:
        st.dataframe(pd.DataFrame(evals)[["id", "window_id", "kind", "status", "error",
                                         "artifact_json", "artifact_html"]], use_container_width=True)
        chosen = st.selectbox("Evaluation", [e["id"] for e in evals])
        record = next(e for e in evals if e["id"] == chosen)
        if record["summary_json"]:
            st.json(json.loads(record["summary_json"]))
        for format in ("json", "html"):
            path = record.get(f"artifact_{format}")
            if path and Path(path).is_file():
                st.download_button(f"Download {format.upper()} report", data=Path(path).read_bytes(),
                                   file_name=f"evaluation-{chosen}.{format}",
                                   mime="application/json" if format == "json" else "text/html")
        if record["error"]:
            st.error(record["error"])
    else:
        st.info("No evaluation artifacts recorded.")
