"""Reference training and optional explicitly simulated HTTP traffic."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from uuid import uuid4

import pandas as pd

from modelwatch.config import load_config
from modelwatch.model import infer
from modelwatch.prepare import prepare


def post(url, token, payload):
    request = Request(url, data=json.dumps(payload).encode(), method="POST",
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def demo(count, shift, labels, evaluate, url):
    """Original public Iris holdout data, explicitly transformed to SIMULATED traffic."""
    cfg = load_config()
    frame = pd.read_csv(f"{os.getenv('MODELWATCH_REFERENCE_DIR', 'data')}/reference-{cfg.reference_version}.csv")
    reference = frame.sample(n=count, replace=count > len(frame), random_state=42).reset_index(drop=True)
    end_ts = int(datetime.now(timezone.utc).timestamp()) // cfg.window_seconds * cfg.window_seconds
    start = datetime.fromtimestamp(end_ts - cfg.window_seconds, timezone.utc)
    end = datetime.fromtimestamp(end_ts, timezone.utc)
    batch_id = str(uuid4())
    batch, delayed = [], []
    for index, row in reference.iterrows():
        features = {col: float(row[col]) if kind == "numeric" else str(row[col])
                    for col, kind in cfg.feature_types.items()}
        first_numeric = next((col for col, t in cfg.feature_types.items() if t == "numeric"), None)
        if first_numeric:
            features[first_numeric] += shift
        prediction, probabilities = infer(features, cfg)
        event_id = f"simulated-{batch_id}-{index}"
        timestamp = start + timedelta(seconds=(index + 1) * cfg.window_seconds / (count + 1))
        batch.append({"event_id": event_id, "timestamp": timestamp.isoformat(),
                      "model_name": cfg.model_name, "model_version": cfg.model_version,
                      "schema_version": cfg.schema_version, "features": features,
                      "prediction": prediction, "probabilities": probabilities,
                      "status": "ok", "source": "SIMULATED_Iris_traffic"})
        if labels:
            delayed.append({"event_id": event_id, "label": int(row["target"]),
                            "labeled_at": datetime.now(timezone.utc).isoformat()})
    token = os.environ["MODELWATCH_API_TOKEN"]
    print("SIMULATED TRAFFIC, not a production model workload")
    for offset in range(0, len(batch), cfg.max_batch_events):
        print(post(url+"/v1/events/batch", token, {"events": batch[offset:offset+cfg.max_batch_events]}))
    for item in delayed:
        post(url+"/v1/labels", token, item)
    print("window", start.isoformat(), end.isoformat())
    if evaluate:
        admin = os.environ["MODELWATCH_ADMIN_TOKEN"]
        win = post(url+"/v1/windows", admin, {"start": start.isoformat(), "end": end.isoformat()})
        print("evaluation", post(url+f"/v1/windows/{win['id']}/evaluate", admin, {}))
        if labels:
            print("performance", post(url+f"/v1/windows/{win['id']}/performance", admin, {}))


def main():
    parser = argparse.ArgumentParser(prog="modelwatch")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    d = sub.add_parser("simulate", help="explicitly simulated monitoring traffic sent through FastAPI")
    d.add_argument("--count", type=int, default=120)
    d.add_argument("--shift", type=float, default=0)
    d.add_argument("--labels", action="store_true")
    d.add_argument("--evaluate", action="store_true")
    d.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(load_config()), indent=2))
    else:
        if args.count < 1:
            parser.error("count must be positive")
        demo(args.count, args.shift, args.labels, args.evaluate, args.url)


if __name__ == "__main__":
    main()
