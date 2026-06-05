"""Prepare the public Iris training/holdout split for the demonstration model."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelwatch.config import Config, reference_dir
from modelwatch.model import model_path
from modelwatch.store import init_db, register_reference


def prepare(cfg: Config):
    if cfg.model_name != "iris-logreg-demo" or cfg.task != "classification":
        raise ValueError("prepare is only for the documented Iris example")
    iris = load_iris(as_frame=True)
    X = iris.data[list(cfg.feature_types)]
    y = iris.target
    X_train, X_ref, y_train, y_ref = train_test_split(
        X, y, stratify=y, test_size=0.35, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=500, random_state=42))])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_ref)
    baseline = {"accuracy": float(accuracy_score(y_ref, predictions)),
                "f1_macro": float(f1_score(y_ref, predictions, average="macro")),
                "count": len(y_ref), "seed": 42, "population": "stratified iris holdout"}
    frame = X_ref.copy()
    frame["prediction"] = predictions
    for index, values in enumerate(pipeline.predict_proba(X_ref).T):
        frame[f"score_{index}"] = values
    frame["target"] = y_ref.to_numpy()
    data_dir = reference_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"reference-{cfg.reference_version}.csv"
    if path.exists():
        old = pd.read_csv(path)
        if not old.equals(pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)))):
            raise ValueError("existing reference differs; new version required")
    else:
        frame.to_csv(path, index=False)
    m = model_path()
    m.parent.mkdir(parents=True, exist_ok=True)
    if not m.exists():
        joblib.dump(pipeline, m)
    init_db()
    record = register_reference(cfg.reference_version, path, cfg, "documented stratified Iris split seed 42", baseline)
    return {"reference": record["version"], "rows": len(frame), "baseline": baseline}
