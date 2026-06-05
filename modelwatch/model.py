"""Iris reference adapter; replace load_model/predict for other models."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelwatch.config import Config


def model_path():
    return Path(os.getenv("MODELWATCH_MODEL", "data/model.joblib"))


def infer(features: dict, cfg: Config):
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError("model artifact not prepared")
    model = joblib.load(path)
    frame = pd.DataFrame([features])[list(cfg.feature_types)]
    pred = model.predict(frame)[0]
    pred = pred.item() if isinstance(pred, np.generic) else pred
    probability = model.predict_proba(frame)[0].tolist() if hasattr(model, "predict_proba") else None
    return pred, probability
