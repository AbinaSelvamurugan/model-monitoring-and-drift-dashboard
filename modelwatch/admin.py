"""Controlled new baseline approval. Approval never silently changes active config."""
import shutil

import pandas as pd

from modelwatch.config import Config, reference_dir
from modelwatch.evaluate import compute_performance
from modelwatch.store import register_reference


def approve_reference(cfg: Config, version: str, source_file: str, reason: str):
    if version == cfg.reference_version:
        raise ValueError("active reference version cannot be overwritten")
    root = reference_dir().resolve()
    incoming = root / source_file  # Pydantic enforces filename-only expression.
    if not incoming.is_file() or incoming.is_symlink() or incoming.resolve().parent != root:
        raise ValueError("source must be a regular CSV in approved data directory")
    frame = pd.read_csv(incoming)
    required = set(cfg.feature_types) | {"prediction", "target"}
    if not required.issubset(frame.columns) or len(frame) < cfg.min_events:
        raise ValueError("reference requires all schema columns, prediction, target and minimum rows")
    for name, kind in cfg.feature_types.items():
        if kind == "numeric":
            if not pd.api.types.is_numeric_dtype(frame[name]) or not frame[name].notna().all():
                raise ValueError(f"invalid numerical reference: {name}")
        elif not frame[name].dropna().map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"invalid categorical reference: {name}")
    labeled = [{"label": row["target"], "prediction": row["prediction"],
                "probabilities": None} for _, row in frame.iterrows()]
    baseline = compute_performance(cfg, labeled)
    baseline.update({"count": len(frame), "population": "operator approved reference",
                     "approval_reason": reason})
    destination = root / f"reference-{version}.csv"
    if destination.exists():
        raise ValueError("reference version exists and cannot be overwritten")
    shutil.copyfile(incoming, destination)
    return register_reference(version, destination, cfg, reason, baseline)
