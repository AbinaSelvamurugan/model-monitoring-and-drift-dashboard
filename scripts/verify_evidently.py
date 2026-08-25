"""Evidently 0.7 integration smoke test on offline test data (not production traffic).

Run after installing the project dependencies.
Failure is nonzero and MUST NOT be interpreted as 'no drift'.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from modelwatch.config import load_config
from modelwatch.evaluate import evidently_report


def main():
    import evidently

    c = load_config()
    expected = "0.7.23"
    if evidently.__version__ != expected:
        raise RuntimeError(f"Expected evidently=={expected}, got {evidently.__version__}")
    data = load_iris(as_frame=True).data[list(c.feature_types)]
    # Public rows, deliberately disjoint slices; TEST/SMOKE ONLY.
    ref = data.iloc[:75].reset_index(drop=True).copy()
    curr = data.iloc[75:].reset_index(drop=True).copy()
    with tempfile.TemporaryDirectory(prefix="modelwatch-evidently-smoke-") as folder:
        result, scores = evidently_report(curr, ref, list(c.feature_types), c,
                                         Path(folder) / "offline-evidently-check")
        assert set(scores) == set(c.feature_types)
        assert (Path(folder) / "offline-evidently-check.json").is_file()
        assert (Path(folder) / "offline-evidently-check.html").is_file()
        print(json.dumps({"data_class": "OFFLINE_REFERENCE_SMOKE_NOT_PRODUCTION",
                          "evidently_version": evidently.__version__,
                          "reference_count": len(ref), "current_count": len(curr),
                          "feature_results": scores,
                          "test_count": len(result.get("tests", []))}, indent=2))


if __name__ == "__main__":
    main()
