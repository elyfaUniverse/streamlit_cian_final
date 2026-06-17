from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "eda_result.csv",
    "roman_xgboost_pipeline.pkl",
    "roman_xgboost_pipeline_metadata.json",
    "model_metrics.json",
    "split_data.pkl",
]

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "joblib",
    "plotly",
    "folium",
    "streamlit-folium",
]


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))

    for package in REQUIRED_PACKAGES:
        metadata.version(package)

    data = pd.read_csv(ROOT / "eda_result.csv", nrows=5)
    required_columns = {"price", "area_total", "region", "district", "nearest_metro"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise SystemExit("eda_result.csv missing columns: " + ", ".join(missing_columns))

    model = joblib.load(ROOT / "roman_xgboost_pipeline.pkl")
    if not hasattr(model, "predict"):
        raise SystemExit("roman_xgboost_pipeline.pkl does not look like a model")

    json.loads((ROOT / "roman_xgboost_pipeline_metadata.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))
    joblib.load(ROOT / "split_data.pkl")

    print("Bundle check passed.")


if __name__ == "__main__":
    main()
