"""Validate the raw Earth Engine data and create a transparent risk baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data/raw/urban_heat_coimbatore_2022_2025.csv"
PROCESSED_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
STATISTICS_PATH = ROOT_DIR / "data/processed/dataset_statistics.json"
METADATA_PATH = ROOT_DIR / "data/processed/baseline_risk_metadata.json"

REQUIRED_COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
    "lst_median_c",
    "lst_p90_c",
    "ndvi_median",
    "ndbi_median",
    "valid_pixel_count",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
BASELINE_FEATURES = ["lst_median_c", "ndbi_median", "ndvi_median"]


def validate_input(frame: pd.DataFrame) -> dict:
    """Report data quality without dropping or modifying any raw records."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    non_finite = {
        column: int((~np.isfinite(frame[column])).sum()) for column in NUMERIC_COLUMNS
    }
    report = {
        "row_count": int(len(frame)),
        "duplicate_cell_ids": int(frame["cell_id"].duplicated().sum()),
        "missing_values": {
            column: int(value) for column, value in frame[REQUIRED_COLUMNS].isna().sum().items()
        },
        "non_finite_values": non_finite,
        "invalid_ndvi_count": int(((frame["ndvi_median"] < -1) | (frame["ndvi_median"] > 1)).sum()),
        "invalid_ndbi_count": int(((frame["ndbi_median"] < -1) | (frame["ndbi_median"] > 1)).sum()),
        "suspicious_lst_count": int(
            ((frame["lst_median_c"] < -50) | (frame["lst_median_c"] > 70)).sum()
            + ((frame["lst_p90_c"] < -50) | (frame["lst_p90_c"] > 80)).sum()
        ),
        "numeric_ranges": {
            column: {
                "minimum": float(frame[column].min()),
                "maximum": float(frame[column].max()),
            }
            for column in NUMERIC_COLUMNS
        },
        "valid_pixel_count_statistics": {
            key: float(value)
            for key, value in frame["valid_pixel_count"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .to_dict()
            .items()
        },
    }
    print(json.dumps(report, indent=2))
    if report["duplicate_cell_ids"] or any(report["missing_values"].values()):
        raise ValueError("Raw data contains duplicate IDs or missing required values.")
    if any(non_finite.values()):
        raise ValueError("Raw data contains non-finite numeric values.")
    if report["invalid_ndvi_count"] or report["invalid_ndbi_count"]:
        raise ValueError("Raw data contains NDVI or NDBI values outside [-1, 1].")
    if report["suspicious_lst_count"]:
        raise ValueError("Raw data contains suspicious LST values.")
    return report


def calculate_statistics(frame: pd.DataFrame) -> dict:
    """Calculate descriptive statistics for all requested analysis features."""
    statistics = {}
    for column in [
        "lst_median_c",
        "lst_p90_c",
        "ndvi_median",
        "ndbi_median",
        "valid_pixel_count",
    ]:
        values = frame[column].describe(percentiles=[0.25, 0.5, 0.75])
        statistics[column] = {
            "count": int(values["count"]),
            "mean": float(values["mean"]),
            "median": float(values["50%"]),
            "standard_deviation": float(values["std"]),
            "minimum": float(values["min"]),
            "maximum": float(values["max"]),
            "25th_percentile": float(values["25%"]),
            "75th_percentile": float(values["75%"]),
        }
    return statistics


def min_max_normalize(values: pd.Series) -> pd.Series:
    """Normalize a feature to [0, 1] using its observed dataset range."""
    minimum = values.min()
    maximum = values.max()
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum == minimum:
        raise ValueError(f"Cannot min-max normalize {values.name}: invalid range.")
    return (values - minimum) / (maximum - minimum)


def create_baseline(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create the explainable 0-100 baseline index and percentile classes."""
    output = frame.copy()
    output["lst_score"] = min_max_normalize(output["lst_median_c"])
    ndbi_score = min_max_normalize(output["ndbi_median"])
    normalized_ndvi = min_max_normalize(output["ndvi_median"])
    output["ndbi_score"] = ndbi_score
    output["vegetation_score"] = 1 - normalized_ndvi
    output["baseline_heat_risk_score"] = 100 * (
        0.50 * output["lst_score"]
        + 0.30 * output["ndbi_score"]
        + 0.20 * output["vegetation_score"]
    )

    low_threshold = float(output["baseline_heat_risk_score"].quantile(1 / 3))
    high_threshold = float(output["baseline_heat_risk_score"].quantile(2 / 3))
    output["heat_risk_class"] = pd.cut(
        output["baseline_heat_risk_score"],
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    metadata = {
        "source_dataset": str(INPUT_PATH.relative_to(ROOT_DIR)),
        "grid_cell_count": int(len(output)),
        "normalization_method": "min-max normalization using each feature's observed range",
        "normalization_ranges": {
            feature: {
                "minimum": float(frame[feature].min()),
                "maximum": float(frame[feature].max()),
            }
            for feature in BASELINE_FEATURES
        },
        "formula": "100 * (0.50 * lst_score + 0.30 * ndbi_score + 0.20 * vegetation_score)",
        "contributions": {
            "lst_score": 0.50,
            "ndbi_score": 0.30,
            "vegetation_score": 0.20,
        },
        "feature_definitions": {
            "lst_score": "higher observed median LST means higher hazard",
            "ndbi_score": "higher observed NDBI means more built-up surface",
            "vegetation_score": "1 - normalized NDVI, so lower vegetation means higher score",
        },
        "class_thresholds": {
            "low_upper_inclusive": low_threshold,
            "medium_upper_inclusive": high_threshold,
            "percentiles": {"low": "0-33.333rd", "medium": "33.333rd-66.667th", "high": "66.667th-100th"},
        },
        "class_counts": {
            str(key): int(value) for key, value in output["heat_risk_class"].value_counts().items()
        },
        "interpretation": "Baseline heat-risk index, not a validated medical or environmental risk model.",
        "ml_status": "No supervised ML model was trained; this index is not an ML target or prediction.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return output, metadata


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    validation = validate_input(frame)
    statistics = calculate_statistics(frame)
    processed, metadata = create_baseline(frame)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)
    STATISTICS_PATH.write_text(
        json.dumps({"validation": validation, "statistics": statistics}, indent=2) + "\n",
        encoding="utf-8",
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote processed dataset: {PROCESSED_PATH}")
    print(f"Wrote statistics: {STATISTICS_PATH}")
    print(f"Wrote baseline metadata: {METADATA_PATH}")
    print(processed["heat_risk_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()