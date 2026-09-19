"""Evaluate LST regression on geographically held-out spatial blocks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
RANDOM_METRICS_PATH = ROOT_DIR / "data/processed/model_metrics.json"
OUTPUT_PATH = ROOT_DIR / "data/processed/spatial_evaluation.json"
IMPORTANCE_PATH = ROOT_DIR / "data/processed/spatial_feature_importance.json"
MODEL_PATH = ROOT_DIR / "data/processed/models/urban_heat_lst_spatial_model.joblib"
FIGURE_DIR = ROOT_DIR / "data/processed/figures"

TARGET = "lst_median_c"
FEATURES = ["ndvi_median", "ndbi_median", "latitude", "longitude"]
EXCLUDED_FEATURES = [
    "lst_median_c",
    "lst_p90_c",
    "baseline_heat_risk_score",
    "heat_risk_class",
    "valid_pixel_count",
]
BLOCK_SIZE_DEGREES = 0.1
RANDOM_STATE = 42
N_ESTIMATORS = 200


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    required = [TARGET, *FEATURES, *EXCLUDED_FEATURES]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    if frame[[TARGET, *FEATURES]].isna().any().any():
        raise ValueError("Target or spatial-model features contain missing values.")
    return frame


def assign_spatial_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign deterministic 0.1-degree latitude/longitude blocks."""
    output = frame.copy()
    minimum_latitude = output["latitude"].min()
    minimum_longitude = output["longitude"].min()
    output["latitude_band"] = np.floor(
        (output["latitude"] - minimum_latitude) / BLOCK_SIZE_DEGREES
    ).astype(int)
    output["longitude_band"] = np.floor(
        (output["longitude"] - minimum_longitude) / BLOCK_SIZE_DEGREES
    ).astype(int)
    output["spatial_block_id"] = (
        "B_"
        + output["latitude_band"].astype(str)
        + "_"
        + output["longitude_band"].astype(str)
    )

    # A deterministic modular rule distributes blocks without random cell assignment.
    output["is_test_block"] = (
        (output["latitude_band"] * 31 + output["longitude_band"] * 17) % 5 == 0
    )
    block_table = output[["spatial_block_id", "is_test_block"]].drop_duplicates()
    metadata = {
        "method": "floor latitude and longitude into fixed geographic bands, then combine band indices",
        "block_size_degrees": BLOCK_SIZE_DEGREES,
        "latitude_origin": float(minimum_latitude),
        "longitude_origin": float(minimum_longitude),
        "test_selection_rule": "(latitude_band * 31 + longitude_band * 17) % 5 == 0",
        "total_blocks": int(len(block_table)),
        "training_blocks": int((~block_table["is_test_block"]).sum()),
        "test_blocks": int(block_table["is_test_block"].sum()),
    }
    return output, metadata


def validate_split(frame: pd.DataFrame) -> tuple[pd.Index, pd.Index, dict]:
    train_indices = frame.index[~frame["is_test_block"]]
    test_indices = frame.index[frame["is_test_block"]]
    train_blocks = set(frame.loc[train_indices, "spatial_block_id"])
    test_blocks = set(frame.loc[test_indices, "spatial_block_id"])
    if train_blocks.intersection(test_blocks):
        raise ValueError("A spatial test block appears in the training data.")
    if set(train_indices).intersection(test_indices):
        raise ValueError("Training and spatial test rows overlap.")
    model_inputs = set(FEATURES)
    if TARGET in model_inputs or model_inputs.intersection(EXCLUDED_FEATURES):
        raise ValueError("Target or leakage-prone columns found in spatial model inputs.")
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Spatial split must contain both training and test rows.")
    return train_indices, test_indices, {
        "no_test_block_in_training": True,
        "no_row_overlap": True,
        "target_excluded_from_inputs": True,
        "leakage_columns_excluded_from_inputs": True,
        "test_rows_are_held_out": True,
    }


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def save_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual, predicted, s=8, alpha=0.28, color="#1b6ca8", edgecolors="none")
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    axis.plot([lower, upper], [lower, upper], "--", color="#d95f02", linewidth=1.5)
    axis.set_title("Spatial Test Blocks: Actual vs Predicted Median LST")
    axis.set_xlabel("Actual LST (°C)")
    axis.set_ylabel("Predicted LST (°C)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "spatial_actual_vs_predicted_lst.png", dpi=180)
    plt.close(figure)


def save_spatial_blocks_plot(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    training = frame[~frame["is_test_block"]]
    testing = frame[frame["is_test_block"]]
    axis.scatter(
        training["longitude"], training["latitude"], s=4, alpha=0.25,
        color="#3182bd", label="Training blocks", edgecolors="none"
    )
    axis.scatter(
        testing["longitude"], testing["latitude"], s=8, alpha=0.55,
        color="#de2d26", label="Testing blocks", edgecolors="none"
    )
    axis.set_title("Spatial Train/Test Blocks")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "spatial_test_blocks.png", dpi=200)
    plt.close(figure)


def save_feature_importance(model: RandomForestRegressor) -> dict[str, float]:
    importance = {
        feature: float(value) for feature, value in zip(FEATURES, model.feature_importances_)
    }
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2) + "\n", encoding="utf-8")
    ordered = sorted(importance.items(), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh([item[0] for item in ordered], [item[1] for item in ordered], color="#756bb1")
    axis.set_title("Spatial Random Forest Feature Importance")
    axis.set_xlabel("Importance")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "spatial_feature_importance.png", dpi=180)
    plt.close(figure)
    return importance


def main() -> None:
    frame = load_dataset()
    frame, block_metadata = assign_spatial_blocks(frame)
    train_indices, test_indices, validation = validate_split(frame)
    actual = frame.loc[test_indices, TARGET].to_numpy()

    training_mean = frame.loc[train_indices, TARGET].mean()
    baseline_metrics = calculate_metrics(actual, np.full(len(test_indices), training_mean))

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(frame.loc[train_indices, FEATURES], frame.loc[train_indices, TARGET])
    predicted = model.predict(frame.loc[test_indices, FEATURES])
    spatial_metrics = calculate_metrics(actual, predicted)

    random_metrics = json.loads(RANDOM_METRICS_PATH.read_text(encoding="utf-8"))["random_forest"]["metrics"]
    performance_difference = {
        metric: float(random_metrics[metric] - spatial_metrics[metric])
        for metric in ("mae", "rmse", "r2")
    }
    importance = save_feature_importance(model)
    save_actual_vs_predicted(actual, predicted)
    save_spatial_blocks_plot(frame)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target": TARGET,
            "feature_names": FEATURES,
            "random_state": RANDOM_STATE,
            "spatial_block_size_degrees": BLOCK_SIZE_DEGREES,
        },
        MODEL_PATH,
    )

    result = {
        "dataset": str(DATA_PATH.relative_to(ROOT_DIR)),
        "target": TARGET,
        "features": FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "spatial_blocking": block_metadata,
        "split": {
            "training_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "training_mean_lst_c": float(training_mean),
        },
        "random_split": random_metrics,
        "spatial_split": spatial_metrics,
        "baseline": baseline_metrics,
        "performance_difference_random_minus_spatial": performance_difference,
        "feature_importance": importance,
        "validation": validation,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "Student prototype; spatial holdout is stronger than random splitting but is not scientific validation.",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved spatial model: {MODEL_PATH}")


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    main()