"""Train and evaluate leakage-safe LST regression experiments."""

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
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
METRICS_PATH = ROOT_DIR / "data/processed/model_metrics.json"
COMPARISON_PATH = ROOT_DIR / "data/processed/model_comparison.json"
IMPORTANCE_PATH = ROOT_DIR / "data/processed/feature_importance.json"
MODEL_PATH = ROOT_DIR / "data/processed/models/urban_heat_lst_model.joblib"
FIGURE_DIR = ROOT_DIR / "data/processed/figures"

TARGET = "lst_median_c"
PRIMARY_FEATURES = ["ndvi_median", "ndbi_median", "latitude", "longitude"]
SECONDARY_FEATURES = ["ndvi_median", "ndbi_median"]
EXCLUDED_FEATURES = [
    "lst_median_c",
    "lst_p90_c",
    "baseline_heat_risk_score",
    "heat_risk_class",
    "valid_pixel_count",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    required = [TARGET, *PRIMARY_FEATURES, *EXCLUDED_FEATURES]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    if frame[[TARGET, *PRIMARY_FEATURES]].isna().any().any():
        raise ValueError("Target or model input features contain missing values.")
    return frame


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def fit_random_forest(
    frame: pd.DataFrame,
    feature_names: list[str],
    train_indices: pd.Index,
    test_indices: pd.Index,
) -> tuple[RandomForestRegressor, dict[str, float], np.ndarray, np.ndarray]:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(frame.loc[train_indices, feature_names], frame.loc[train_indices, TARGET])
    actual = frame.loc[test_indices, TARGET].to_numpy()
    predicted = model.predict(frame.loc[test_indices, feature_names])
    return model, calculate_metrics(actual, predicted), actual, predicted


def save_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual, predicted, s=8, alpha=0.28, color="#1b6ca8", edgecolors="none")
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#d95f02", linewidth=1.5)
    axis.set_title("Actual vs Predicted Median LST")
    axis.set_xlabel("Actual LST (°C)")
    axis.set_ylabel("Predicted LST (°C)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "actual_vs_predicted_lst.png", dpi=180)
    plt.close(figure)


def save_residual_distribution(residuals: np.ndarray) -> dict[str, float]:
    statistics = {
        "count": int(len(residuals)),
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "standard_deviation": float(np.std(residuals, ddof=1)),
        "minimum": float(np.min(residuals)),
        "maximum": float(np.max(residuals)),
        "25th_percentile": float(np.percentile(residuals, 25)),
        "75th_percentile": float(np.percentile(residuals, 75)),
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(residuals, bins=40, color="#7570b3", edgecolor="white", linewidth=0.4)
    axis.axvline(0, color="#d95f02", linestyle="--", linewidth=1.5)
    axis.set_title("Random Forest LST Residual Distribution")
    axis.set_xlabel("Residual: actual LST - predicted LST (°C)")
    axis.set_ylabel("Test cells")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "residual_distribution.png", dpi=180)
    plt.close(figure)
    return statistics


def save_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> dict[str, float]:
    importance = {
        feature: float(value)
        for feature, value in zip(feature_names, model.feature_importances_)
    }
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2) + "\n", encoding="utf-8")
    ordered = sorted(importance.items(), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh([item[0] for item in ordered], [item[1] for item in ordered], color="#1b9e77")
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Importance")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "feature_importance.png", dpi=180)
    plt.close(figure)
    return importance


def main() -> None:
    frame = load_dataset()
    indices = frame.index
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_indices = pd.Index(train_indices)
    test_indices = pd.Index(test_indices)

    mean_prediction = frame.loc[train_indices, TARGET].mean()
    test_actual = frame.loc[test_indices, TARGET].to_numpy()
    mean_predicted = np.full(len(test_indices), mean_prediction)
    mean_metrics = calculate_metrics(test_actual, mean_predicted)

    model, random_forest_metrics, actual, predicted = fit_random_forest(
        frame, PRIMARY_FEATURES, train_indices, test_indices
    )
    residuals = actual - predicted
    residual_statistics = save_residual_distribution(residuals)
    save_actual_vs_predicted(actual, predicted)
    importance = save_feature_importance(model, PRIMARY_FEATURES)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target": TARGET,
            "feature_names": PRIMARY_FEATURES,
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )

    secondary_model, secondary_metrics, _, _ = fit_random_forest(
        frame, SECONDARY_FEATURES, train_indices, test_indices
    )
    metrics = {
        "dataset": str(DATA_PATH.relative_to(ROOT_DIR)),
        "target": TARGET,
        "split": {
            "train_fraction": 0.80,
            "test_fraction": TEST_SIZE,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "random_state": RANDOM_STATE,
        },
        "primary_features": PRIMARY_FEATURES,
        "optional_feature_included": False,
        "excluded_features": EXCLUDED_FEATURES,
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "mean_training_target_baseline": {
            "training_mean_lst_c": float(mean_prediction),
            "metrics": mean_metrics,
        },
        "random_forest": {"metrics": random_forest_metrics},
        "residual_statistics": residual_statistics,
        "feature_importance": importance,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "Student prototype on a bounded Coimbatore satellite dataset; not scientifically validated.",
    }
    comparison = {
        "same_test_split": True,
        "random_state": RANDOM_STATE,
        "experiments": {
            "coordinates_plus_environment": {
                "features": PRIMARY_FEATURES,
                "metrics": random_forest_metrics,
            },
            "environment_only": {
                "features": SECONDARY_FEATURES,
                "metrics": secondary_metrics,
            },
        },
        "interpretation": "The coordinates-plus-environment experiment is compared with an environment-only experiment to examine the effect of spatial location.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    del secondary_model


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    main()