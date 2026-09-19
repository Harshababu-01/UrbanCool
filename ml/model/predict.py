"""Load the trained UrbanCool LST model and predict from non-LST features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "data/processed/models/urban_heat_lst_model.joblib"
DATA_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"


def load_model() -> dict:
    """Load the model artifact and its feature contract."""
    return joblib.load(MODEL_PATH)


def predict_lst(features: pd.DataFrame) -> pd.Series:
    """Predict median LST from the model's non-LST input features only."""
    artifact = load_model()
    feature_names = artifact["feature_names"]
    missing = [feature for feature in feature_names if feature not in features]
    if missing:
        raise ValueError(f"Missing prediction features: {missing}")
    return pd.Series(artifact["model"].predict(features[feature_names]), index=features.index)


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    artifact = load_model()
    predictions = predict_lst(frame.head(5))
    print(f"Features used: {artifact['feature_names']}")
    print("Example predicted median LST values (C):")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()