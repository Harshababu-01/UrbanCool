"""Transparent, deterministic cooling recommendations for UrbanCool cells."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
OUTPUT_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_recommendations.csv"
METADATA_PATH = ROOT_DIR / "data/processed/recommendation_metadata.json"

REQUIRED_COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
    "lst_median_c",
    "lst_p90_c",
    "ndvi_median",
    "ndbi_median",
    "baseline_heat_risk_score",
    "heat_risk_class",
]
THRESHOLD_PERCENTILES = {
    "low_vegetation_ndvi": 0.33,
    "high_built_up_ndbi": 0.67,
    "high_heat_lst": 0.67,
    "high_heat_baseline_score": 0.67,
}
PRIORITY_THRESHOLDS = {"high": 66.0, "medium": 33.0}


def load_processed_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Recommendation inputs contain missing values.")
    if frame["cell_id"].duplicated().any():
        raise ValueError("Recommendation input contains duplicate cell IDs.")
    return frame


def calculate_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    return {
        name: float(frame[column].quantile(percentile))
        for name, column, percentile in (
            ("low_vegetation_ndvi", "ndvi_median", THRESHOLD_PERCENTILES["low_vegetation_ndvi"]),
            ("high_built_up_ndbi", "ndbi_median", THRESHOLD_PERCENTILES["high_built_up_ndbi"]),
            ("high_heat_lst", "lst_median_c", THRESHOLD_PERCENTILES["high_heat_lst"]),
            (
                "high_heat_baseline_score",
                "baseline_heat_risk_score",
                THRESHOLD_PERCENTILES["high_heat_baseline_score"],
            ),
        )
    }


def calculate_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {"minimum": float(frame[column].min()), "maximum": float(frame[column].max())}
        for column in ("lst_median_c", "ndvi_median", "ndbi_median", "baseline_heat_risk_score")
    }


def scale(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def score_from_threshold(value: float, threshold: float, maximum: float, inverse: bool = False) -> float:
    if inverse:
        return 100.0 * scale(threshold - value, threshold - maximum, 0.0)
    return 100.0 * scale(value, threshold, maximum)


def priority_level(score: float) -> str:
    if score >= PRIORITY_THRESHOLDS["high"]:
        return "High"
    if score >= PRIORITY_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def generate_recommendations(
    cell: dict[str, Any] | pd.Series,
    thresholds: dict[str, float],
    ranges: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Generate four scored interventions and return the top three."""
    values = dict(cell)
    heat_signal = scale(
        float(values["baseline_heat_risk_score"]),
        thresholds["high_heat_baseline_score"],
        ranges["baseline_heat_risk_score"]["maximum"],
    )
    heat_likely = (
        float(values["baseline_heat_risk_score"]) >= thresholds["high_heat_baseline_score"]
        or float(values["lst_median_c"]) >= thresholds["high_heat_lst"]
    )
    vegetation_gap = 1.0 - scale(
        float(values["ndvi_median"]),
        ranges["ndvi_median"]["minimum"],
        thresholds["low_vegetation_ndvi"],
    )
    built_up_pressure = scale(
        float(values["ndbi_median"]),
        thresholds["high_built_up_ndbi"],
        ranges["ndbi_median"]["maximum"],
    )
    heat_score = 100.0 * heat_signal
    vegetation_score = 100.0 * vegetation_gap
    built_up_score = 100.0 * built_up_pressure

    if heat_likely and float(values["ndvi_median"]) < thresholds["low_vegetation_ndvi"]:
        tree_reason = (
            f"High heat exposure (median LST {values['lst_median_c']:.2f} °C) combines "
            f"with low vegetation (NDVI {values['ndvi_median']:.3f})."
        )
    elif float(values["ndvi_median"]) < thresholds["low_vegetation_ndvi"]:
        tree_reason = (
            f"Vegetation is relatively low (NDVI {values['ndvi_median']:.3f}), "
            "so additional tree or vegetation cover is prioritized."
        )
    else:
        tree_reason = (
            f"NDVI is {values['ndvi_median']:.3f}; vegetation cover remains a lower-priority "
            "cooling opportunity relative to other interventions."
        )

    if heat_likely and float(values["ndbi_median"]) >= thresholds["high_built_up_ndbi"]:
        roof_reason = (
            f"High heat exposure combines with high built-up intensity "
            f"(NDBI {values['ndbi_median']:.3f})."
        )
    elif float(values["ndbi_median"]) >= thresholds["high_built_up_ndbi"]:
        roof_reason = f"Built-up intensity is relatively high (NDBI {values['ndbi_median']:.3f})."
    else:
        roof_reason = f"NDBI is {values['ndbi_median']:.3f}; reflective roofs are a lower-priority option here."

    if heat_likely:
        shade_reason = (
            f"Heat exposure is elevated (median LST {values['lst_median_c']:.2f} °C; "
            f"baseline score {values['baseline_heat_risk_score']:.1f})."
        )
    else:
        shade_reason = (
            f"Median LST is {values['lst_median_c']:.2f} °C and baseline score is "
            f"{values['baseline_heat_risk_score']:.1f}; targeted shade is a lower-priority option."
        )

    corridor_reason = (
        f"Connect vegetation where heat exposure (LST {values['lst_median_c']:.2f} °C) "
        f"and vegetation level (NDVI {values['ndvi_median']:.3f}) indicate cooling need."
        if heat_likely and float(values["ndvi_median"]) < thresholds["low_vegetation_ndvi"]
        else f"NDVI {values['ndvi_median']:.3f} and baseline score {values['baseline_heat_risk_score']:.1f} support a lower-priority corridor opportunity."
    )

    recommendations = [
        {
            "intervention": "Tree / Vegetation Cover",
            "score": round(100.0 * (0.60 * heat_signal + 0.40 * vegetation_gap), 2),
            "reason": tree_reason,
        },
        {
            "intervention": "Cool / Reflective Roofs",
            "score": round(100.0 * (0.60 * heat_signal + 0.40 * built_up_pressure), 2),
            "reason": roof_reason,
        },
        {
            "intervention": "Shaded Public / Pedestrian Areas",
            "score": round(heat_score, 2),
            "reason": shade_reason,
        },
        {
            "intervention": "Green Corridors / Connected Vegetation",
            "score": round(100.0 * (0.50 * heat_signal + 0.50 * vegetation_gap), 2),
            "reason": corridor_reason,
        },
    ]
    for recommendation in recommendations:
        recommendation["priority"] = priority_level(recommendation["score"])
    ranked = sorted(
        enumerate(recommendations),
        key=lambda item: (-item[1]["score"], item[0]),
    )
    return [item[1] for item in ranked[:3]]


def build_metadata(frame: pd.DataFrame, thresholds: dict[str, float], ranges: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "source_dataset": str(DATA_PATH.relative_to(ROOT_DIR)),
        "cell_count": int(len(frame)),
        "intervention_categories": [
            "Tree / Vegetation Cover",
            "Cool / Reflective Roofs",
            "Shaded Public / Pedestrian Areas",
            "Green Corridors / Connected Vegetation",
        ],
        "threshold_percentiles": THRESHOLD_PERCENTILES,
        "threshold_values": thresholds,
        "observed_feature_ranges": ranges,
        "scoring_logic": {
            "tree_vegetation_cover": "100 * (0.60 * heat_signal + 0.40 * vegetation_gap)",
            "cool_reflective_roofs": "100 * (0.60 * heat_signal + 0.40 * built_up_pressure)",
            "shaded_public_pedestrian_areas": "100 * heat_signal",
            "green_corridors_connected_vegetation": "100 * (0.50 * heat_signal + 0.50 * vegetation_gap)",
            "heat_signal": "min-max score from high-heat baseline percentile to observed maximum baseline score",
            "vegetation_gap": "min-max score from observed minimum NDVI to low-vegetation percentile, reversed",
            "built_up_pressure": "min-max score from high-built-up percentile to observed maximum NDBI",
            "priority_levels": PRIORITY_THRESHOLDS,
        },
        "limitations": [
            "Decision-support suggestions only; no intervention outcome or temperature reduction is guaranteed.",
            "Rules use seasonal satellite-derived features and do not include cost, feasibility, land ownership, or community input.",
            "Thresholds are dataset-relative percentiles, not universal physical thresholds.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_batch(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    ranges: dict[str, dict[str, float]],
) -> pd.DataFrame:
    output_columns = REQUIRED_COLUMNS[:]
    rows = []
    for _, row in frame.iterrows():
        recommendations = generate_recommendations(row, thresholds, ranges)
        record = {column: row[column] for column in output_columns}
        for index, recommendation in enumerate(recommendations, start=1):
            record[f"recommendation_{index}"] = recommendation["intervention"]
            record[f"recommendation_{index}_score"] = recommendation["score"]
            record[f"recommendation_{index}_priority"] = recommendation["priority"]
            record[f"recommendation_{index}_reason"] = recommendation["reason"]
        rows.append(record)
    return pd.DataFrame(rows)


def lookup_cell(cell_id: str, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return environmental values and top recommendations for one cell ID."""
    source = load_processed_dataset() if frame is None else frame
    matches = source[source["cell_id"] == cell_id]
    if matches.empty:
        raise KeyError(f"Unknown cell_id: {cell_id}")
    thresholds = calculate_thresholds(source)
    ranges = calculate_ranges(source)
    row = matches.iloc[0]
    return {
        "cell_id": cell_id,
        "environmental_features": {column: row[column] for column in REQUIRED_COLUMNS[1:]},
        "recommendations": generate_recommendations(row, thresholds, ranges),
    }


def main() -> None:
    frame = load_processed_dataset()
    thresholds = calculate_thresholds(frame)
    ranges = calculate_ranges(frame)
    output = generate_batch(frame, thresholds, ranges)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    METADATA_PATH.write_text(
        json.dumps(build_metadata(frame, thresholds, ranges), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Processed cells: {len(output)}")
    print(f"Cells with recommendations: {output['recommendation_1'].notna().sum()}")
    print(f"Wrote recommendations: {OUTPUT_PATH}")
    print(f"Wrote metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()