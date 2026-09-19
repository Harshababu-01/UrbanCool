"""Create matplotlib visualizations for the processed UrbanCool dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
FIGURE_DIR = ROOT_DIR / "data/processed/figures"


def save_distribution(frame: pd.DataFrame, column: str, title: str, filename: str, color: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(frame[column], bins=40, color=color, edgecolor="white", linewidth=0.4)
    axis.set_title(title)
    axis.set_xlabel(column)
    axis.set_ylabel("Grid cells")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def save_scatter(frame: pd.DataFrame, x: str, y: str, title: str, filename: str, color: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(frame[x], frame[y], s=5, alpha=0.25, color=color, edgecolors="none")
    axis.set_title(title)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_distribution(frame, "lst_median_c", "Median LST Distribution", "lst_distribution.png", "#d95f02")
    save_distribution(frame, "ndvi_median", "Median NDVI Distribution", "ndvi_distribution.png", "#1b9e77")
    save_distribution(frame, "ndbi_median", "Median NDBI Distribution", "ndbi_distribution.png", "#7570b3")
    save_distribution(
        frame,
        "baseline_heat_risk_score",
        "Baseline Heat-Risk Score Distribution",
        "heat_risk_score_distribution.png",
        "#e7298a",
    )
    save_scatter(
        frame,
        "ndvi_median",
        "lst_median_c",
        "NDVI vs Median LST",
        "ndvi_vs_lst.png",
        "#1b9e77",
    )
    save_scatter(
        frame,
        "ndbi_median",
        "lst_median_c",
        "NDBI vs Median LST",
        "ndbi_vs_lst.png",
        "#7570b3",
    )

    class_counts = frame["heat_risk_class"].value_counts().reindex(["Low", "Medium", "High"])
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(class_counts.index, class_counts.values, color=["#66c2a5", "#fc8d62", "#e78ac3"])
    axis.set_title("Baseline Heat-Risk Classes")
    axis.set_xlabel("Heat-risk class")
    axis.set_ylabel("Grid cells")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "heat_risk_class_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(
        frame["longitude"],
        frame["latitude"],
        c=frame["baseline_heat_risk_score"],
        s=10 + frame["baseline_heat_risk_score"] * 0.35,
        cmap="inferno",
        alpha=0.72,
        linewidths=0,
    )
    axis.set_title("Coimbatore Baseline Heat-Risk Spatial View")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Baseline heat-risk score (0-100)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "coimbatore_heat_risk_map.png", dpi=200)
    plt.close(fig)

    print(f"Generated figures in: {FIGURE_DIR}")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()