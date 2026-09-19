"""Create and populate the UrbanCool SQLite database from real CSV outputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from backend.app.db.database import DATABASE_PATH


ROOT_DIR = Path(__file__).resolve().parents[3]
CELLS_CSV = ROOT_DIR / "data/processed/urban_heat_coimbatore_processed.csv"
RECOMMENDATIONS_CSV = ROOT_DIR / "data/processed/urban_heat_coimbatore_recommendations.csv"


def create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS cells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL UNIQUE,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            lst_median_c REAL NOT NULL,
            lst_p90_c REAL NOT NULL,
            ndvi_median REAL NOT NULL,
            ndbi_median REAL NOT NULL,
            baseline_heat_risk_score REAL NOT NULL,
            heat_risk_class TEXT NOT NULL CHECK (heat_risk_class IN ('Low', 'Medium', 'High'))
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL,
            rank INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 3),
            intervention TEXT NOT NULL,
            score REAL NOT NULL,
            priority TEXT NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(cell_id, rank),
            FOREIGN KEY(cell_id) REFERENCES cells(cell_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_cells_risk_class ON cells(heat_risk_class);
        CREATE INDEX IF NOT EXISTS idx_recommendations_cell_id ON recommendations(cell_id);
        """
    )


def import_data() -> dict[str, int]:
    cells = pd.read_csv(CELLS_CSV)
    recommendations = pd.read_csv(RECOMMENDATIONS_CSV)
    required_cells = [
        "cell_id", "latitude", "longitude", "lst_median_c", "lst_p90_c",
        "ndvi_median", "ndbi_median", "baseline_heat_risk_score", "heat_risk_class",
    ]
    missing_cells = [column for column in required_cells if column not in cells]
    if missing_cells:
        raise ValueError(f"Cells CSV is missing columns: {missing_cells}")
    if cells["cell_id"].duplicated().any():
        raise ValueError("Cells CSV contains duplicate cell IDs.")
    if cells[required_cells].isna().any().any():
        raise ValueError("Cells CSV contains null values in required fields.")

    required_recommendations = [
        "cell_id",
        *[f"recommendation_{rank}{suffix}" for rank in range(1, 4) for suffix in ("", "_score", "_priority", "_reason")],
    ]
    missing_recommendations = [column for column in required_recommendations if column not in recommendations]
    if missing_recommendations:
        raise ValueError(f"Recommendations CSV is missing columns: {missing_recommendations}")
    if set(recommendations["cell_id"]) != set(cells["cell_id"]):
        raise ValueError("Recommendation cell IDs do not exactly match the cells CSV.")

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            create_tables(connection)
            connection.execute("DELETE FROM recommendations")
            connection.execute("DELETE FROM cells")
            connection.executemany(
                """
                INSERT INTO cells (
                    cell_id, latitude, longitude, lst_median_c, lst_p90_c,
                    ndvi_median, ndbi_median, baseline_heat_risk_score, heat_risk_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                cells[required_cells].itertuples(index=False, name=None),
            )
            recommendation_rows = []
            for row in recommendations.itertuples(index=False):
                values = row._asdict()
                for rank in range(1, 4):
                    recommendation_rows.append(
                        (
                            values["cell_id"],
                            rank,
                            values[f"recommendation_{rank}"],
                            values[f"recommendation_{rank}_score"],
                            values[f"recommendation_{rank}_priority"],
                            values[f"recommendation_{rank}_reason"],
                        )
                    )
            connection.executemany(
                """
                INSERT INTO recommendations (cell_id, rank, intervention, score, priority, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                recommendation_rows,
            )
        counts = {
            "cells": connection.execute("SELECT COUNT(*) FROM cells").fetchone()[0],
            "recommendations": connection.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0],
        }
        print(f"Database: {DATABASE_PATH}")
        print(f"Imported cells: {counts['cells']}")
        print(f"Imported recommendations: {counts['recommendations']}")
        return counts
    finally:
        connection.close()


if __name__ == "__main__":
    import_data()