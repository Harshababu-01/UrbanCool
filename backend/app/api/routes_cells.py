"""Cell and recommendation API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.db.database import get_connection
from backend.app.schemas.api_schemas import (
    CellDetail,
    CellMetrics,
    CellRisk,
    CellSummary,
    Location,
    RecommendationResponse,
)


router = APIRouter(prefix="/api/cells", tags=["cells"])


def recommendation_rows(connection, cell_id: str) -> list[RecommendationResponse]:
    rows = connection.execute(
        "SELECT rank, intervention, score, priority, reason FROM recommendations WHERE cell_id = ? ORDER BY rank",
        (cell_id,),
    ).fetchall()
    return [RecommendationResponse(**dict(row)) for row in rows]


def cell_summary(row) -> CellSummary:
    return CellSummary(
        cell_id=row["cell_id"],
        location=Location(latitude=row["latitude"], longitude=row["longitude"]),
        risk=CellRisk(score=row["baseline_heat_risk_score"], **{"class": row["heat_risk_class"]}),
    )


@router.get("", response_model=list[CellSummary])
def list_cells(
    risk_class: Optional[str] = Query(default=None, pattern="^(Low|Medium|High)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    connection = get_connection()
    try:
        query = "SELECT * FROM cells"
        parameters: list[object] = []
        if risk_class:
            query += " WHERE heat_risk_class = ?"
            parameters.append(risk_class)
        query += " ORDER BY cell_id LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])
        rows = connection.execute(query, parameters).fetchall()
        return [cell_summary(row) for row in rows]
    finally:
        connection.close()


@router.get("/{cell_id}", response_model=CellDetail)
def get_cell(cell_id: str):
    connection = get_connection()
    try:
        row = connection.execute("SELECT * FROM cells WHERE cell_id = ?", (cell_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Cell not found")
        summary = cell_summary(row)
        return CellDetail(
            **summary.model_dump(),
            metrics=CellMetrics(
                lst_median_c=row["lst_median_c"],
                lst_p90_c=row["lst_p90_c"],
                ndvi_median=row["ndvi_median"],
                ndbi_median=row["ndbi_median"],
            ),
            recommendations=recommendation_rows(connection, cell_id),
        )
    finally:
        connection.close()


@router.get("/{cell_id}/recommendations", response_model=list[RecommendationResponse])
def get_recommendations(cell_id: str):
    connection = get_connection()
    try:
        exists = connection.execute("SELECT 1 FROM cells WHERE cell_id = ?", (cell_id,)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Cell not found")
        return recommendation_rows(connection, cell_id)
    finally:
        connection.close()