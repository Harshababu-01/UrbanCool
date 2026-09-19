"""Dataset statistics API route."""

from fastapi import APIRouter

from backend.app.db.database import get_connection
from backend.app.schemas.api_schemas import StatisticsResponse


router = APIRouter(prefix="/api", tags=["statistics"])


@router.get("/statistics", response_model=StatisticsResponse)
def get_statistics():
    connection = get_connection()
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_cells,
                SUM(CASE WHEN heat_risk_class = 'Low' THEN 1 ELSE 0 END) AS low_risk_cells,
                SUM(CASE WHEN heat_risk_class = 'Medium' THEN 1 ELSE 0 END) AS medium_risk_cells,
                SUM(CASE WHEN heat_risk_class = 'High' THEN 1 ELSE 0 END) AS high_risk_cells,
                AVG(lst_median_c) AS average_lst,
                AVG(ndvi_median) AS average_ndvi,
                AVG(ndbi_median) AS average_ndbi
            FROM cells
            """
        ).fetchone()
        return StatisticsResponse(**dict(row))
    finally:
        connection.close()