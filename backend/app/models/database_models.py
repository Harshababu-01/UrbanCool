"""Database row shape documentation for the SQLite-backed API."""

CELL_COLUMNS = (
    "cell_id",
    "latitude",
    "longitude",
    "lst_median_c",
    "lst_p90_c",
    "ndvi_median",
    "ndbi_median",
    "baseline_heat_risk_score",
    "heat_risk_class",
)
RECOMMENDATION_COLUMNS = ("rank", "intervention", "score", "priority", "reason")