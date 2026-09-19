"""Frontend-friendly Pydantic response schemas."""

from pydantic import BaseModel, ConfigDict, Field


class Location(BaseModel):
    latitude: float
    longitude: float


class CellMetrics(BaseModel):
    lst_median_c: float
    lst_p90_c: float
    ndvi_median: float
    ndbi_median: float


class CellRisk(BaseModel):
    score: float
    class_name: str = Field(alias="class")

    model_config = ConfigDict(populate_by_name=True)


class RecommendationResponse(BaseModel):
    rank: int
    intervention: str
    score: float
    priority: str
    reason: str


class CellSummary(BaseModel):
    cell_id: str
    location: Location
    risk: CellRisk


class CellDetail(CellSummary):
    metrics: CellMetrics
    recommendations: list[RecommendationResponse]


class StatisticsResponse(BaseModel):
    total_cells: int
    low_risk_cells: int
    medium_risk_cells: int
    high_risk_cells: int
    average_lst: float
    average_ndvi: float
    average_ndbi: float


class AdvisorRequest(BaseModel):
    cell_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)


class AdvisorGrounding(BaseModel):
    cell_id: str
    latitude: float
    longitude: float
    lst_median_c: float
    lst_p90_c: float
    ndvi_median: float
    ndbi_median: float
    baseline_heat_risk_score: float
    heat_risk_class: str
    recommendations: list[RecommendationResponse]


class AdvisorResponse(BaseModel):
    cell_id: str
    answer: str
    provider: str
    grounded: bool
    generated_at: str
    grounding: AdvisorGrounding