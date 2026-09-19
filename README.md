# UrbanCool: Earth Engine Extraction

This milestone extracts real Landsat 8/9 Collection 2 Level-2 features for a reproducible Coimbatore study area.

## Setup and run

From the project root in PowerShell:

```powershell
\.venv\Scripts\Activate.ps1
python ml/data/extract_gee.py
```

Earth Engine authentication must already be completed. The script initializes Earth Engine with project `ai-insight-484516`.

The study area is the `Coimbatore` district feature from `FAO/GAUL/2015/level2`. This is an isolated fallback because a dependable official municipal boundary is not exposed by the public Earth Engine catalog used here. The boundary function can later be replaced with an official municipal GeoJSON without changing the rest of the pipeline.

## Outputs

- `data/raw/urban_heat_coimbatore_2022_2025.csv`
- `data/raw/urban_heat_coimbatore_2022_2025_metadata.json`

Each CSV row is one stable 500 m grid cell. `lst_median_c` is the seasonal median land-surface temperature in Celsius, `lst_p90_c` is its 90th percentile, `ndvi_median` describes vegetation, `ndbi_median` describes built-up intensity, and `valid_pixel_count` counts valid Landsat temperature observations used for the cell.

## Current Pipeline

Earth Engine
-> Landsat 8/9
-> Collection 2 Level-2 preprocessing
-> LST/NDVI/NDBI
-> 500 m grid
-> raw CSV
-> baseline heat-risk analysis

Run the analysis and visualizations from the project root:

```powershell
\.venv\Scripts\Activate.ps1
python ml/data/analyze_dataset.py
python ml/data/visualize_dataset.py
```

The analysis validates the real raw dataset, writes `data/processed/urban_heat_coimbatore_processed.csv`, descriptive statistics, baseline metadata, and matplotlib figures under `data/processed/figures/`.

The baseline heat-risk index is a transparent descriptive indicator: 50% observed LST, 30% observed NDBI, and 20% vegetation deficit derived from NDVI. It is not a validated medical or environmental risk model. A future ML prediction model will need a carefully designed target and leakage review; it is not trained in this milestone, and LST is not being used as an ML input here.

## Machine Learning Prototype

The current ML experiment predicts `lst_median_c` as a continuous target. LST target columns, the baseline score, and the baseline class are deliberately excluded from the inputs to prevent leakage. The primary Random Forest inputs are `ndvi_median`, `ndbi_median`, `latitude`, and `longitude`; `valid_pixel_count` is intentionally excluded because it is an observation-availability measure rather than an environmental predictor.

The experiment uses an 80/20 train/test split with `random_state=42`. A training-set mean LST predictor is evaluated against a `RandomForestRegressor` with 200 trees. An additional NDVI/NDBI-only experiment uses the same held-out test cells to compare environmental inputs without coordinates. Metrics and model artifacts are written under `data/processed/`.

Run the experiment from the project root:

```powershell
\.venv\Scripts\Activate.ps1
python ml/model/train_model.py
python ml/model/predict.py
```

This is a student prototype on a bounded Coimbatore satellite dataset, not a scientifically validated heat-prediction model.

## Spatial Generalization Evaluation

The random train/test split can overestimate performance for spatial data because neighboring grid cells can be highly correlated. The spatial evaluation therefore divides the grid into fixed 0.1-degree latitude/longitude blocks and holds out deterministic geographic blocks rather than randomly assigning individual cells. No test block is used during training.

The spatial model uses the same leakage-safe target and features as the random-split experiment. Its results are saved in `data/processed/spatial_evaluation.json`, with the spatial model at `data/processed/models/urban_heat_lst_spatial_model.joblib` and diagnostic figures under `data/processed/figures/`. This is a stronger student-prototype evaluation, not scientific validation.

Run it from the project root:

```powershell
\.venv\Scripts\Activate.ps1
python ml/model/spatial_evaluation.py
```

## Cooling Recommendation Engine

Real satellite-derived environmental features
-> transparent intervention scoring
-> top 3 location-specific cooling recommendations

The recommendation engine uses each cell's LST, NDVI, NDBI, baseline heat score, and heat class. Its thresholds are calculated from the real processed dataset using percentiles, and its deterministic rules score tree/vegetation cover, cool/reflective roofs, shaded public or pedestrian areas, and green corridors.

This rule-based decision support is separate from ML prediction. The ML model estimates median LST from non-LST features, while the recommendation engine translates observed environmental conditions into explainable intervention priorities. Recommendations are suggestions, not guaranteed intervention outcomes.

Run batch generation from the project root:

```powershell
\.venv\Scripts\Activate.ps1
python -m backend.app.services.recommendation_engine
python -m unittest tests/test_recommendation_engine.py
```

The generated recommendations are saved to `data/processed/urban_heat_coimbatore_recommendations.csv` and the threshold/scoring documentation is saved to `data/processed/recommendation_metadata.json`.

## UrbanCool Backend

The backend loads the existing processed heat-risk and recommendation CSVs into SQLite at `data/processed/urbancool.db`. FastAPI exposes the data for a future frontend without storing raw satellite imagery.

Initialize or refresh the database from the project root:

```powershell
\.venv\Scripts\Activate.ps1
python -m backend.app.db.init_db
```

Start the API:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Available requests include:

```text
GET http://127.0.0.1:8000/api/health
GET http://127.0.0.1:8000/api/cells?risk_class=High&limit=50
GET http://127.0.0.1:8000/api/cells/UC_...
GET http://127.0.0.1:8000/api/cells/UC_.../recommendations
GET http://127.0.0.1:8000/api/statistics
```

## Grounded AI Advisor

The AI Advisor is an explanation layer over the existing UrbanCool data. The frontend sends only a
selected `cell_id` and question to `POST /api/advisor`. FastAPI retrieves that cell's metrics,
official heat-risk score/class, and deterministic recommendations from SQLite before sending the
grounded context to Google Gemini through the official GenAI SDK. The LLM cannot change the official risk score or
recommendation scores, and it must identify information that is unavailable from the dataset.

Configure the provider on the backend only. Never put these values in Vite environment variables or
the frontend bundle:

```powershell
$env:GEMINI_API_KEY = "your-Gemini-api-key"
# Optional: $env:GEMINI_MODEL = "gemini-2.5-flash"
python -m uvicorn backend.app.main:app --reload
```

Without `GEMINI_API_KEY`, the endpoint returns an explicit unavailable
state; it does not fabricate an AI response. The advisor uses processed satellite-derived data and
is not real-time. Local planning, ownership, cost, feasibility, health, and community context are
outside the current dataset and require local assessment.