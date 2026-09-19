"""Extract seasonal Landsat heat features for the Coimbatore study area."""

from __future__ import annotations

import json
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import ee
import pandas as pd
from google.auth.transport.requests import AuthorizedSession


CITY = "Coimbatore"
START_DATE = "03-01"
END_DATE = "06-01"  # Earth Engine end dates are exclusive.
YEARS = (2022, 2023, 2024, 2025)
GRID_SIZE_METERS = 500
EE_PROJECT = "ai-insight-484516"
OUTPUT_PATH = Path("data/raw/urban_heat_coimbatore_2022_2025.csv")
METADATA_PATH = Path("data/raw/urban_heat_coimbatore_2022_2025_metadata.json")

EXPECTED_COLUMNS = [
    "cell_id",
    "latitude",
    "longitude",
    "lst_median_c",
    "lst_p90_c",
    "ndvi_median",
    "ndbi_median",
    "valid_pixel_count",
]


def initialize_earth_engine() -> None:
    """Initialize Earth Engine with the configured Cloud project."""
    ee.Initialize(project=EE_PROJECT)


def get_study_area() -> ee.Geometry:
    """Return the documented Coimbatore district boundary used for this prototype.

    GAUL Level 2 is used because an official municipal boundary is not exposed by
    a dependable public Earth Engine dataset. This function is the replacement
    point for an official municipal GeoJSON when one is supplied.
    """
    boundary = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", "Tamil Nadu"))
        .filter(ee.Filter.eq("ADM2_NAME", "Coimbatore"))
    )
    if boundary.size().getInfo() != 1:
        raise RuntimeError("Expected exactly one Coimbatore GAUL Level-2 boundary.")
    return boundary.geometry()


def build_grid(study_area: ee.Geometry) -> ee.FeatureCollection:
    """Build a stable 500 m grid with IDs derived from projected coordinates."""
    grid = study_area.coveringGrid("EPSG:3857", GRID_SIZE_METERS)

    def add_grid_properties(feature: ee.Feature) -> ee.Feature:
        centroid = feature.geometry().centroid(1).coordinates()
        longitude = ee.Number(centroid.get(0))
        latitude = ee.Number(centroid.get(1))
        projected = feature.geometry().centroid(1).transform("EPSG:3857").coordinates()
        x_index = ee.Number(projected.get(0)).divide(GRID_SIZE_METERS).floor()
        y_index = ee.Number(projected.get(1)).divide(GRID_SIZE_METERS).floor()
        cell_id = (
            ee.String("UC_")
            .cat(x_index.format("%d"))
            .cat("_")
            .cat(y_index.format("%d"))
        )
        return feature.set(
            {"cell_id": cell_id, "latitude": latitude, "longitude": longitude}
        )

    return grid.map(add_grid_properties).filterBounds(study_area)


def mask_and_scale(image: ee.Image) -> ee.Image:
    """Apply Landsat C2 L2 QA_PIXEL masking and official scale factors."""
    qa_pixel = image.select("QA_PIXEL")
    # Bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
    qa_mask = qa_pixel.bitwiseAnd(int("00111110", 2)).eq(0)
    temperature = image.select("ST_B10").multiply(0.00341802).add(149.0)
    valid_temperature = image.select("ST_B10").mask().gt(0)
    optical = image.select(["SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)

    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = optical.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    return (
        temperature.rename("LST_K")
        .subtract(273.15)
        .rename("LST")
        .addBands([ndvi, ndbi])
        .updateMask(qa_mask)
        .updateMask(valid_temperature)
    )


def get_landsat_collection(study_area: ee.Geometry) -> ee.ImageCollection:
    """Create the four-year March-May Landsat 8/9 feature collection."""
    collections = []
    for year in YEARS:
        start = f"{year}-{START_DATE}"
        end = f"{year}-{END_DATE}"
        for dataset in ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"):
            collections.append(
                ee.ImageCollection(dataset)
                .filterDate(start, end)
                .filterBounds(study_area)
            )

    merged = collections[0]
    for collection in collections[1:]:
        merged = merged.merge(collection)
    return merged.map(mask_and_scale)


def create_composite(collection: ee.ImageCollection) -> ee.Image:
    """Create median, 90th percentile LST, and valid-observation bands."""
    lst = collection.select("LST")
    return ee.Image(
        lst.median()
        .rename("lst_median_c")
        .addBands(lst.reduce(ee.Reducer.percentile([90])).rename("lst_p90_c"))
        .addBands(collection.select("NDVI").median().rename("ndvi_median"))
        .addBands(collection.select("NDBI").median().rename("ndbi_median"))
        .addBands(lst.count().rename("valid_pixel_count"))
    )


def reduce_grid(composite: ee.Image, grid: ee.FeatureCollection) -> pd.DataFrame:
    """Reduce all composite bands to one row per grid cell."""
    feature_bands = [
        "cell_id",
        "latitude",
        "longitude",
        "lst_median_c",
        "lst_p90_c",
        "ndvi_median",
        "ndbi_median",
    ]
    reduction_image = composite.select(feature_bands[3:] + ["valid_pixel_count"])
    reducer = ee.Reducer.median().combine(ee.Reducer.sum(), sharedInputs=True)
    reduced = reduction_image.reduceRegions(
        collection=grid,
        reducer=reducer,
        scale=GRID_SIZE_METERS,
        tileScale=4,
    )
    raw_frame = download_table(reduced, [])
    metadata_columns = ["cell_id", "latitude", "longitude"]
    transport_columns = metadata_columns + ["system:index", ".geo"]
    value_columns = [column for column in raw_frame if column not in transport_columns]
    if len(value_columns) != 10:
        raise ValueError(f"Unexpected Earth Engine reducer columns: {value_columns}")
    frame = raw_frame[metadata_columns].copy()
    for band, reducer_name in zip(feature_bands[3:], ("median",) * 4):
        column = next(
            column for column in value_columns if band in column and reducer_name in column
        )
        frame[band] = raw_frame[column]
    count_column = next(
        column
        for column in value_columns
        if "valid_pixel_count" in column and "sum" in column
    )
    frame["valid_pixel_count"] = raw_frame[count_column]
    return frame


def download_table(table: ee.FeatureCollection, selectors: list[str]) -> pd.DataFrame:
    """Download a server-side table without the 5,000-feature getInfo limit."""
    params = {
        "table": table,
        "format": "CSV",
        "filename": "urban_heat_coimbatore_2022_2025",
    }
    if selectors:
        params["selectors"] = ",".join(selectors)
    download_id = ee.data.getTableDownloadId(params)
    response = AuthorizedSession(ee.data._get_state().credentials).get(
        ee.data.makeTableDownloadUrl(download_id)
    )
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate and report the extracted dataset without silently dropping rows."""
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    if len(frame) == 0:
        raise ValueError("The extracted dataset contains no grid cells.")
    if frame["cell_id"].duplicated().any():
        raise ValueError("The extracted dataset contains duplicate cell IDs.")

    coordinates_valid = frame[["latitude", "longitude"]].notna().all(axis=1)
    if not coordinates_valid.all():
        raise ValueError("Some grid cells have invalid latitude/longitude values.")

    for column, lower, upper in (
        ("lst_median_c", -50, 70),
        ("lst_p90_c", -50, 80),
        ("ndvi_median", -1.01, 1.01),
        ("ndbi_median", -1.01, 1.01),
    ):
        values = frame[column].dropna()
        if not values.empty and ((values < lower) | (values > upper)).any():
            raise ValueError(f"{column} contains values outside the plausible range.")

    print(f"Rows/grid cells: {len(frame)}")
    print("Missing values:")
    print(frame[EXPECTED_COLUMNS].isna().sum().to_string())
    for column in ("lst_median_c", "lst_p90_c", "ndvi_median", "ndbi_median"):
        values = frame[column].dropna()
        if values.empty:
            print(f"{column}: no valid observations")
        else:
            print(
                f"{column}: min={values.min():.4f}, max={values.max():.4f}, "
                f"median={values.median():.4f}"
            )


def write_metadata(study_area: ee.Geometry, row_count: int) -> None:
    metadata = {
        "city": CITY,
        "study_area": {
            "source": "FAO/GAUL/2015/level2",
            "filters": {
                "ADM0_NAME": "India",
                "ADM1_NAME": "Tamil Nadu",
                "ADM2_NAME": "Coimbatore",
            },
            "note": "Coimbatore district fallback; replace get_study_area with an official municipal GeoJSON when available.",
            "geometry_type": study_area.getInfo()["type"],
        },
        "grid_size_meters": GRID_SIZE_METERS,
        "landsat_datasets": ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"],
        "date_range": {"start_month_day": START_DATE, "end_month_day_exclusive": END_DATE},
        "years": list(YEARS),
        "preprocessing": {
            "qa_pixel_mask_bits": {
                "dilated_cloud": 1,
                "cirrus": 2,
                "cloud": 3,
                "cloud_shadow": 4,
                "snow": 5,
            },
            "optical_scaling": "SR_B* * 0.0000275 - 0.2",
            "temperature_scaling": "ST_B10 * 0.00341802 + 149.0 K, then K - 273.15",
            "invalid_temperature_observations": "masked using the ST_B10 image mask",
        },
        "feature_definitions": {
            "lst_median_c": "median seasonal LST in degrees Celsius",
            "lst_p90_c": "90th percentile seasonal LST in degrees Celsius",
            "ndvi_median": "median (SR_B5 - SR_B4) / (SR_B5 + SR_B4)",
            "ndbi_median": "median (SR_B6 - SR_B5) / (SR_B6 + SR_B5)",
            "valid_pixel_count": "count of valid Landsat LST observations used per cell",
        },
        "row_count": row_count,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    initialize_earth_engine()
    study_area = get_study_area()
    grid = build_grid(study_area)
    collection = get_landsat_collection(study_area)
    composite = create_composite(collection)
    frame = reduce_grid(composite, grid)
    validate_dataset(frame)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    write_metadata(study_area, len(frame))
    print(f"Wrote CSV: {OUTPUT_PATH}")
    print(f"Wrote metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()