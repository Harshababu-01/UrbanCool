import unittest
from pathlib import Path

import pandas as pd

from backend.app.services.recommendation_engine import (
    calculate_ranges,
    calculate_thresholds,
    generate_recommendations,
    load_processed_dataset,
)


class RecommendationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_processed_dataset()
        cls.thresholds = calculate_thresholds(cls.frame)
        cls.ranges = calculate_ranges(cls.frame)
        cls.base = cls.frame.iloc[0].to_dict()

    def make_cell(self, **updates):
        cell = self.base.copy()
        cell.update(updates)
        return cell

    def test_high_heat_low_ndvi_prioritizes_vegetation(self):
        cell = self.make_cell(
            lst_median_c=self.ranges["lst_median_c"]["maximum"],
            baseline_heat_risk_score=self.ranges["baseline_heat_risk_score"]["maximum"],
            ndvi_median=self.ranges["ndvi_median"]["minimum"],
            ndbi_median=self.ranges["ndbi_median"]["minimum"],
        )
        recommendations = generate_recommendations(cell, self.thresholds, self.ranges)
        self.assertEqual(recommendations[0]["intervention"], "Tree / Vegetation Cover")
        self.assertIn(recommendations[0]["priority"], {"Medium", "High"})

    def test_high_heat_high_ndbi_prioritizes_roofs(self):
        cell = self.make_cell(
            lst_median_c=self.thresholds["high_heat_lst"] + 1,
            baseline_heat_risk_score=self.thresholds["high_heat_baseline_score"] + 5,
            ndvi_median=self.ranges["ndvi_median"]["maximum"],
            ndbi_median=self.ranges["ndbi_median"]["maximum"],
        )
        recommendations = generate_recommendations(cell, self.thresholds, self.ranges)
        self.assertEqual(recommendations[0]["intervention"], "Cool / Reflective Roofs")

    def test_moderate_heat_still_returns_ordered_recommendations(self):
        cell = self.make_cell(
            lst_median_c=self.frame["lst_median_c"].median(),
            baseline_heat_risk_score=self.frame["baseline_heat_risk_score"].median(),
            ndvi_median=self.frame["ndvi_median"].median(),
            ndbi_median=self.frame["ndbi_median"].median(),
        )
        recommendations = generate_recommendations(cell, self.thresholds, self.ranges)
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(
            [item["score"] for item in recommendations],
            sorted([item["score"] for item in recommendations], reverse=True),
        )

    def test_low_heat_receives_lower_priorities(self):
        cell = self.make_cell(
            lst_median_c=self.ranges["lst_median_c"]["minimum"],
            baseline_heat_risk_score=self.ranges["baseline_heat_risk_score"]["minimum"],
            ndvi_median=self.ranges["ndvi_median"]["maximum"],
            ndbi_median=self.ranges["ndbi_median"]["minimum"],
        )
        recommendations = generate_recommendations(cell, self.thresholds, self.ranges)
        self.assertTrue(all(item["score"] <= 33.0 for item in recommendations))


if __name__ == "__main__":
    unittest.main()