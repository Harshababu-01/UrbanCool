import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        statistics = cls.client.get("/api/statistics")
        if statistics.status_code != 200:
            raise RuntimeError("Initialize data/processed/urbancool.db before running API tests.")
        cls.cell_id = cls.client.get("/api/cells", params={"limit": 1}).json()[0]["cell_id"]

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "UrbanCool API"})

    def test_cells_returns_data(self):
        response = self.client.get("/api/cells", params={"limit": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 5)

    def test_high_risk_filter(self):
        response = self.client.get("/api/cells", params={"risk_class": "High", "limit": 20})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())
        self.assertTrue(all(item["risk"]["class"] == "High" for item in response.json()))

    def test_cell_detail(self):
        response = self.client.get(f"/api/cells/{self.cell_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["cell_id"], self.cell_id)
        self.assertIn("metrics", payload)
        self.assertEqual(len(payload["recommendations"]), 3)

    def test_invalid_cell_returns_404(self):
        response = self.client.get("/api/cells/UC_DOES_NOT_EXIST")
        self.assertEqual(response.status_code, 404)

    def test_recommendations_are_ranked(self):
        response = self.client.get(f"/api/cells/{self.cell_id}/recommendations")
        self.assertEqual(response.status_code, 200)
        ranks = [item["rank"] for item in response.json()]
        self.assertEqual(ranks, [1, 2, 3])

    def test_statistics_counts(self):
        response = self.client.get("/api/statistics")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_cells"], 31676)
        self.assertEqual(
            payload["low_risk_cells"] + payload["medium_risk_cells"] + payload["high_risk_cells"],
            payload["total_cells"],
        )


if __name__ == "__main__":
    unittest.main()