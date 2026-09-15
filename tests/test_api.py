import unittest

try:
    from fastapi.testclient import TestClient
    from api.main import app, require_predictor
except ModuleNotFoundError:  # API dependencies are installed from requirements-api.txt.
    TestClient = None


class FakePredictor:
    def predict(self, sequence: str, protein_id: str, **_: object) -> dict:
        if "*" in sequence:
            raise ValueError("Invalid amino-acid symbols: ['*']")
        probability = 0.8
        return {
            "protein_id": protein_id,
            "sequence_length": len(sequence),
            "predicted_label": "membrane",
            "membrane_probability": probability,
            "soluble_probability": 1 - probability,
            "threshold": 0.5,
            "model_name": "fake-esm2",
            "checkpoint_epoch": 3,
            "device": "cpu",
            "long_sequence_mode": False,
            "aggregation": "maximum_window_probability",
            "window_size": 1022,
            "window_stride": 894,
            "number_of_windows": 1,
            "top_window": {"window_index": 1, "start": 1, "end": len(sequence), "membrane_probability": probability},
            "windows": [{"window_index": 1, "start": 1, "end": len(sequence), "membrane_probability": probability}],
        }


@unittest.skipIf(TestClient is None, "Install requirements-api.txt")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.dependency_overrides[require_predictor] = lambda: FakePredictor()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()

    def test_service_info(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "/v1/predict")

    def test_prediction_contract(self) -> None:
        response = self.client.post("/v1/predict", json={"protein_id": "P1", "sequence": "ACDE"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_label"], "membrane")
        self.assertAlmostEqual(response.json()["membrane_probability"], 0.8)

    def test_empty_sequence_is_rejected_by_schema(self) -> None:
        response = self.client.post("/v1/predict", json={"sequence": ""})
        self.assertEqual(response.status_code, 422)

    def test_predictor_validation_error_is_422(self) -> None:
        response = self.client.post("/v1/predict", json={"sequence": "ACD*"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Invalid amino-acid", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
