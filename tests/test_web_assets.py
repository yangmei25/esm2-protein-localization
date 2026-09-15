import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WebAssetTests(unittest.TestCase):
    def test_demo_assets_exist(self):
        for relative_path in ("web/index.html", "web/styles.css", "web/app.js"):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_demo_calls_prediction_endpoint(self):
        javascript = (ROOT / "web/app.js").read_text(encoding="utf-8")
        self.assertIn("/v1/predict", javascript)

    def test_demo_handles_non_json_server_errors(self):
        javascript = (ROOT / "web/app.js").read_text(encoding="utf-8")
        self.assertIn("response.text()", javascript)
        self.assertIn("Check the service terminal", javascript)

    def test_demo_has_sequence_input(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        self.assertIn('id="sequence"', html)

    def test_demo_displays_runtime_model_identity(self):
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "web/app.js").read_text(encoding="utf-8")
        for element_id in ("model-name", "checkpoint-epoch", "device"):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("payload.model_name", javascript)


if __name__ == "__main__":
    unittest.main()
