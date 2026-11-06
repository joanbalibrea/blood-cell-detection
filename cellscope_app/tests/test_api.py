"""Run with: .venv/bin/python -m unittest cellscope_app.tests.test_api -v"""
import base64
from collections import Counter
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from cellscope_app.inference import COLORS, ROOT, decode_image
from cellscope_app.main import app, detector, MAX_REQUEST_BYTES

SAMPLE = ROOT.parent / "TXL-PBC/images/test/03967e3a34d05f4583e6d361f2c333a0.png"


class CellScopeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.image_bytes = SAMPLE.read_bytes()
        cls.prediction = cls.upload(0.25).json()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    @classmethod
    def upload(cls, confidence=0.25, data=None, name="smear.png"):
        return cls.client.post(f"/api/analyze?confidence={confidence}",
                               files={"file": (name, cls.image_bytes if data is None else data)})

    def test_health_and_actual_class_mapping(self):
        status = self.client.get("/api/health").json()
        self.assertTrue(status["ready"], status.get("error"))
        self.assertEqual(status["device"], "cpu")
        self.assertEqual(status["classes"], {"0": "WBC", "1": "RBC", "2": "Platelets"})
        self.assertEqual(status["imgsz"], 640)

    def test_real_detections_counts_confidences_and_coordinates(self):
        data = self.prediction
        self.assertIn("detections", data)
        with Image.open(io.BytesIO(self.image_bytes)) as image:
            self.assertEqual(data["image"], {"width": image.width, "height": image.height})
        detected = Counter(d["class_name"] for d in data["detections"])
        self.assertGreater(detected["RBC"], 1)
        self.assertGreater(detected["WBC"], 0)
        self.assertGreater(detected["Platelets"], 0)
        self.assertEqual(data["total_count"], len(data["detections"]))
        self.assertEqual(data["counts"], dict(detected))
        self.assertEqual([data["rbc_count"], data["wbc_count"], data["platelet_count"]],
                         [detected["RBC"], detected["WBC"], detected["Platelets"]])
        for detection in data["detections"]:
            self.assertGreaterEqual(detection["confidence"], 0.25)
            self.assertLessEqual(detection["confidence"], 1)
            self.assertEqual(detector.names[detection["class_id"]], detection["class_name"])
            x1, y1, x2, y2 = detection["bbox"].values()
            self.assertTrue(0 <= x1 < x2 <= data["image"]["width"])
            self.assertTrue(0 <= y1 < y2 <= data["image"]["height"])

    def test_annotation_png_dimensions_and_box_colors(self):
        data = self.prediction
        png = base64.b64decode(data["annotated_image"].split(",", 1)[1])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        image = Image.open(io.BytesIO(png)).convert("RGB")
        self.assertEqual(image.size, (data["image"]["width"], data["image"]["height"]))
        colors = set(image.get_flattened_data())
        for value in COLORS.values():
            self.assertIn(tuple(bytes.fromhex(value[1:])), colors)

    def test_confidence_threshold_recalculates_detections(self):
        lower = self.upload(0.05).json()
        higher = self.upload(0.95).json()
        self.assertTrue(all(d["confidence"] >= 0.95 for d in higher["detections"]))
        self.assertGreater(lower["total_count"], higher["total_count"])

    def test_invalid_empty_and_disguised_uploads(self):
        for data, name, expected in [(b"", "empty.png", 400), (b"not a picture", "fake.png", 400),
                                     (b"text", "fake.txt", 415)]:
            with self.subTest(name=name):
                response = self.upload(data=data, name=name)
                self.assertEqual(response.status_code, expected)
                self.assertIsInstance(response.json()["detail"], str)
        gif = io.BytesIO()
        Image.new("RGB", (10, 10)).save(gif, "GIF")
        self.assertEqual(self.upload(data=gif.getvalue()).status_code, 400)

    def test_large_request_and_invalid_confidence(self):
        self.assertEqual(self.upload(data=b"x" * (MAX_REQUEST_BYTES + 1)).status_code, 413)
        self.assertEqual(self.client.post("/api/analyze", content=iter([b"x" * 1_048_576] * 13),
                                         headers={"Content-Type": "application/octet-stream"}).status_code, 413)
        for threshold in [0, 1, "nan", "bad"]:
            self.assertEqual(self.upload(threshold).status_code, 422)

    def test_jpeg_orientation_rgba_and_pixel_limit(self):
        jpeg = io.BytesIO()
        source = Image.new("RGB", (30, 20), "red")
        exif = Image.Exif(); exif[274] = 6
        source.save(jpeg, "JPEG", exif=exif)
        self.assertEqual(decode_image(jpeg.getvalue()).size, (20, 30))
        rgba = io.BytesIO(); Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(rgba, "PNG")
        self.assertEqual(decode_image(rgba.getvalue()).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.upload(data=jpeg.getvalue(), name="sample.jpeg").status_code, 200)
        self.assertEqual(self.upload(data=rgba.getvalue()).status_code, 200)
        large = io.BytesIO(); Image.new("L", (4001, 4000)).save(large, "PNG")
        self.assertEqual(self.upload(data=large.getvalue()).status_code, 400)

    def test_static_frontend_and_private_paths(self):
        self.assertIn("CellScope — Integrated Blood Cell Analysis", self.client.get("/").text)
        for asset in ["app.js", "style.css"]:
            self.assertEqual(self.client.get("/static/" + asset).status_code, 200)
        for path in ["/.env", "/models/best.pt", "/dashboard/index.html", "/README.md"]:
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.get("/").headers["cache-control"], "no-store")

    def test_unavailable_model_returns_helpful_error(self):
        with patch.object(detector, "model", None), patch.object(detector, "error", "Checkpoint unavailable; restart."):
            response = self.upload()
            self.assertEqual(response.status_code, 503)
            self.assertIn("restart", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
