"""Real local-checkpoint integration tests plus explicit review-policy edge cases.

Run: .venv/bin/python -m unittest discover -s cellscope_app/tests -v
Synthetic predictions below exercise error/policy branches only, never the app's
normal inference path or any reported model performance.
"""
import base64
from collections import Counter
import csv
import io
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import torch

from bloodmnist_classifier.training.common import CLASSES, transforms_for
from cellscope_app.classification import WBC_CLASSES
from cellscope_app.inference import ROOT, decode_image
from cellscope_app.main import app, classifier, detector, pipeline
from cellscope_app.pipeline import crop_wbc, review_prediction, subtype_breakdown

SAMPLES = ROOT.parent / "TXL-PBC/images/test"
PRIMARY = SAMPLES / "03967e3a34d05f4583e6d361f2c333a0.png"


class IntegratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        response = cls.upload(PRIMARY.read_bytes())
        assert response.status_code == 200, response.text
        cls.report = response.json()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    @classmethod
    def upload(cls, data, query=""):
        return cls.client.post("/api/analyze?mode=integrated" + query,
                               files={"file": ("smear.png", data, "image/png")})

    def assert_consistent(self, report):
        self.assertEqual(report["total_count"], len(report["detections"]))
        self.assertEqual(report["wbc_count"], report["wbc_breakdown"]["total"])
        self.assertEqual(sum(report["wbc_breakdown"]["counts"].values()), report["wbc_count"])
        counts = Counter(d["class_name"] for d in report["detections"])
        for name, count in report["counts"].items():
            self.assertEqual(count, counts[name])
        for d in report["detections"]:
            c = d["classification"]
            self.assertEqual(detector.names[d["class_id"]], d["class_name"])
            if d["class_name"] != "WBC":
                self.assertEqual(c["status"], "not_applicable")
                self.assertNotIn("crop", d)
                continue
            self.assertIn(c["status"], {"predicted", "uncertain", "disagreement"})
            p = c["probabilities"]
            self.assertEqual(list(p), CLASSES)
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()))
            self.assertAlmostEqual(sum(p.values()), 1, places=5)
            self.assertEqual(c["predicted_class"], CLASSES[c["class_id"]])
            self.assertEqual(c["predicted_class"], max(p, key=p.get))
            self.assertEqual(c["confidence"], p[c["predicted_class"]])
            x1, y1, x2, y2 = d["crop"]["bbox"].values()
            self.assertTrue(0 <= x1 < x2 <= report["image"]["width"])
            self.assertTrue(0 <= y1 < y2 <= report["image"]["height"])
            with Image.open(io.BytesIO(base64.b64decode(d["crop"]["image"].split(",")[1]))) as image:
                self.assertEqual(image.width, image.height)
                self.assertLessEqual(image.width, 192)

    def test_real_metadata_and_exact_preprocessing(self):
        health = self.client.get("/api/health").json()
        self.assertTrue(health["integrated_ready"], health)
        c = health["classifier"]
        self.assertEqual(c["classes"], CLASSES)
        self.assertEqual(c["best_epoch"], 25)
        self.assertEqual(c["preprocessing"]["image_size"], 224)
        self.assertEqual(c["preprocessing"]["resize_size"], 256)
        self.assertEqual(c["preprocessing"]["mean"], [.485, .456, .406])
        image = decode_image(PRIMARY.read_bytes())
        wbc = next(d for d in self.report["detections"] if d["class_name"] == "WBC")
        crop, _ = crop_wbc(image, wbc["bbox"])
        actual = classifier.transform(crop)
        self.assertEqual(list(actual.shape), [3, 224, 224])
        self.assertTrue(torch.equal(actual, transforms_for(classifier.config)(crop)))
        red = classifier.transform(Image.new("RGB", (80, 80), "red"))
        expected = torch.tensor([(1-.485)/.229, (0-.456)/.224, (0-.406)/.225])
        self.assertTrue(torch.allclose(red[:, 100, 100], expected))
        batch = classifier.predict_batch([crop, crop])
        single = classifier.predict_batch([crop])[0]
        for prediction in batch:
            self.assertEqual(prediction["predicted_class"], single["predicted_class"])
            for key in CLASSES:
                self.assertAlmostEqual(prediction["probabilities"][key], single["probabilities"][key], places=5)

    def test_real_smears_and_counts(self):
        self.assert_consistent(self.report)
        paths = [SAMPLES / "d0410b5b65fd577885b711e70e21c3e9.png", next(iter(sorted(SAMPLES.glob("*.png"))))]
        for path in paths:
            with self.subTest(image=path.name):
                response = self.upload(path.read_bytes())
                self.assertEqual(response.status_code, 200, response.text)
                self.assert_consistent(response.json())

    def test_png_csv_json_agree(self):
        report = self.report
        rows = list(csv.DictReader(io.StringIO(report["csv"])))
        self.assertEqual(len(rows), report["total_count"])
        for row, d in zip(rows, report["detections"]):
            self.assertEqual(int(row["detection_number"]), d["number"])
            self.assertEqual(row["yolo_class"], d["class_name"])
            self.assertEqual(float(row["yolo_confidence"]), d["confidence"])
            for coordinate in d["bbox"]:
                self.assertEqual(float(row[coordinate]), d["bbox"][coordinate])
            if d["class_name"] == "WBC":
                self.assertEqual(row["classifier_prediction"], d["classification"]["predicted_class"])
                self.assertEqual(float(row["classifier_confidence"]), d["classification"]["confidence"])
                for name in CLASSES:
                    self.assertEqual(float(row["probability_" + name]), d["classification"]["probabilities"][name])
        png = base64.b64decode(report["annotated_image"].split(",")[1])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.size, (report["image"]["width"], report["image"]["height"]))

    def test_multiple_real_cells_are_classified_in_one_batch(self):
        # A technical montage of real smear pixels, not an accuracy benchmark.
        source = decode_image(PRIMARY.read_bytes())
        montage = Image.new("RGB", (source.width * 2, source.height * 2), "white")
        for x, y in [(0, 0), (source.width, 0), (0, source.height), source.size]:
            montage.paste(source, (x, y))
        data = io.BytesIO(); montage.save(data, "PNG")
        with patch.object(classifier, "predict_batch", wraps=classifier.predict_batch) as predict:
            response = self.upload(data.getvalue())
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertGreaterEqual(report["wbc_count"], 2)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(len(predict.call_args.args[0]), report["wbc_count"])
        self.assert_consistent(report)

    def test_detection_only_preserves_counts_and_skips_classifier(self):
        with patch.object(classifier, "predict_batch", side_effect=AssertionError("Must not classify")):
            r = self.client.post("/api/analyze?mode=detection", files={"file": ("smear.png", PRIMARY.read_bytes())})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["counts"], self.report["counts"])
        self.assertEqual(r.json()["wbc_breakdown"]["counts"]["not_run"], self.report["wbc_count"])

    def test_real_uncertainty_and_no_wbc(self):
        r = self.upload(PRIMARY.read_bytes(), "&subtype_confidence=1").json()
        for d in r["detections"]:
            c = d["classification"]
            if d["class_name"] == "WBC" and c["confidence"] < 1:
                self.assertTrue(c["low_confidence"])
                self.assertTrue(c["review_required"])
        data = io.BytesIO(); Image.new("RGB", (100, 100), "white").save(data, "PNG")
        r = self.upload(data.getvalue()).json()
        self.assertEqual(r["wbc_count"], 0)
        self.assertEqual(sum(r["wbc_breakdown"]["counts"].values()), 0)

    def test_bad_settings_and_classifier_unavailable(self):
        for query in ["&subtype_confidence=nan", "&subtype_confidence=1.1", "&padding=-0.1", "&padding=nan", "&mode=other"]:
            self.assertEqual(self.upload(PRIMARY.read_bytes(), query).status_code, 422)
        with patch.object(classifier, "model", None), patch.object(classifier, "error", "Classifier unavailable; restart."):
            self.assertEqual(self.upload(PRIMARY.read_bytes()).status_code, 503)
            r = self.client.post("/api/analyze", files={"file": ("smear.png", PRIMARY.read_bytes())})
            self.assertEqual(r.status_code, 200)

    def test_classification_failure_is_reported_without_losing_detections(self):
        with patch.object(classifier, "predict_batch", side_effect=RuntimeError("Test failure")):
            r = self.upload(PRIMARY.read_bytes()).json()
        self.assertEqual(r["counts"], self.report["counts"])
        self.assertEqual(r["wbc_breakdown"]["counts"]["failed"], r["wbc_count"])
        self.assertTrue(all(d["classification"]["review_required"] for d in r["detections"] if d["class_name"] == "WBC"))


class CropAndPolicyTests(unittest.TestCase):
    def test_boundary_clipping_padding_and_invalid_crops(self):
        image = Image.new("RGB", (100, 70), "red")
        box = dict(x_min=-10., y_min=-4., x_max=35., y_max=48.)
        crop, meta = crop_wbc(image, box)
        self.assertEqual(crop.width, crop.height)
        self.assertTrue(meta["white_edge_padding"])
        self.assertEqual(crop.getpixel((crop.width // 2, crop.height // 2)), (255, 0, 0))
        self.assertEqual(box["x_min"], -10.)
        thin = Image.new("RGB", (16000, 1), "red")
        bounded, metadata = crop_wbc(thin, dict(x_min=0, y_min=0, x_max=16000, y_max=1))
        self.assertEqual(bounded.size, (1024, 1024))
        self.assertGreater(metadata["square_size"], metadata["inference_crop_size"])
        for values in [(10, 10, 10, 15), (30, 40, 10, 5), (101, 2, 110, 6), (0, 0, float("nan"), 30)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                crop_wbc(image, dict(zip(box, values)))

    def test_disagreement_preserves_all_eight_probabilities_and_reconciles(self):
        detections = []
        for i, name in enumerate(CLASSES):
            p = {c: .04 for c in CLASSES}; p[name] = .72
            original = {"class_id": i, "predicted_class": name, "confidence": .72, "probabilities": p}
            reviewed = review_prediction(original, .8)
            self.assertEqual(reviewed["predicted_class"], name)
            self.assertEqual(reviewed["probabilities"], p)
            self.assertTrue(reviewed["low_confidence"])
            if name in {"platelet", "erythroblast"}:
                self.assertEqual(reviewed["status"], "disagreement")
            else:
                self.assertEqual(reviewed["status"], "uncertain")
            detections.append({"class_name": "WBC", "classification": reviewed})
        breakdown = subtype_breakdown(detections)
        self.assertEqual(breakdown["total"], 8)
        self.assertEqual(breakdown["counts"]["disagreement"], 2)
        self.assertEqual(breakdown["counts"]["uncertain"], 6)
        self.assertEqual(breakdown["disagreement_predictions"], {"erythroblast": 1, "platelet": 1})

    def test_invalid_crop_remains_in_pipeline_counts(self):
        fake = {"counts": {"WBC": 1, "RBC": 0, "Platelets": 0}, "total_count": 1,
                "wbc_count": 1, "elapsed_ms": 0, "detections": [{"number": 1, "class_name": "WBC",
                "confidence": .9, "class_id": 0, "bbox": dict(x_min=10, y_min=10, x_max=10, y_max=10)}]}
        with patch.object(detector, "analyze", return_value=fake), patch.object(classifier, "predict_batch") as predict:
            r = pipeline.analyze(Image.new("RGB", (30, 30)), .25)
        predict.assert_not_called()
        self.assertEqual(r["wbc_count"], 1)
        self.assertEqual(r["wbc_breakdown"]["counts"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
