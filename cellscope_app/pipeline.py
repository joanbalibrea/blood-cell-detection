"""Combine original YOLO detections with independently scored WBC subtypes."""
import csv
import io
import logging
import math
import time

from PIL import Image

from .classification import CLASSES, WBC_CLASSES
from .inference import annotate_image, png_data

DOMAIN_NOTICE = ("EfficientNet was trained on isolated BloodMNIST cells. Staining, scale and crop "
                 "differences can change predictions on YOLO crops. BloodMNIST test accuracy does "
                 "not measure this pipeline's subtype accuracy. Review crops; research use only.")


def crop_wbc(image, bbox, padding=0.12):
    """Square context crop; clip source pixels and fill missing edge context white.

    The square avoids distorting cells. Padding offsets the validation center crop;
    the original float detection box and YOLO counts are never changed.
    """
    values = [float(bbox[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
    if not all(math.isfinite(v) for v in values) or not 0 <= padding <= .25:
        raise ValueError("Invalid crop coordinates or padding.")
    x1, y1, x2, y2 = values
    x1, x2 = max(0., min(x1, image.width)), max(0., min(x2, image.width))
    y1, y2 = max(0., min(y1, image.height)), max(0., min(y2, image.height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty WBC bounding box after clipping to the image.")
    side = max(1, math.ceil(max(x2 - x1, y2 - y1) * (1 + 2 * padding)))
    left, top = math.floor((x1 + x2 - side) / 2), math.floor((y1 + y2 - side) / 2)
    bounds = (max(0, left), max(0, top), min(image.width, left + side), min(image.height, top + side))
    # Bound working memory even for an unusually wide, thin uploaded image.
    working_size = min(side, 1024)
    scale = working_size / side
    pixels = image.crop(bounds)
    if scale < 1:
        pixels = pixels.resize((max(1, round(pixels.width * scale)), max(1, round(pixels.height * scale))),
                               Image.Resampling.LANCZOS)
    crop = Image.new("RGB", (working_size, working_size), "white")
    crop.paste(pixels, (round((bounds[0] - left) * scale), round((bounds[1] - top) * scale)))
    return crop, {"bbox": dict(zip(("x_min", "y_min", "x_max", "y_max"), bounds)),
                  "square_size": side, "inference_crop_size": working_size, "padding_fraction": padding,
                  "white_edge_padding": bounds != (left, top, left + side, top + side)}


def review_prediction(prediction, threshold):
    disagreement = prediction["predicted_class"] in {"platelet", "erythroblast"}
    uncertain = prediction["confidence"] < threshold
    status = "disagreement" if disagreement else "uncertain" if uncertain else "predicted"
    warnings = []
    if disagreement:
        warnings.append("YOLO detected WBC; EfficientNet predicted " + prediction["predicted_class"] + ". Review both models.")
    if uncertain:
        warnings.append("Classifier confidence is below the selected subtype threshold.")
    return {**prediction, "status": status, "model_disagreement": disagreement,
            "low_confidence": uncertain, "review_required": disagreement or uncertain, "warnings": warnings}


def subtype_breakdown(detections):
    counts = {c: 0 for c in WBC_CLASSES}
    counts.update(uncertain=0, disagreement=0, failed=0, not_run=0)
    disagreement_predictions = {"erythroblast": 0, "platelet": 0}
    for detection in detections:
        if detection["class_name"] != "WBC":
            continue
        result = detection["classification"]
        key = result["predicted_class"] if result["status"] == "predicted" else result["status"]
        counts[key] += 1
        if result["status"] == "disagreement":
            disagreement_predictions[result["predicted_class"]] += 1
    return {"counts": counts, "total": sum(counts.values()),
            "disagreement_predictions": disagreement_predictions,
            "policy": "Exclusive categories: disagreement, then uncertain, then subtype; failed/not_run separate. "
                      "A disagreement can also have low_confidence=true without being counted twice."}


def detections_csv(report):
    output = io.StringIO(newline="")
    fields = ["detection_number", "yolo_class_id", "yolo_class", "yolo_confidence",
              "x_min", "y_min", "x_max", "y_max", "classifier_class_id", "classifier_prediction",
              "classifier_confidence", "status", "review_required", "model_disagreement", "low_confidence",
              "warnings"] + ["probability_" + c for c in CLASSES]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for d in report["detections"]:
        c = d["classification"]
        writer.writerow({"detection_number": d["number"], "yolo_class_id": d["class_id"],
                         "yolo_class": d["class_name"], "yolo_confidence": d["confidence"], **d["bbox"],
                         "classifier_class_id": c.get("class_id", ""),
                         "classifier_prediction": c.get("predicted_class", ""),
                         "classifier_confidence": c.get("confidence", ""), "status": c["status"],
                         "review_required": c["review_required"],
                         "model_disagreement": c.get("model_disagreement", False),
                         "low_confidence": c.get("low_confidence", False),
                         "warnings": " | ".join(c.get("warnings", [])),
                         **{"probability_" + k: v for k, v in c.get("probabilities", {}).items()}})
    return output.getvalue()


class IntegratedPipeline:
    def __init__(self, detector, classifier):
        self.detector, self.classifier = detector, classifier

    def analyze(self, image, confidence, mode="integrated", subtype_confidence=.70, padding=.12):
        started = time.perf_counter()
        report = self.detector.analyze(image, confidence)
        detection_ms = report["elapsed_ms"]
        pending = []

        def flush():
            if not pending:
                return
            try:
                predictions = self.classifier.predict_batch([p[1] for p in pending])
                if len(predictions) != len(pending):
                    raise RuntimeError("Classifier result count differs from crop count.")
                for (detection, _), prediction in zip(pending, predictions):
                    detection["classification"] = review_prediction(prediction, subtype_confidence)
            except Exception:
                logging.getLogger(__name__).exception("WBC batch classification failed")
                for detection, _ in pending:
                    detection["classification"] = {"status": "failed", "review_required": True,
                                                    "warnings": ["Classification failed. Review crop or restart the server."]}
            pending.clear()

        tick = time.perf_counter()
        for d in report["detections"]:
            d["classification"] = {"status": "not_applicable", "review_required": False, "warnings": []}
            if d["class_name"] != "WBC":
                continue
            d["classification"] = {"status": "not_run", "review_required": False, "warnings": []}
            if mode == "detection":
                continue
            try:
                crop, metadata = crop_wbc(image, d["bbox"], padding)
                thumbnail = crop.copy()
                thumbnail.thumbnail((192, 192), Image.Resampling.LANCZOS)
                d["crop"] = {**metadata, "image": png_data(thumbnail)}
                pending.append((d, crop))
                if len(pending) == 16:
                    flush()
            except ValueError as exc:
                d["classification"] = {"status": "failed", "review_required": True, "warnings": [str(exc)]}
        flush()
        classification_ms = round((time.perf_counter() - tick) * 1000, 1)
        report.update(schema_version="2.0", mode=mode, subtype_confidence_threshold=subtype_confidence,
                      crop_padding=padding, classifier=self.classifier.status(),
                      wbc_breakdown=subtype_breakdown(report["detections"]), domain_notice=DOMAIN_NOTICE)
        if mode == "integrated":
            report["annotated_image"] = png_data(annotate_image(image, report["detections"]))
        report["timings_ms"] = {"detection": detection_ms, "classification_and_crops": classification_ms}
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        report["csv"] = detections_csv(report)
        return report
