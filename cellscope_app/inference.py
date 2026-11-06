"""Load the supplied checkpoint once; run serialized, offline CPU predictions."""
import base64
import io
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import warnings
import zipfile

# Set before importing Ultralytics: inference must never install or fetch assets.
os.environ["YOLO_AUTOINSTALL"] = "False"
os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_VERBOSE"] = "False"

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

# Keep Pillow's decoder before Ultralytics patches Image.open with a HEIF fallback.
# Invalid JPEG/PNG uploads must not trigger optional plugin installation/imports.
_PIL_OPEN = Image.open

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "best.pt"
MAX_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
COLORS = {"RBC": "#f3ad62", "WBC": "#b39bff", "Platelets": "#6ce3a2"}
LOG = logging.getLogger(__name__)


def decode_image(data: bytes) -> Image.Image:
    """Validate contents rather than trusting filename or client MIME type."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _PIL_OPEN(io.BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ValueError("Upload a single image, not an animated PNG.")
                if probe.width * probe.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel limit.")
                probe.verify()
            with _PIL_OPEN(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                return Image.alpha_composite(background, image).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Unreadable image. Upload a valid JPEG or PNG, up to 16 megapixels.") from exc


def png_data(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def annotate_image(image, detections):
    """Number every box; put long WBC subtype labels on a second line."""
    annotated = image.copy()
    width, height = image.size
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(10, min(24, round(min(width, height) / 36))))
    line_width = max(2, round(min(width, height) / 250))
    # Draw WBC labels last so crowded RBC labels do not hide subtype predictions.
    for detection in sorted(detections, key=lambda d: d["class_name"] == "WBC"):
        box = list(detection["bbox"].values())
        color = COLORS[detection["class_name"]]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        draw.rectangle(box, outline=color, width=line_width)
        label = f"#{detection['number']} {detection['class_name']} D:{detection['confidence']:.2f}"
        classification = detection.get("classification", {})
        if classification.get("predicted_class"):
            subtype = classification["predicted_class"].replace("immature_granulocytes", "Imm. granulocytes").replace("_", " ")
            label += f"\n{subtype} C:{classification['confidence']:.2f}"
            if classification.get("review_required"):
                label += " [review]"
        text_box = draw.multiline_textbbox((0, 0), label, font=font, spacing=2)
        label_width, label_height = text_box[2] + 8, text_box[3] + 6
        left = max(0, min(box[0], width - label_width))
        top = max(0, min(box[1] - label_height, height - label_height))
        draw.rectangle((left, top, min(width, left + label_width), min(height, top + label_height)), fill=color)
        draw.multiline_text((left + 4, top + 1), label, fill="#0d131c", font=font, spacing=2)
    return annotated


class Detector:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.error = None
        self.warning = None
        self.source = "models/best.pt"
        self.names = {}
        self.imgsz = 640
        self.end2end = False

    def load(self):
        if self.model is not None:
            return
        try:
            if not WEIGHTS.is_file():
                raise FileNotFoundError("Missing models/best.pt. Restore your trained checkpoint and restart.")
            try:
                model = YOLO(str(WEIGHTS), task="detect")
            except (RuntimeError, TypeError, EOFError, zipfile.BadZipFile, OSError) as exc:
                # The supplied ZIP was damaged by text encoding, but its separately
                # extracted PyTorch records are intact. Repackage those exact bytes;
                # never edit either source, download weights, or use another model.
                extracted = WEIGHTS.with_suffix("")
                required = ("data.pkl", "version", "byteorder", "data")
                if not all((extracted / part).exists() for part in required):
                    raise RuntimeError("models/best.pt is unreadable. Restore a fresh binary copy of your trained checkpoint.") from exc
                with tempfile.TemporaryDirectory(prefix="cellscope-checkpoint-") as folder:
                    archive = Path(folder) / "best.pt"
                    with zipfile.ZipFile(archive, "w", strict_timestamps=False) as target:
                        for path in sorted(extracted.rglob("*")):
                            if path.is_file() and path.name != ".DS_Store":
                                target.write(path, "best/" + path.relative_to(extracted).as_posix())
                    model = YOLO(str(archive), task="detect")
                self.source = "models/best/ (existing extracted checkpoint)"
                self.warning = (
                    "models/best.pt is unreadable. Using the existing extracted trained checkpoint "
                    "in models/best/. Original model files are unchanged. Restore a fresh best.pt "
                    "and restart to load it directly."
                )
                LOG.warning(self.warning)
            self.names = {int(key): str(value) for key, value in model.names.items()}
            if self.names != {0: "WBC", 1: "RBC", 2: "Platelets"}:
                raise RuntimeError(f"Unexpected checkpoint classes: {self.names}. Expected WBC, RBC, Platelets; no remapping was applied.")
            self.imgsz = int(model.ckpt.get("train_args", {}).get("imgsz", 640))
            self.end2end = bool(getattr(model.model.model[-1], "end2end", False))
            # Warm the predictor on CPU, so even the first upload uses ready weights.
            model.predict(Image.new("RGB", (64, 64), "white"), device="cpu",
                          imgsz=self.imgsz, conf=0.25, verbose=False, save=False)
            self.model = model
        except Exception as exc:
            LOG.exception("CellScope checkpoint could not be loaded")
            self.error = f"Model unavailable: {exc}"

    def status(self):
        return {"ready": self.model is not None, "device": "cpu", "classes": self.names,
                "model": "YOLO26s", "weights": self.source, "imgsz": self.imgsz,
                "end2end": self.end2end, "warning": self.warning, "error": self.error,
                "max_upload_mb": 12, "max_pixels": MAX_PIXELS}

    def analyze(self, image: Image.Image, confidence: float):
        with self.lock:
            started = time.perf_counter()
            result = self.model.predict(image, conf=confidence, imgsz=self.imgsz,
                                        device="cpu", verbose=False, save=False)[0]
            # Keep Ultralytics' own detection-head postprocessing. No custom NMS.
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
        width, height = image.size
        detections = []
        counts = {name: 0 for name in self.names.values()}
        for index, (box, score, cls) in enumerate(zip(boxes, scores, classes), 1):
            class_id = int(cls)
            name = self.names[class_id]
            x1, y1, x2, y2 = box
            box = [max(0.0, min(float(x1), width)), max(0.0, min(float(y1), height)),
                   max(0.0, min(float(x2), width)), max(0.0, min(float(y2), height))]
            counts[name] += 1
            detections.append({"number": index, "class_id": class_id, "class_name": name,
                               "confidence": float(score), "bbox": dict(zip(
                                   ("x_min", "y_min", "x_max", "y_max"), box))})
        return {"image": {"width": width, "height": height}, "total_count": len(detections),
                "counts": counts, "rbc_count": counts["RBC"], "wbc_count": counts["WBC"],
                "platelet_count": counts["Platelets"], "confidence_threshold": confidence,
                "detections": detections, "annotated_image": png_data(annotate_image(image, detections)),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "model": self.status(), "coordinate_system": "Original image pixels after EXIF orientation",
                "notice": "Object counts in this image, not clinical blood-cell concentrations."}
