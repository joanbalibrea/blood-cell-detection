"""Offline EfficientNet-B0 loading and batched, single-cell CPU inference."""
import logging
import threading

import torch

from bloodmnist_classifier.training.common import CLASSES, build_model, transforms_for
from .inference import ROOT

WEIGHTS = ROOT / "bloodmnist_classifier" / "models" / "best.pt"
WBC_CLASSES = tuple(c for c in CLASSES if c not in {"erythroblast", "platelet"})


class SubtypeClassifier:
    def __init__(self):
        self.model = None
        self.error = None
        self.config = {}
        self.best_epoch = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        try:
            payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            config = payload["config"]
            if config["architecture"] != "efficientnet_b0" or config["classes"] != CLASSES:
                raise ValueError("Incompatible EfficientNet architecture or class ordering.")
            if (config["image_size"] != 224 or config["resize_size"] != 256
                    or config["mean"] != [0.485, 0.456, 0.406]
                    or config["std"] != [0.229, 0.224, 0.225]):
                raise ValueError("Unexpected preprocessing metadata; verify the training configuration.")
            model = build_model(pretrained=False)  # Never download ImageNet weights.
            model.load_state_dict(payload["state_dict"], strict=True)
            model.cpu().eval()
            self.transform = transforms_for(config, training=False)
            self.config = config
            self.best_epoch = payload.get("best_epoch")
            with torch.inference_mode():
                model(torch.zeros(1, 3, config["image_size"], config["image_size"]))
            self.model = model
            self.error = None
        except Exception as exc:
            logging.getLogger(__name__).exception("EfficientNet checkpoint could not be loaded")
            self.error = f"Classifier unavailable: {exc}. Restore the trained checkpoint and restart."

    def status(self):
        return {"ready": self.model is not None, "device": "cpu", "error": self.error,
                "model": "EfficientNet-B0", "weights": WEIGHTS.relative_to(ROOT).as_posix(),
                "classes": CLASSES, "best_epoch": self.best_epoch,
                "preprocessing": {"color": "RGB", "resize_size": self.config.get("resize_size"),
                                  "image_size": self.config.get("image_size"),
                                  "interpolation": "bicubic", "crop": "center",
                                  "mean": self.config.get("mean"), "std": self.config.get("std")}}

    def predict_batch(self, images):
        if self.model is None:
            raise RuntimeError(self.error or "Classifier is not ready.")
        if not images:
            return []
        with self.lock, torch.inference_mode():
            batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
            probabilities = self.model(batch).softmax(1).cpu()
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Classifier returned nonfinite probabilities.")
        predictions = []
        for row in probabilities.tolist():
            order = sorted(range(len(CLASSES)), key=row.__getitem__, reverse=True)
            index = order[0]
            predictions.append({"class_id": index, "predicted_class": CLASSES[index],
                                "confidence": row[index], "probabilities": dict(zip(CLASSES, row)),
                                "top_predictions": [{"class_name": CLASSES[i], "confidence": row[i]}
                                                    for i in order[:3]]})
        return predictions
