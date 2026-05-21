"""
utils/detector.py
Load a YOLOv8 model and run inference, returning normalised detection dicts.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

# Ultralytics is imported lazily so Streamlit can still import this module
# even before the wheel is installed (helpful during cold-start).
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


_MODEL_CACHE: dict[str, "YOLO"] = {}

# Model weights are downloaded automatically by Ultralytics on first use.
_MODEL_FILES = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


def load_model(model_name: str = "yolov8n"):
    """Return a cached YOLO model instance."""
    if not _YOLO_AVAILABLE:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    if model_name not in _MODEL_CACHE:
        weights = _MODEL_FILES.get(model_name, "yolov8n.pt")
        _MODEL_CACHE[model_name] = YOLO(weights)
    return _MODEL_CACHE[model_name]


def run_detection(
    model,
    image_np: np.ndarray,
    conf: float = 0.40,
    iou: float = 0.45,
) -> list[dict]:
    """
    Run YOLO inference on a BGR or RGB numpy array.

    Returns a list of dicts:
        {
            "class":      str,          # COCO class name
            "class_id":   int,
            "confidence": float,
            "bbox_px":    [x1,y1,x2,y2] # absolute pixel coords
        }
    """
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed.")

    results = model.predict(
        source=image_np,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections: list[dict] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        names = result.names  # {id: name}
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            detections.append(
                {
                    "class":      names[cls_id],
                    "class_id":   cls_id,
                    "confidence": conf_val,
                    "bbox_px":    [x1, y1, x2, y2],
                }
            )

    return detections
