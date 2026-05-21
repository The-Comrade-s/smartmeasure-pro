"""
utils/reference.py
──────────────────
Precision reference-object database.

Each entry contains:
  width_cm / height_cm   — ISO / manufacturer spec (ground truth)
  width_std / height_std — 1-sigma population std-dev (cm)
                           used to weight multi-reference fusion
  yolo_class             — COCO class name that YOLO reports
  aspect_ratio           — width/height (used for orientation sanity check)
  depth_cm               — typical depth/thickness (for 3-D heuristics)
  reliability            — 0-1: how consistent the size is across instances
"""
from __future__ import annotations

REFERENCE_OBJECTS: dict[str, dict] = {

    # ── Payment / ID cards  (ISO/IEC 7810 ID-1 — exact) ──────────────────────
    "credit card": {
        "width_cm": 8.560, "height_cm": 5.398,
        "width_std": 0.02,  "height_std": 0.02,
        "yolo_class": None,   # YOLO doesn't detect cards natively
        "aspect_ratio": 1.586, "depth_cm": 0.076,
        "reliability": 0.99,
        "notes": "ISO/IEC 7810 ID-1 — most reliable reference available",
    },
    "A4 paper": {
        "width_cm": 21.000, "height_cm": 29.700,
        "width_std": 0.05,  "height_std": 0.05,
        "yolo_class": None,
        "aspect_ratio": 0.707, "depth_cm": 0.01,
        "reliability": 0.99,
        "notes": "ISO 216 — place flat, shoot from directly above",
    },
    "US Letter paper": {
        "width_cm": 21.590, "height_cm": 27.940,
        "width_std": 0.05,  "height_std": 0.05,
        "yolo_class": None,
        "aspect_ratio": 0.773, "depth_cm": 0.01,
        "reliability": 0.99,
        "notes": "ANSI A — place flat, shoot from directly above",
    },

    # ── Everyday objects (tight tolerances) ───────────────────────────────────
    "cell phone": {
        "width_cm": 7.15,  "height_cm": 15.10,
        "width_std": 0.60, "height_std": 1.20,
        "yolo_class": "cell phone",
        "aspect_ratio": 0.473, "depth_cm": 0.85,
        "reliability": 0.80,
        "notes": "Average modern smartphone — high variance across models",
    },
    "laptop": {
        "width_cm": 32.50, "height_cm": 22.50,
        "width_std": 2.50, "height_std": 1.50,
        "yolo_class": "laptop",
        "aspect_ratio": 1.444, "depth_cm": 1.8,
        "reliability": 0.72,
        "notes": "13–15\" laptops; 16\" outliers skew high",
    },
    "keyboard": {
        "width_cm": 44.00, "height_cm": 14.50,
        "width_std": 3.50, "height_std": 1.00,
        "yolo_class": "keyboard",
        "aspect_ratio": 3.034, "depth_cm": 2.5,
        "reliability": 0.70,
    },
    "mouse": {
        "width_cm": 6.20,  "height_cm": 11.50,
        "width_std": 0.60, "height_std": 1.00,
        "yolo_class": "mouse",
        "aspect_ratio": 0.539, "depth_cm": 3.8,
        "reliability": 0.75,
    },
    "book": {
        "width_cm": 15.20, "height_cm": 22.80,
        "width_std": 2.50, "height_std": 3.00,
        "yolo_class": "book",
        "aspect_ratio": 0.667, "depth_cm": 2.0,
        "reliability": 0.65,
        "notes": "Paperback average; hardbacks are larger",
    },
    "cup": {
        "width_cm": 8.20,  "height_cm": 9.50,
        "width_std": 1.20, "height_std": 1.50,
        "yolo_class": "cup",
        "aspect_ratio": 0.863, "depth_cm": 8.2,
        "reliability": 0.65,
    },
    "bottle": {
        "width_cm": 7.00,  "height_cm": 25.00,
        "width_std": 1.00, "height_std": 4.00,
        "yolo_class": "bottle",
        "aspect_ratio": 0.280, "depth_cm": 7.0,
        "reliability": 0.60,
    },
    "scissors": {
        "width_cm": 5.50,  "height_cm": 20.00,
        "width_std": 1.00, "height_std": 3.00,
        "yolo_class": "scissors",
        "aspect_ratio": 0.275, "depth_cm": 1.0,
        "reliability": 0.68,
    },
    "banana": {
        "width_cm": 16.50, "height_cm": 4.00,
        "width_std": 2.00, "height_std": 0.80,
        "yolo_class": "banana",
        "aspect_ratio": 4.125, "depth_cm": 3.5,
        "reliability": 0.60,
    },
    "apple": {
        "width_cm": 7.60,  "height_cm": 7.60,
        "width_std": 0.80, "height_std": 0.80,
        "yolo_class": "apple",
        "aspect_ratio": 1.000, "depth_cm": 7.6,
        "reliability": 0.70,
    },
    "orange": {
        "width_cm": 7.20,  "height_cm": 7.20,
        "width_std": 0.80, "height_std": 0.80,
        "yolo_class": "orange",
        "aspect_ratio": 1.000, "depth_cm": 7.2,
        "reliability": 0.70,
    },

    # ── Furniture / large objects ──────────────────────────────────────────────
    "chair": {
        "width_cm": 50.00, "height_cm": 90.00,
        "width_std": 8.00, "height_std": 12.00,
        "yolo_class": "chair",
        "aspect_ratio": 0.556, "depth_cm": 50.0,
        "reliability": 0.50,
    },
    "dining table": {
        "width_cm": 120.00, "height_cm": 75.00,
        "width_std": 20.00, "height_std": 5.00,
        "yolo_class": "dining table",
        "aspect_ratio": 1.600, "depth_cm": 80.0,
        "reliability": 0.45,
    },
    "tv": {
        "width_cm": 122.00, "height_cm": 70.00,
        "width_std": 25.00, "height_std": 15.00,
        "yolo_class": "tv",
        "aspect_ratio": 1.743, "depth_cm": 7.0,
        "reliability": 0.55,
    },
    "clock": {
        "width_cm": 30.00, "height_cm": 30.00,
        "width_std": 10.00, "height_std": 10.00,
        "yolo_class": "clock",
        "aspect_ratio": 1.000, "depth_cm": 5.0,
        "reliability": 0.45,
    },

    # ── People / animals ─────────────────────────────────────────────────────
    "person": {
        "width_cm": 45.00, "height_cm": 170.00,
        "width_std": 8.00, "height_std": 12.00,
        "yolo_class": "person",
        "aspect_ratio": 0.265, "depth_cm": 22.0,
        "reliability": 0.55,
        "notes": "Shoulder width × standing height; very context-dependent",
    },
    "cat": {
        "width_cm": 20.00, "height_cm": 25.00,
        "width_std": 4.00, "height_std": 5.00,
        "yolo_class": "cat",
        "aspect_ratio": 0.800, "depth_cm": 30.0,
        "reliability": 0.55,
    },
    "dog": {
        "width_cm": 40.00, "height_cm": 45.00,
        "width_std": 15.00, "height_std": 15.00,
        "yolo_class": "dog",
        "aspect_ratio": 0.889, "depth_cm": 25.0,
        "reliability": 0.40,
        "notes": "Huge breed variance — only use as last resort",
    },

    # ── Vehicles ──────────────────────────────────────────────────────────────
    "car": {
        "width_cm": 185.00, "height_cm": 145.00,
        "width_std": 15.00, "height_std": 10.00,
        "yolo_class": "car",
        "aspect_ratio": 1.276, "depth_cm": 450.0,
        "reliability": 0.60,
    },
    "bicycle": {
        "width_cm": 170.00, "height_cm": 110.00,
        "width_std": 15.00, "height_std": 10.00,
        "yolo_class": "bicycle",
        "aspect_ratio": 1.545, "depth_cm": 60.0,
        "reliability": 0.60,
    },

    # ── No reference ──────────────────────────────────────────────────────────
    "— none (heuristic only) —": {
        "width_cm": 1.0, "height_cm": 1.0,
        "width_std": 1.0, "height_std": 1.0,
        "yolo_class": None,
        "aspect_ratio": 1.0, "depth_cm": 1.0,
        "reliability": 0.0,
        "notes": "Falls back to focal-length model — least accurate",
    },
}
