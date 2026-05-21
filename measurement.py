"""
utils/measurement.py  (v2 — precision calibration + perspective correction)
────────────────────────────────────────────────────────────────────────────

Pipeline
--------
1. Multi-reference fusion
2. Perspective correction (homography or tilt)
3. Spatial scale field (bilinear interpolation across image)
4. Uncertainty / confidence-interval propagation
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from .reference import REFERENCE_OBJECTS
from .perspective import PerspectiveCorrector, estimate_tilt_from_bbox


# ── Focal-length heuristic (last resort) ──────────────────────────────────────
_FOCAL_PX_AT_1000W = 1050.0
_ASSUMED_DIST_CM   = 55.0

def _heuristic_px_per_cm(image_w: int, image_h: int) -> tuple:
    focal = _FOCAL_PX_AT_1000W * (image_w / 1000.0)
    ppcm  = focal / _ASSUMED_DIST_CM
    return ppcm, ppcm


# ── Reference matching ─────────────────────────────────────────────────────────

def _match_detections_to_reference(detections, ref_name, ref_info):
    primary_label = ref_info.get("yolo_class") or ref_name.lower()
    matched = []
    for det in detections:
        if det["class"].lower() == primary_label.lower():
            matched.append({**det, "_ref_name": ref_name, "_ref_info": ref_info})
    return matched


def _all_reference_detections(detections):
    class_to_ref = {
        v["yolo_class"].lower(): (k, v)
        for k, v in REFERENCE_OBJECTS.items()
        if v.get("yolo_class")
    }
    out = []
    for det in detections:
        key = det["class"].lower()
        if key in class_to_ref:
            ref_name, ref_info = class_to_ref[key]
            out.append({**det, "_ref_name": ref_name, "_ref_info": ref_info})
    return out


# ── Scale from a single reference detection ────────────────────────────────────

def _scale_from_reference(det, corrector):
    ref_info = det["_ref_info"]
    x1, y1, x2, y2 = det["bbox_px"]
    conf = det["confidence"]

    if corrector and corrector.mode != "none":
        cr = corrector.correct_bbox(x1, y1, x2, y2)
        box_w = cr.width_px
        box_h = cr.height_px
    else:
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)

    px_per_cm_x = box_w / ref_info["width_cm"]
    px_per_cm_y = box_h / ref_info["height_cm"]

    std_x    = ref_info.get("width_std",  1.0) / ref_info["width_cm"]
    std_y    = ref_info.get("height_std", 1.0) / ref_info["height_cm"]
    std_norm = (std_x + std_y) / 2.0
    prec_w   = 1.0 / max(std_norm, 0.01)

    reliability = ref_info.get("reliability", 0.5)
    weight = reliability * conf * prec_w

    tilt_deg = estimate_tilt_from_bbox(
        (x1, y1, x2, y2),
        ref_info["width_cm"],
        ref_info["height_cm"],
        image_height=0,
    )

    return {
        "px_per_cm_x": px_per_cm_x,
        "px_per_cm_y": px_per_cm_y,
        "weight":      weight,
        "cx":          (x1 + x2) / 2.0,
        "cy":          (y1 + y2) / 2.0,
        "ref_name":    det["_ref_name"],
        "ref_info":    ref_info,
        "tilt_deg":    tilt_deg,
        "bbox_px":     (x1, y1, x2, y2),
    }


# ── Spatial scale field ────────────────────────────────────────────────────────

class ScaleField:
    def __init__(self, calibrations, image_w, image_h, heuristic_ppcm):
        self.image_w = image_w
        self.image_h = image_h

        if not calibrations:
            self._mode = "heuristic"
            self._ppcm_x = heuristic_ppcm[0]
            self._ppcm_y = heuristic_ppcm[1]
            self._mean_x = self._ppcm_x
            self._mean_y = self._ppcm_y
            self._lo_x = self._ppcm_x * 0.5; self._hi_x = self._ppcm_x * 1.5
            self._lo_y = self._ppcm_y * 0.5; self._hi_y = self._ppcm_y * 1.5
            return

        weights = np.array([c["weight"] for c in calibrations], dtype=float)
        if weights.sum() < 1e-9:
            weights[:] = 1.0
        weights /= weights.sum()

        xs  = np.array([c["cx"] for c in calibrations])
        ys  = np.array([c["cy"] for c in calibrations])
        ppx = np.array([c["px_per_cm_x"] for c in calibrations])
        ppy = np.array([c["px_per_cm_y"] for c in calibrations])
        n   = len(calibrations)

        if n == 1:
            self._mode   = "constant"
            self._ppcm_x = float(ppx[0])
            self._ppcm_y = float(ppy[0])
        elif n == 2:
            self._mode    = "linear"
            self._xs = xs; self._ys = ys
            self._ppx = ppx; self._ppy = ppy
            self._weights = weights
        else:
            self._mode = "plane"
            A = np.column_stack([np.ones(n), xs / image_w, ys / image_h])
            W = np.diag(weights)
            try:
                AW = A.T @ W
                self._coef_x = np.linalg.lstsq(AW @ A, AW @ ppx, rcond=None)[0]
                self._coef_y = np.linalg.lstsq(AW @ A, AW @ ppy, rcond=None)[0]
            except np.linalg.LinAlgError:
                self._mode   = "constant"
                self._ppcm_x = float(np.average(ppx, weights=weights))
                self._ppcm_y = float(np.average(ppy, weights=weights))

        self._mean_x = float(np.average(ppx, weights=weights))
        self._mean_y = float(np.average(ppy, weights=weights))
        self._lo_x = self._mean_x * 0.50; self._hi_x = self._mean_x * 1.50
        self._lo_y = self._mean_y * 0.50; self._hi_y = self._mean_y * 1.50

    def sample(self, cx, cy):
        if self._mode in ("heuristic", "constant"):
            return self._ppcm_x, self._ppcm_y
        elif self._mode == "linear":
            d = np.maximum(np.hypot(self._xs - cx, self._ys - cy), 1.0)
            inv_d = (1.0 / d) * self._weights
            inv_d /= inv_d.sum()
            px = float(np.dot(inv_d, self._ppx))
            py = float(np.dot(inv_d, self._ppy))
            return np.clip(px, self._lo_x, self._hi_x), np.clip(py, self._lo_y, self._hi_y)
        else:  # plane
            feat = np.array([1.0, cx / self.image_w, cy / self.image_h])
            px = float(feat @ self._coef_x)
            py = float(feat @ self._coef_y)
            return np.clip(px, self._lo_x, self._hi_x), np.clip(py, self._lo_y, self._hi_y)

    @property
    def mode(self):
        return self._mode

    @property
    def mean_ppcm(self):
        return self._mean_x, self._mean_y


# ── Uncertainty propagation ────────────────────────────────────────────────────

def _measurement_uncertainty(est_w, est_h, ref_std_x, ref_std_y,
                              px_per_cm_x, px_per_cm_y,
                              yolo_conf, calibration_mode, perspective_mode):
    frac_ref_x = (ref_std_x / px_per_cm_x) / max(est_w, 0.01)
    frac_ref_y = (ref_std_y / px_per_cm_y) / max(est_h, 0.01)
    loc_frac   = 0.02 + (1.0 - yolo_conf) * 0.05
    persp_frac = {"homography": 0.02, "tilt": 0.07, "none": 0.15}.get(perspective_mode, 0.15)
    calib_frac = {"plane": 0.03, "linear": 0.05, "constant": 0.06, "heuristic": 0.20}.get(calibration_mode, 0.10)
    total_x = np.sqrt(frac_ref_x**2 + loc_frac**2 + persp_frac**2 + calib_frac**2)
    total_y = np.sqrt(frac_ref_y**2 + loc_frac**2 + persp_frac**2 + calib_frac**2)
    return round(est_w * total_x, 2), round(est_h * total_y, 2)


# ── Main public function ───────────────────────────────────────────────────────

def estimate_measurements(
    detections,
    image_shape,
    ref_name,
    ref_info,
    perspective_corrector=None,
):
    H, W = image_shape[:2]
    heuristic = _heuristic_px_per_cm(W, H)

    corrector = perspective_corrector

    # Auto-detect tilt from primary reference if no corrector provided
    if corrector is None:
        primary_matches = _match_detections_to_reference(detections, ref_name, ref_info)
        if primary_matches:
            det0 = primary_matches[0]
            x1, y1, x2, y2 = det0["bbox_px"]
            tilt = estimate_tilt_from_bbox(
                (x1, y1, x2, y2),
                ref_info["width_cm"],
                ref_info["height_cm"],
                image_height=H,
            )
            if abs(tilt) > 3.0:
                corrector = PerspectiveCorrector((H, W))
                corrector.set_tilt(tilt)

    # Gather references
    primary_refs = _match_detections_to_reference(detections, ref_name, ref_info)
    all_refs     = _all_reference_detections(detections)
    seen_ids     = {id(d) for d in primary_refs}
    combined_refs = primary_refs + [d for d in all_refs if id(d) not in seen_ids]

    calibrations = [_scale_from_reference(r, corrector) for r in combined_refs]

    field      = ScaleField(calibrations, W, H, heuristic)
    persp_mode = corrector.mode if corrector else "none"
    ref_bbox_set = {tuple(d["bbox_px"]) for d in combined_refs}

    result = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_px"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ppcm_x, ppcm_y = field.sample(cx, cy)

        cfx = cfy = 1.0
        if corrector and corrector.mode != "none":
            cr    = corrector.correct_bbox(x1, y1, x2, y2)
            box_w = cr.width_px
            box_h = cr.height_px
            cfx   = cr.scale_factor_x
            cfy   = cr.scale_factor_y
        else:
            box_w = max(x2 - x1, 1.0)
            box_h = max(y2 - y1, 1.0)

        est_w = box_w / ppcm_x
        est_h = box_h / ppcm_y

        cref = calibrations[0]["ref_info"] if calibrations else ref_info
        mw, mh = _measurement_uncertainty(
            est_w, est_h,
            cref.get("width_std",  0.5),
            cref.get("height_std", 0.5),
            ppcm_x, ppcm_y,
            det["confidence"],
            field.mode,
            persp_mode,
        )

        result.append({
            **det,
            "est_width_cm":        round(est_w, 2),
            "est_height_cm":       round(est_h, 2),
            "margin_w_cm":         mw,
            "margin_h_cm":         mh,
            "px_per_cm_x":         round(ppcm_x, 3),
            "px_per_cm_y":         round(ppcm_y, 3),
            "calibration_mode":    field.mode,
            "perspective_mode":    persp_mode,
            "correction_factor_x": round(cfx, 3),
            "correction_factor_y": round(cfy, 3),
            "n_references":        len(calibrations),
            "center_x":            int(cx),
            "center_y":            int(cy),
            "is_reference":        tuple(det["bbox_px"]) in ref_bbox_set,
        })

    return result
