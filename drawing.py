"""
utils/drawing.py  (v2)
Draw bounding boxes, labels, measurement overlays, and perspective-correction
indicators on an RGB numpy image.
"""
from __future__ import annotations

import cv2
import numpy as np
import colorsys, hashlib


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _class_colour(class_name: str) -> tuple:
    digest = int(hashlib.md5(class_name.encode()).hexdigest(), 16)
    hue    = (digest % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))   # BGR

def _contrasting(bgr: tuple) -> tuple:
    b, g, r = bgr
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if lum > 128 else (255, 255, 255)

# Fixed colours for special labels
_CYAN_BGR   = (255, 230,   0)
_GOLD_BGR   = (  0, 215, 255)
_RED_BGR    = (  0,   0, 220)
_GREEN_BGR  = (  0, 200,   0)


# ── Helper: bgr ↔ rgb toggling ────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Overlay helper: draws a filled rectangle under text ───────────────────────

def _put_label_block(
    img_bgr: np.ndarray,
    lines: list[str],
    anchor_x: int, anchor_y: int,
    bg_colour: tuple,
    font=cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float = 0.48,
    thickness: int = 1,
    padding: int = 4,
) -> np.ndarray:
    if not lines:
        return img_bgr
    txt_col = _contrasting(bg_colour)
    (_, lh), base = cv2.getTextSize("Ay", font, font_scale, 1)
    line_h  = lh + base + 3
    max_lw  = max(cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines)
    block_h = line_h * len(lines) + padding * 2
    bx2 = anchor_x + max_lw + padding * 2
    by2 = anchor_y + block_h
    cv2.rectangle(img_bgr, (anchor_x, anchor_y), (bx2, by2), bg_colour, -1)
    for i, line in enumerate(lines):
        ty = anchor_y + padding + (i + 1) * line_h - base
        cv2.putText(img_bgr, line, (anchor_x + padding, ty),
                    font, font_scale, txt_col, thickness, cv2.LINE_AA)
    return img_bgr


# ── Perspective mode banner ────────────────────────────────────────────────────

def _draw_persp_banner(img_bgr: np.ndarray, persp_mode: str, calib_mode: str,
                        n_refs: int) -> np.ndarray:
    mode_str   = persp_mode.upper()
    calib_str  = calib_mode.upper()
    banner_col = {
        "homography": _GREEN_BGR,
        "tilt":       _GOLD_BGR,
        "none":       _RED_BGR,
    }.get(persp_mode, _RED_BGR)

    text = f"PERSP:{mode_str}  CALIB:{calib_str}  REFS:{n_refs}"
    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(img_bgr, text, (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_col, 1, cv2.LINE_AA)
    return img_bgr


# ── Main draw function ─────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    measurements: list[dict],
    cfg: dict,
) -> np.ndarray:
    """
    Draw all detections onto *image* (RGB).

    cfg keys
    --------
    show_labels, show_conf, show_dims, show_uncertainty,
    show_crosshair, show_persp_banner, thickness
    """
    thickness   = cfg.get("thickness", 2)
    font        = cv2.FONT_HERSHEY_DUPLEX
    font_scale  = 0.48
    pad         = 4

    if not measurements:
        return image

    # Grab global info for the banner from first measurement
    persp_mode = measurements[0].get("perspective_mode", "none")
    calib_mode = measurements[0].get("calibration_mode", "heuristic")
    n_refs     = measurements[0].get("n_references", 0)

    for m in measurements:
        x1, y1, x2, y2 = m["bbox_px"]
        is_ref  = m.get("is_reference", False)
        colour  = _GOLD_BGR if is_ref else _class_colour(m["class"])

        img_bgr = _to_bgr(image)

        # ── Semi-transparent fill ─────────────────────────────────────────
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.08, img_bgr, 0.92, 0, img_bgr)

        # ── Bounding box ──────────────────────────────────────────────────
        # Double-stroke: thin inner + thicker outer for visibility
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0),       thickness + 2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), colour,           thickness)

        # Reference object gets a corner-tick decoration
        if is_ref:
            tick = 12
            for (px, py), (dx, dy) in [
                ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                ((x2, y2), (-1, -1)), ((x1, y2), (1, -1)),
            ]:
                cv2.line(img_bgr, (px, py), (px + dx * tick, py), colour, 2, cv2.LINE_AA)
                cv2.line(img_bgr, (px, py), (px, py + dy * tick), colour, 2, cv2.LINE_AA)

        image = _to_rgb(img_bgr)

        # ── Label lines ───────────────────────────────────────────────────
        lines: list[str] = []

        if cfg.get("show_labels"):
            label = m["class"]
            if is_ref:
                label = f"[REF] {label}"
            lines.append(label)

        if cfg.get("show_conf"):
            lines.append(f"{m['confidence']:.0%}")

        if cfg.get("show_dims"):
            w_str = f"{m['est_width_cm']:.1f}"
            h_str = f"{m['est_height_cm']:.1f}"
            if cfg.get("show_uncertainty") and m.get("margin_w_cm") is not None:
                lines.append(f"{w_str}±{m['margin_w_cm']:.1f} x {h_str}±{m['margin_h_cm']:.1f} cm")
            else:
                lines.append(f"{w_str} x {h_str} cm")

        # Correction factor badge
        if cfg.get("show_correction") and persp_mode != "none":
            fx = m.get("correction_factor_x", 1.0)
            fy = m.get("correction_factor_y", 1.0)
            if abs(fx - 1.0) > 0.02 or abs(fy - 1.0) > 0.02:
                lines.append(f"cf:{fx:.2f}x{fy:.2f}")

        if not lines:
            img_bgr = _to_bgr(image)
        else:
            # Place label block above box (clamp to image top)
            (_, lh), base = cv2.getTextSize("Ay", font, font_scale, 1)
            line_h  = lh + base + 3
            block_h = line_h * len(lines) + pad * 2
            by1 = y1 - block_h if y1 - block_h >= 0 else y1

            img_bgr = _to_bgr(image)
            img_bgr = _put_label_block(img_bgr, lines, x1, by1, colour,
                                        font, font_scale, 1, pad)
            image = _to_rgb(img_bgr)

        # ── Crosshair at centre ───────────────────────────────────────────
        if cfg.get("show_crosshair"):
            cx, cy_c = m["center_x"], m["center_y"]
            arm = 10
            img_bgr = _to_bgr(image)
            cv2.line(img_bgr, (cx - arm, cy_c), (cx + arm, cy_c), colour, 1, cv2.LINE_AA)
            cv2.line(img_bgr, (cx, cy_c - arm), (cx, cy_c + arm), colour, 1, cv2.LINE_AA)
            cv2.circle(img_bgr, (cx, cy_c), 3, colour, -1, cv2.LINE_AA)
            image = _to_rgb(img_bgr)

    # ── Perspective-mode banner ────────────────────────────────────────────────
    if cfg.get("show_persp_banner", True) and measurements:
        img_bgr = _to_bgr(image)
        img_bgr = _draw_persp_banner(img_bgr, persp_mode, calib_mode, n_refs)
        image = _to_rgb(img_bgr)

    # ── Watermark ─────────────────────────────────────────────────────────────
    H, W = image.shape[:2]
    img_bgr = _to_bgr(image)
    cv2.putText(img_bgr, "SmartMeasure v2",
                (W - 180, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return _to_rgb(img_bgr)
