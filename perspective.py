"""
utils/perspective.py
────────────────────
Perspective correction for SmartMeasure.

Three strategies, applied in order of reliability:

1. **Quad warp** — user marks 4 corners of a known flat rectangle
   (e.g. A4 paper, credit card).  We compute the full homography H
   that maps the distorted image to a rectified top-down view, then
   measure all bounding boxes in that rectified space.

2. **Vanishing-point tilt correction** — when only a tilt angle is
   available (single-axis camera pitch), we apply a 1-D foreshortening
   correction per row using the horizon line derived from the vp.

3. **None** — pass-through; measurements stay in raw-pixel space.

Public API
----------
  PerspectiveCorrector   — stateful class; call .set_quad() or
                           .set_tilt(), then .rectify_image() and
                           .correct_bbox().
  detect_reference_quad  — attempt automatic corner detection for a
                           rectangular reference in the YOLO detections.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Quad:
    """Four corners of a known rectangle in image pixel coords (TL,TR,BR,BL)."""
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array([self.tl, self.tr, self.br, self.bl], dtype=np.float32)


@dataclass
class CorrectionResult:
    """Output of PerspectiveCorrector.correct_bbox()."""
    x1: float; y1: float; x2: float; y2: float
    width_px: float;  height_px: float
    correction_applied: str        # "homography" | "tilt" | "none"
    scale_factor_x: float = 1.0   # how much the correction stretched X
    scale_factor_y: float = 1.0   # how much the correction stretched Y


# ─────────────────────────────────────────────────────────────────────────────
# Core corrector
# ─────────────────────────────────────────────────────────────────────────────

class PerspectiveCorrector:
    """
    Holds perspective calibration state and applies corrections.

    Usage
    -----
    pc = PerspectiveCorrector(image_shape=(H, W))

    # Option A — provide reference quad corners (most accurate)
    pc.set_quad(quad, ref_width_cm, ref_height_cm)

    # Option B — provide estimated camera tilt only
    pc.set_tilt(tilt_deg)

    # Apply to a bounding box
    result = pc.correct_bbox(x1, y1, x2, y2)
    """

    def __init__(self, image_shape: tuple[int, int]):
        self.H, self.W = image_shape[:2]
        self._H_mat: Optional[np.ndarray] = None     # 3×3 homography
        self._H_inv: Optional[np.ndarray] = None
        self._tilt_deg: float = 0.0
        self._mode: str = "none"

        # After set_quad we store the px/cm ratio in the rectified plane
        self.rect_px_per_cm_x: Optional[float] = None
        self.rect_px_per_cm_y: Optional[float] = None
        self._rect_w: int = 0    # width  of the rectified canvas
        self._rect_h: int = 0    # height of the rectified canvas

    # ── Calibration setters ──────────────────────────────────────────────────

    def set_quad(
        self,
        quad: Quad,
        ref_width_cm:  float,
        ref_height_cm: float,
    ) -> None:
        """
        Compute homography from four corner points of a known rectangle.

        The destination canvas is sized so that 1 pixel ≈ the smallest
        pixel-per-cm ratio implied by the quad's own pixel size vs real size,
        keeping the canvas at a sensible resolution.
        """
        src = quad.as_array()

        # Destination rectangle — we set its size proportional to real dims
        # while keeping max dimension ≤ max(W, H) to avoid blowup.
        scale = max(self.W, self.H) / max(ref_width_cm, ref_height_cm)
        dst_w = int(ref_width_cm  * scale)
        dst_h = int(ref_height_cm * scale)

        self._rect_w = dst_w
        self._rect_h = dst_h

        dst = np.array(
            [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]],
            dtype=np.float32,
        )

        self._H_mat, _ = cv2.findHomography(src, dst, method=0)
        if self._H_mat is not None:
            self._H_inv = np.linalg.inv(self._H_mat)
            self._mode = "homography"

            # px/cm in the rectified plane is simply the scale factor
            self.rect_px_per_cm_x = scale
            self.rect_px_per_cm_y = scale
        else:
            self._mode = "none"

    def set_tilt(self, tilt_deg: float) -> None:
        """
        Set a single-axis camera pitch (degrees from vertical).
        0° = camera pointing straight down; 90° = camera horizontal.
        """
        self._tilt_deg = float(tilt_deg)
        self._mode = "tilt" if abs(tilt_deg) > 1.0 else "none"

    # ── Rectified image ──────────────────────────────────────────────────────

    def rectify_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Return a perspective-corrected version of the image.
        Only meaningful when mode == "homography".
        """
        if self._mode == "homography" and self._H_mat is not None:
            return cv2.warpPerspective(
                image_rgb,
                self._H_mat,
                (self._rect_w, self._rect_h),
                flags=cv2.INTER_LINEAR,
            )
        return image_rgb.copy()

    # ── BBox correction ──────────────────────────────────────────────────────

    def correct_bbox(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> CorrectionResult:
        """
        Return perspective-corrected bounding-box dimensions.

        For homography mode we project all four corners through H and
        measure the corrected quad's axis-aligned extent.

        For tilt mode we apply a per-row foreshortening factor derived
        from the vanishing-point geometry.
        """
        if self._mode == "homography" and self._H_mat is not None:
            return self._correct_homography(x1, y1, x2, y2)
        elif self._mode == "tilt":
            return self._correct_tilt(x1, y1, x2, y2)
        else:
            w = x2 - x1;  h = y2 - y1
            return CorrectionResult(x1, y1, x2, y2, w, h, "none")

    def _correct_homography(self, x1, y1, x2, y2) -> CorrectionResult:
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, self._H_mat).reshape(-1, 2)

        rx1, ry1 = warped[:, 0].min(), warped[:, 1].min()
        rx2, ry2 = warped[:, 0].max(), warped[:, 1].max()
        rw, rh = rx2 - rx1, ry2 - ry1

        # Scale factor = ratio of corrected to original pixel dimensions
        sx = rw / max(x2 - x1, 1)
        sy = rh / max(y2 - y1, 1)

        return CorrectionResult(rx1, ry1, rx2, ry2, rw, rh, "homography", sx, sy)

    def _correct_tilt(self, x1, y1, x2, y2) -> CorrectionResult:
        """
        Foreshortening correction for a camera tilted by self._tilt_deg.

        Geometry: the image plane subtends angle θ = tilt_deg from vertical.
        Objects at image row y are foreshortened by factor:
            F(y) = 1 / cos( θ * (y / H - 0.5) * π/2 )
        where y=H/2 is the horizon centre.

        We integrate F over the bbox height to get corrected height,
        and apply an average F at the bbox centre for width.
        """
        theta = np.radians(self._tilt_deg)
        H = float(self.H)

        def foreshortenening(y_px: float) -> float:
            norm_y = (y_px / H - 0.5)   # –0.5 … +0.5
            angle  = theta * norm_y * np.pi / 2.0
            cos_a  = np.cos(angle)
            return 1.0 / max(cos_a, 0.1)

        # Average foreshortening over bbox height (numerical integration)
        ys = np.linspace(y1, y2, max(int(y2 - y1), 2))
        Fs = np.array([foreshortenening(y) for y in ys])
        F_height = float(Fs.mean())

        # Lateral foreshortening at the bbox vertical centre
        F_width  = foreshortenening((y1 + y2) / 2.0)

        raw_w = x2 - x1;  raw_h = y2 - y1
        corr_w = raw_w * F_width
        corr_h = raw_h * F_height

        return CorrectionResult(
            x1, y1,
            x1 + corr_w, y1 + corr_h,
            corr_w, corr_h,
            "tilt",
            F_width, F_height,
        )

    @property
    def mode(self) -> str:
        return self._mode


# ─────────────────────────────────────────────────────────────────────────────
# Automatic quad detection from a bounding box
# ─────────────────────────────────────────────────────────────────────────────

def detect_reference_quad(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    ref_aspect: float = 1.586,   # credit card default
) -> Optional[Quad]:
    """
    Given a tight bounding box around a rectangular reference object,
    attempt to detect its four precise corners using edge detection +
    contour approximation.

    Returns a Quad if successful, else None.

    Parameters
    ----------
    image_rgb  : full image
    bbox       : (x1, y1, x2, y2) tight region containing the reference
    ref_aspect : expected width/height ratio of the reference
    """
    x1, y1, x2, y2 = bbox
    pad = 10
    x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
    x2p = min(image_rgb.shape[1], x2 + pad)
    y2p = min(image_rgb.shape[0], y2 + pad)

    roi = image_rgb[y1p:y2p, x1p:x2p]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive edge threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Approximate to a polygon
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) != 4:
        # Fallback: use the rotated bounding rect corners
        rect = cv2.minAreaRect(cnt)
        box_pts = cv2.boxPoints(rect).astype(np.float32)
        approx = box_pts.reshape(-1, 1, 2)

    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None

    # Re-order corners: TL → TR → BR → BL
    pts = _order_corners(pts)
    # Translate back to full-image coords
    pts[:, 0] += x1p
    pts[:, 1] += y1p

    tl, tr, br, bl = pts
    return Quad(tuple(tl), tuple(tr), tuple(br), tuple(bl))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into [TL, TR, BR, BL] order."""
    # Sort by Y first (top two, bottom two)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL has smallest sum
    rect[2] = pts[np.argmax(s)]   # BR has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR has smallest diff
    rect[3] = pts[np.argmax(diff)]  # BL has largest diff
    return rect


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: estimate tilt from a detected reference object
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tilt_from_bbox(
    bbox: tuple[int, int, int, int],
    ref_width_cm:  float,
    ref_height_cm: float,
    image_height:  int,
) -> float:
    """
    Estimate camera tilt (degrees) from the apparent aspect ratio of a
    reference object whose true aspect ratio is known.

    If the reference appears squashed vertically, the camera is tilted.
    Returns tilt angle in degrees (0 = overhead / no tilt).
    """
    x1, y1, x2, y2 = bbox
    obs_w = x2 - x1;  obs_h = y2 - y1
    if obs_w <= 0 or obs_h <= 0:
        return 0.0

    true_aspect = ref_width_cm / ref_height_cm
    obs_aspect  = obs_w / obs_h

    # If obs_aspect > true_aspect the height is foreshortened → camera tilted
    if obs_aspect <= true_aspect or true_aspect <= 0:
        return 0.0

    ratio = true_aspect / obs_aspect   # cos(tilt_eff) ≈ ratio for small angles
    ratio = np.clip(ratio, 0.01, 1.0)
    tilt_rad = np.arccos(ratio)
    return float(np.degrees(tilt_rad))
