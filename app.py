import streamlit as st

st.set_page_config(
    page_title="SmartMeasure v2",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:#121212; --surface:#1E1E1E; --border:#555555;
    --accent:#4DA3FF; --accent2:#66B2FF; --warn:#FFC857;
    --green:#22c55e; --red:#ef4444;
    --text:#FFFFFF; --muted:#C7C7C7;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
.sm-title{font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;margin-bottom:0}
.sm-sub{color:var(--muted);font-size:.9rem;letter-spacing:.05em;margin-top:4px}
.metric-row{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 18px;flex:1;min-width:110px}
.metric-card .val{font-family:'Space Mono',monospace;font-size:1.5rem;color:var(--accent);font-weight:700}
.metric-card .lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.det-table{width:100%;border-collapse:collapse;font-size:.85rem}
.det-table th{background:var(--border);padding:7px 11px;text-align:left;font-family:'Space Mono',monospace;font-size:.68rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase}
.det-table td{padding:7px 11px;border-bottom:1px solid var(--border);vertical-align:middle}
.det-table tr:hover td{background:var(--border)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.7rem;font-weight:600;background:rgba(0,229,255,.15);color:var(--accent);border:1px solid rgba(0,229,255,.3)}
.badge-ref{background:rgba(255,215,0,.15);color:#ffd700;border:1px solid rgba(255,215,0,.3)}
.badge-mode-h{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.badge-mode-t{background:rgba(245,158,11,.15);color:var(--warn);border:1px solid rgba(245,158,11,.3)}
.badge-mode-n{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.info-box{background:rgba(0,229,255,.07);border-left:3px solid var(--accent);border-radius:6px;padding:11px 15px;font-size:.875rem;margin:.6rem 0}
.warn-box{background:rgba(245,158,11,.08);border-left:3px solid var(--warn);border-radius:6px;padding:11px 15px;font-size:.875rem;margin:.6rem 0}
.good-box{background:rgba(34,197,94,.08);border-left:3px solid var(--green);border-radius:6px;padding:11px 15px;font-size:.875rem;margin:.6rem 0}
[data-testid="stFileUploader"]{background:var(--surface)!important;border:2px dashed var(--border)!important;border-radius:12px!important}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important}
.stButton>button{background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%)!important;color:#000!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-size:.78rem!important;letter-spacing:.05em!important;padding:.5rem 1.2rem!important;font-weight:700!important}
.stProgress>div>div{background:var(--accent)!important}
#MainMenu,footer,header{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image
import io, time

from utils.detector    import load_model, run_detection
from utils.drawing     import draw_detections
from utils.measurement import estimate_measurements
from utils.reference   import REFERENCE_OBJECTS
from utils.perspective import PerspectiveCorrector, Quad, detect_reference_quad

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="sm-title">📐 SmartMeasure v2</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sm-sub">YOLO · Multi-Reference Fusion · Perspective Correction · Uncertainty Estimation</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ YOLO Settings")
    model_size     = st.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m"], index=0)
    conf_threshold = st.slider("Confidence",  0.10, 0.95, 0.40, 0.05)
    iou_threshold  = st.slider("IoU (NMS)",   0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 📏 Primary Reference Object")
    ref_name = st.selectbox(
        "Reference",
        list(REFERENCE_OBJECTS.keys()),
        help="Object with known real size — used as calibration anchor",
    )
    ref_info = REFERENCE_OBJECTS[ref_name]

    if ref_info["reliability"] >= 0.90:
        rel_badge = "🟢"
    elif ref_info["reliability"] >= 0.65:
        rel_badge = "🟡"
    else:
        rel_badge = "🔴"

    st.markdown(
        f'<div class="info-box">'
        f'<b>{ref_name}</b><br>'
        f'W: {ref_info["width_cm"]} ± {ref_info.get("width_std","?")}&nbsp;cm &nbsp;|&nbsp; '
        f'H: {ref_info["height_cm"]} ± {ref_info.get("height_std","?")}&nbsp;cm<br>'
        f'Reliability: {rel_badge} {ref_info.get("reliability",0):.0%}'
        f'</div>',
        unsafe_allow_html=True,
    )

    custom_ref = st.checkbox("Override reference dimensions")
    if custom_ref:
        cw = st.number_input("Width (cm)",  value=float(ref_info["width_cm"]),  min_value=0.1)
        ch = st.number_input("Height (cm)", value=float(ref_info["height_cm"]), min_value=0.1)
        ref_info = {**ref_info, "width_cm": cw, "height_cm": ch}

    st.markdown("---")
    st.markdown("### 🔭 Perspective Correction")

    persp_mode_sel = st.radio(
        "Correction mode",
        ["Auto-detect (recommended)", "Manual quad (4 corners)", "Manual tilt angle", "Disabled"],
        index=0,
    )

    manual_quad_corners  = None
    manual_tilt_deg      = 0.0

    if persp_mode_sel == "Manual tilt angle":
        manual_tilt_deg = st.slider(
            "Camera tilt (° from vertical)", 0.0, 75.0, 15.0, 1.0,
            help="0° = camera directly overhead; 45° = 45-degree angle"
        )
    elif persp_mode_sel == "Manual quad (4 corners)":
        st.markdown(
            '<div class="warn-box">Enter the pixel coordinates of the four corners '
            'of your reference object (TL → TR → BR → BL).</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            tl_x = st.number_input("TL x", value=50,  step=1)
            tr_x = st.number_input("TR x", value=300, step=1)
            br_x = st.number_input("BR x", value=310, step=1)
            bl_x = st.number_input("BL x", value=45,  step=1)
        with col2:
            tl_y = st.number_input("TL y", value=50,  step=1)
            tr_y = st.number_input("TR y", value=55,  step=1)
            br_y = st.number_input("BR y", value=200, step=1)
            bl_y = st.number_input("BL y", value=195, step=1)
        manual_quad_corners = Quad(
            (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)
        )

    st.markdown("---")
    st.markdown("### 🎨 Visualisation")
    show_labels      = st.checkbox("Class labels",          value=True)
    show_conf        = st.checkbox("Confidence score",      value=True)
    show_dims        = st.checkbox("Estimated dimensions",  value=True)
    show_uncertainty = st.checkbox("±Uncertainty margins",  value=True)
    show_correction  = st.checkbox("Correction factors",    value=False)
    show_crosshair   = st.checkbox("Object centres",        value=False)
    show_banner      = st.checkbox("Calibration banner",    value=True)
    box_thickness    = st.slider("Box thickness", 1, 5, 2)

    draw_cfg = dict(
        show_labels=show_labels, show_conf=show_conf,
        show_dims=show_dims, show_uncertainty=show_uncertainty,
        show_correction=show_correction, show_crosshair=show_crosshair,
        show_persp_banner=show_banner, thickness=box_thickness,
    )

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(name):
    return load_model(name)

with st.spinner("Loading YOLO model…"):
    model = get_model(model_size)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an image — JPG, PNG, or WEBP",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.markdown(
        '<div class="warn-box">📂 Upload an image to begin. '
        'For best accuracy: place a <b>credit card</b> or <b>A4 sheet</b> flat in the scene '
        'and select it as the reference object.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
image_pil = Image.open(uploaded).convert("RGB")
image_np  = np.array(image_pil)
H_img, W_img = image_np.shape[:2]

prog = st.progress(0, text="Running YOLO inference…")
t0   = time.time()
detections = run_detection(model, image_np, conf=conf_threshold, iou=iou_threshold)
elapsed_ms = (time.time() - t0) * 1000
prog.progress(40, text="Building perspective corrector…")

# ── Build PerspectiveCorrector ────────────────────────────────────────────────
corrector = None

if persp_mode_sel == "Manual quad (4 corners)" and manual_quad_corners is not None:
    corrector = PerspectiveCorrector((H_img, W_img))
    corrector.set_quad(manual_quad_corners, ref_info["width_cm"], ref_info["height_cm"])

elif persp_mode_sel == "Manual tilt angle":
    corrector = PerspectiveCorrector((H_img, W_img))
    corrector.set_tilt(manual_tilt_deg)

elif persp_mode_sel == "Auto-detect (recommended)":
    # Try to find reference quad automatically
    ref_label = ref_info.get("yolo_class") or ref_name.lower()
    for det in detections:
        if det["class"].lower() == ref_label.lower():
            quad = detect_reference_quad(
                image_np, tuple(det["bbox_px"]),
                ref_aspect=ref_info["width_cm"] / ref_info["height_cm"],
            )
            if quad is not None:
                corrector = PerspectiveCorrector((H_img, W_img))
                corrector.set_quad(quad, ref_info["width_cm"], ref_info["height_cm"])
                break
    # Fallback to tilt if quad not found
    # (estimate_measurements will auto-detect tilt if corrector is still None)

prog.progress(70, text="Computing measurements…")
measurements = estimate_measurements(
    detections, image_np.shape, ref_name, ref_info, corrector
)
prog.progress(90, text="Rendering output…")
annotated = draw_detections(image_np.copy(), measurements, draw_cfg)
prog.progress(100, text="Done!")
time.sleep(0.15)
prog.empty()

# ── Calibration status banner ─────────────────────────────────────────────────
if measurements:
    pm   = measurements[0].get("perspective_mode", "none")
    cm   = measurements[0].get("calibration_mode", "heuristic")
    nref = measurements[0].get("n_references", 0)

    if pm == "homography" and cm in ("constant", "plane", "linear"):
        st.markdown(
            f'<div class="good-box">✅ <b>Homography calibration active</b> — '
            f'{nref} reference(s) found. Calibration mode: <b>{cm}</b>. '
            f'Expected accuracy: <b>±3–8%</b>.</div>',
            unsafe_allow_html=True,
        )
    elif pm == "tilt":
        st.markdown(
            f'<div class="warn-box">⚠️ <b>Tilt correction active</b> (no quad detected) — '
            f'{nref} reference(s) found. Calibration: <b>{cm}</b>. '
            f'Expected accuracy: <b>±8–15%</b>.</div>',
            unsafe_allow_html=True,
        )
    elif cm in ("constant", "plane", "linear") and nref > 0:
        st.markdown(
            f'<div class="warn-box">🟡 <b>Reference calibrated, no perspective correction</b> — '
            f'{nref} reference(s). Accuracy: <b>±5–12%</b> (flat scene assumed).</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="warn-box">🔴 <b>Heuristic fallback</b> — no reference detected. '
            'Select a visible reference for much better accuracy.</div>',
            unsafe_allow_html=True,
        )

# ── Metrics ───────────────────────────────────────────────────────────────────
n_det    = len(measurements)
classes  = list({m["class"] for m in measurements})
avg_conf = np.mean([m["confidence"] for m in measurements]) if measurements else 0
n_refs   = sum(1 for m in measurements if m.get("is_reference"))

st.markdown(
    f"""<div class="metric-row">
      <div class="metric-card"><div class="val">{n_det}</div><div class="lbl">Detections</div></div>
      <div class="metric-card"><div class="val">{len(classes)}</div><div class="lbl">Classes</div></div>
      <div class="metric-card"><div class="val">{avg_conf:.0%}</div><div class="lbl">Avg Conf</div></div>
      <div class="metric-card"><div class="val">{n_refs}</div><div class="lbl">References</div></div>
      <div class="metric-card"><div class="val">{elapsed_ms:.0f}ms</div><div class="lbl">Inference</div></div>
      <div class="metric-card"><div class="val">{W_img}×{H_img}</div><div class="lbl">Resolution</div></div>
    </div>""",
    unsafe_allow_html=True,
)

# ── Images ────────────────────────────────────────────────────────────────────
col_orig, col_ann = st.columns(2, gap="medium")
with col_orig:
    st.markdown("**Original**")
    st.image(image_pil, use_container_width=True)
with col_ann:
    st.markdown("**Detections + Measurements**")
    st.image(annotated, use_container_width=True)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("⬇ Download Annotated Image",
                       data=buf.getvalue(),
                       file_name="smartmeasure_v2_output.png",
                       mime="image/png")

# ── Detection table ───────────────────────────────────────────────────────────
if measurements:
    st.markdown("### 🔍 Detection Results")

    pm_mode_css = {"homography": "badge-mode-h", "tilt": "badge-mode-t"}.get(
        measurements[0].get("perspective_mode", "none"), "badge-mode-n"
    )

    rows = ""
    for i, m in enumerate(measurements, 1):
        ref_cls = "badge-ref" if m.get("is_reference") else "badge"
        pm_lbl  = m.get("perspective_mode", "none").upper()
        cm_lbl  = m.get("calibration_mode", "?").upper()
        mw      = m.get("margin_w_cm", 0)
        mh      = m.get("margin_h_cm", 0)
        bw      = m["bbox_px"][2] - m["bbox_px"][0]
        bh      = m["bbox_px"][3] - m["bbox_px"][1]
        rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><span class='{ref_cls}'>{m['class']}</span></td>"
            f"<td>{m['confidence']:.1%}</td>"
            f"<td>{bw}×{bh}px</td>"
            f"<td><b>{m['est_width_cm']:.1f}±{mw:.1f}</b> × "
            f"<b>{m['est_height_cm']:.1f}±{mh:.1f}</b> cm</td>"
            f"<td><span class='badge {pm_mode_css}'>{pm_lbl}</span> {cm_lbl}</td>"
            f"<td>{m['n_references']}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""<table class="det-table">
          <thead><tr>
            <th>#</th><th>Class</th><th>Conf</th><th>BBox</th>
            <th>W±σ × H±σ (cm)</th><th>Persp / Calib</th><th>Refs</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="warn-box">No objects detected. Lower the confidence threshold.</div>',
        unsafe_allow_html=True,
    )

# ── Calibration diagnostics ───────────────────────────────────────────────────
if measurements:
    with st.expander("📊 Calibration Diagnostics"):
        pm   = measurements[0].get("perspective_mode", "none")
        cm   = measurements[0].get("calibration_mode", "heuristic")
        nref = measurements[0].get("n_references", 0)

        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Perspective mode", pm.upper())
        with dcol2:
            st.metric("Calibration mode", cm.upper())
        with dcol3:
            st.metric("Reference objects used", nref)

        if nref > 0:
            m0      = measurements[0]
            ppcm_x  = m0.get("px_per_cm_x", 0)
            ppcm_y  = m0.get("px_per_cm_y", 0)
            cx_f    = m0.get("correction_factor_x", 1.0)
            cy_f    = m0.get("correction_factor_y", 1.0)
            st.markdown(
                f"**Scale (centre of image):** `{ppcm_x:.2f} px/cm` (X) · `{ppcm_y:.2f} px/cm` (Y)  \n"
                f"**Perspective correction factors:** `{cx_f:.3f}` (X) · `{cy_f:.3f}` (Y)"
            )

# ── How it works ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ How SmartMeasure v2 achieves better accuracy"):
    st.markdown("""
**1 · Multi-reference fusion**

Every detected object whose class exists in the reference database contributes
a calibration point.  Each point is weighted by:
- Reference *reliability* (ISO-spec objects like credit cards score 0.99)
- YOLO *detection confidence*
- Inverse *standard deviation* of the object's real-world size

**2 · Spatial scale field**

With 1 reference: constant px/cm ratio.  
With 2: linear gradient.  
With 3+: weighted least-squares plane fit — captures lens distortion gradients.

**3 · Perspective correction modes**

| Mode | How activated | Expected error |
|---|---|---|
| Homography | 4-corner quad of reference detected | ±3–8% |
| Tilt correction | Tilt ≥3° inferred from reference aspect ratio | ±8–15% |
| None | No reference in scene | ±15–30% |

**4 · Uncertainty propagation**

Each measurement shows `est ± margin` where margin combines:
- Reference object size variance (σ from database)
- YOLO bbox localisation error (~2–5% of box size)
- Residual perspective error
- Calibration model uncertainty

**Tips for ±5% accuracy**
1. Place a **credit card** flat in the scene (ISO spec, 0.02 cm tolerance)
2. Use **Auto-detect** mode — the app will find its corners automatically
3. Shoot from **directly above** or at most 20° angle
4. Use `yolov8s` or `yolov8m` for better bbox precision
    """)
