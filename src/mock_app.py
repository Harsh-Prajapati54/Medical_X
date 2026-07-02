import streamlit as st
from PIL import Image
import numpy as np
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Med-X · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Card container ── */
.card {
    background-color: #2a2a2a;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ── Card title ── */
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Upload zone ── */
.upload-zone {
    border: 1.5px dashed #444;
    border-radius: 10px;
    padding: 28px 16px;
    text-align: center;
    background: #222;
}
.upload-icon-wrap {
    width: 52px; height: 52px;
    background: #1a3d2e;
    border-radius: 12px;
    display: flex; align-items: center;
    justify-content: center;
    margin: 0 auto 12px;
    font-size: 24px;
}
.upload-main {
    font-size: 15px; font-weight: 600;
    color: #f0f0f0; margin-bottom: 4px;
}
.upload-sub {
    font-size: 12px; color: #888;
    margin-bottom: 0;
}

/* ── st.file_uploader override ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section {
    border: 1.5px dashed #444 !important;
    border-radius: 10px !important;
    background: #222 !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #1D9E75 !important;
}

/* ── Preview box ── */
.preview-empty {
    background: #222;
    border-radius: 8px;
    height: 130px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #555;
    font-size: 12px;
    border: 0.5px solid #333;
}
.preview-empty span { font-size: 30px; margin-bottom: 6px; }

/* ── Meta rows ── */
.meta-table { width: 100%; margin-top: 10px; }
.meta-table tr td {
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 0.5px solid #333;
}
.meta-table tr:last-child td { border-bottom: none; }
.meta-label { color: #888; }
.meta-val { color: #e0e0e0; font-weight: 500; text-align: right; }

/* ── Run prediction button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background-color: #2a2a2a !important;
    color: #e0e0e0 !important;
    border: 1px solid #444 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin-top: 12px;
}
[data-testid="stButton"] > button:hover {
    border-color: #1D9E75 !important;
    color: #1D9E75 !important;
}

/* ── Findings badge ── */
.findings-badge {
    background: #ffddcc;
    color: #7a2800;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
}
.no-findings-badge {
    background: #ccf0e0;
    color: #0a4d2a;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
}

/* ── Disease bar row ── */
.disease-row {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 10px;
}
.disease-name {
    font-size: 13px; font-weight: 600;
    color: #f0f0f0; min-width: 120px;
}
.bar-track {
    flex: 1; height: 6px;
    background: #333; border-radius: 100px;
    overflow: hidden;
}
.bar-fill-red   { height: 100%; border-radius: 100px; background: #E24B4A; }
.bar-fill-amber { height: 100%; border-radius: 100px; background: #EF9F27; }
.bar-fill-green { height: 100%; border-radius: 100px; background: #1D9E75; }
.bar-pct {
    font-size: 12px; color: #999;
    min-width: 32px; text-align: right;
}

/* ── Legend ── */
.legend { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #999; }
.dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── Disclaimer ── */
.disclaimer {
    background: #2e2200;
    border: 0.5px solid #6b4c00;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #c8a44a;
    line-height: 1.6;
    margin-top: 4px;
}

/* ── Subtext ── */
.sub-text {
    font-size: 11px; color: #666;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Mock prediction function ────────────────────────────────────────────────────
# Replace this with your actual DenseNet121 inference
def run_inference(image: Image.Image):
    """
    TODO: Load your DenseNet121 model and run inference here.
    Returns a dict of {disease_name: confidence_score}.
    This is a mock output for UI testing.
    """
    mock_scores = {
        "Infiltration":  0.72,
        "Effusion":      0.58,
        "Atelectasis":   0.41,
        "No Finding":    0.19,
        "Pneumonia":     0.12,
        "Cardiomegaly":  0.08,
        "Mass":          0.06,
        "Nodule":        0.05,
        "Consolidation": 0.04,
        "Emphysema":     0.03,
        "Pleural Thickening": 0.02,
        "Hernia":        0.01,
        "Edema":         0.01,
        "Fibrosis":      0.01,
    }
    return dict(sorted(mock_scores.items(), key=lambda x: x[1], reverse=True))


def mock_gradcam(image: Image.Image) -> Image.Image:
    """
    TODO: Replace with your actual Grad-CAM implementation.
    Returns a heatmap overlay as a PIL Image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    img_array = np.array(image.convert("L").resize((224, 224)))
    heatmap = np.zeros((224, 224))
    cx, cy = 90, 100
    for i in range(224):
        for j in range(224):
            heatmap[i, j] = np.exp(-((i - cx)**2 + (j - cy)**2) / (2 * 40**2))
    heatmap = heatmap / heatmap.max()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor="#1a1a1a")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def bar_color(score: float) -> str:
    if score >= 0.5:
        return "bar-fill-red"
    elif score >= 0.25:
        return "bar-fill-amber"
    return "bar-fill-green"


# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.1], gap="medium")

# ════════════════════════════════════════════════════════
# LEFT — Upload + Preview
# ════════════════════════════════════════════════════════
with col_left:

    # Upload card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⬆ Upload X-ray image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)  # end upload card

    # Preview card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖼 Preview</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        file_size_kb = round(uploaded_file.size / 1024, 1)
        w, h = image.size

        st.markdown(f"""
        <table class="meta-table">
            <tr><td class="meta-label">Filename</td><td class="meta-val">{uploaded_file.name}</td></tr>
            <tr><td class="meta-label">File size</td><td class="meta-val">{file_size_kb} KB</td></tr>
            <tr><td class="meta-label">Dimensions</td><td class="meta-val">{w} × {h} px</td></tr>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-empty">
            <span>🖼</span>
            No image uploaded yet<br>
            <span style="font-size:10px">st.image() renders here</span>
        </div>
        <table class="meta-table">
            <tr><td class="meta-label">Filename</td><td class="meta-val">—</td></tr>
            <tr><td class="meta-label">File size</td><td class="meta-val">—</td></tr>
            <tr><td class="meta-label">Dimensions</td><td class="meta-val">—</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end preview card

    # Run prediction button
    run_clicked = st.button("⚙ Run prediction", disabled=(uploaded_file is None))


# ════════════════════════════════════════════════════════
# RIGHT — Predictions + Grad-CAM
# ════════════════════════════════════════════════════════
with col_right:

    # Predictions card
    st.markdown('<div class="card">', unsafe_allow_html=True)

    pred_header_left, pred_header_right = st.columns([1.4, 1])
    with pred_header_left:
        st.markdown('<div class="card-title">📋 Predictions</div>', unsafe_allow_html=True)
    with pred_header_right:
        if run_clicked or st.session_state.get("predictions"):
            st.markdown('<div class="findings-badge">Findings detected</div>',
                        unsafe_allow_html=True)

    if run_clicked and uploaded_file is not None:
        image = Image.open(uploaded_file)
        with st.spinner("Running DenseNet121 inference…"):
            scores = run_inference(image)
            gradcam_img = mock_gradcam(image)
        st.session_state["predictions"] = scores
        st.session_state["gradcam"] = gradcam_img

    scores = st.session_state.get("predictions", {})

    if scores:
        st.markdown('<div class="sub-text">Confidence scores per class · DenseNet121 sigmoid output</div>',
                    unsafe_allow_html=True)
        top5 = list(scores.items())[:5]
        bars_html = ""
        for name, score in top5:
            pct = int(score * 100)
            cls = bar_color(score)
            bars_html += f"""
            <div class="disease-row">
                <div class="disease-name">{name}</div>
                <div class="bar-track">
                    <div class="{cls}" style="width:{pct}%"></div>
                </div>
                <div class="bar-pct">{score:.2f}</div>
            </div>"""
        remaining = len(scores) - 5
        bars_html += f'<div class="sub-text" style="margin-top:6px">+ {remaining} more classes · shown top-5 by score</div>'
        st.markdown(bars_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sub-text">Upload an image and click <strong>Run prediction</strong> to see results.</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end predictions card

    # Grad-CAM card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Grad-CAM heatmap</div>', unsafe_allow_html=True)

    gradcam = st.session_state.get("gradcam")
    if gradcam:
        st.image(gradcam, use_container_width=True)
        st.markdown("""
        <div class="legend">
            <div class="legend-item"><div class="dot" style="background:#E24B4A"></div>High attention</div>
            <div class="legend-item"><div class="dot" style="background:#EF9F27"></div>Medium</div>
            <div class="legend-item"><div class="dot" style="background:#1D9E75"></div>Low</div>
        </div>
        <div class="sub-text" style="margin-top:8px">
            Overlay via st.image() with matplotlib · highlights model focus regions
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-empty">
            <span>🔬</span>
            Grad-CAM renders after prediction
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end gradcam card

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Not a medical diagnosis.</strong>
        This tool is for educational and research purposes only.
        Always consult a licensed radiologist or physician for clinical decisions.
        Model accuracy varies with image quality and patient demographics.
    </div>
    """, unsafe_allow_html=True)