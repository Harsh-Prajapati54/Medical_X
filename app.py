import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from predict import load_model, get_transform, ALL_LABELS, GradCAM, overlay_gradcam
import cv2
import numpy as np

st.set_page_config(
    page_title="Med-X · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
)

# Global CSS
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 3px solid #2a2a2a !important;
}
 
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
    display: flex;
    flex-direction: column;
    height: 100%;
}
 
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
 
[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
    margin: 0 !important;
}
 
.sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px !important;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 0px !important;
}
.sb-logo-icon {
    width: 40px; height: 40px;
    background: #1D9E75;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.sb-logo-text { font-size: 18px; font-weight: 700; color: #f0f0f0; }
.sb-logo-sub  { font-size: 13px; color: #666; margin-top: 1px; }
 
.sb-section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #444;
    padding: 14px 16px 4px;
}
 
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    background: transparent !important;
    border: none !important;
    color: #999 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    margin: 0px !important;
}
 
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
}
 
.nav-active > button {
    background: #e8f5f0 !important;
    color: #1D9E75 !important;
    font-weight: 600 !important;
}
 
.sb-footer {
    padding: 14px 16px;
    border-top: 1px solid #2a2a2a;
    margin-top: auto;
}
.sb-footer-row {
    font-size: 12px;
    color: #555;
    margin-bottom: 3px;
}
.sb-footer-row:last-child { margin-bottom: 0; }
.sb-footer-row span { color: #888; }

.card {
    background-color: #2a2a2a;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 14px;
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
.meta-table { width: 100%; margin-top: 10px; }
.meta-table tr td {
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 0.5px solid #333;
}
.meta-table tr:last-child td { border-bottom: none; }
.meta-label { color: #888; }
.meta-val { color: #e0e0e0; font-weight: 500; text-align: right; }

.findings-badge {
    background: #ffddcc;
    color: #7a2800;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-block;
}
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
.sub-text { font-size: 11px; color: #666; margin-bottom: 10px; }

.legend { display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #999; }
.dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed rgba(255, 255, 255, 0.18);
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.03);
    margin-bottom: 1rem;
}
.upload-zone-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
.upload-zone h4 {
    color: #e0e0e0;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.25rem;
}
.upload-zone p {
    color: #888;
    font-size: 0.78rem;
    margin: 0;
}

/* ── Findings badge ── */
.findings-badge {
    display: inline-block;
    background: #f87171;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    float: right;
    margin-top: -2px;
}

/* ── Prediction rows ── */
.pred-section-label {
    font-size: 0.7rem;
    color: #888;
    margin: 0.2rem 0 0.8rem;
    letter-spacing: 0.03em;
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.pred-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pred-label {
    color: #d0d0d0;
    font-size: 0.88rem;
    font-weight: 500;
    flex: 1;
}
.pred-bar-wrap {
    flex: 2;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.pred-bar {
    height: 100%;
    border-radius: 4px;
}
.pred-score {
    color: #ccc;
    font-size: 0.82rem;
    font-family: monospace;
    min-width: 32px;
    text-align: right;
}
.pred-more {
    font-size: 0.72rem;
    color: #666;
    margin-top: 0.5rem;
}

/* ── Grad-CAM legend ── */
.gradcam-legend {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-top: 0.6rem;
    font-size: 0.78rem;
    color: #888;
}
.gradcam-legend span {
    display: flex;
    align-items: center;
    gap: 5px;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.gradcam-caption {
    font-size: 0.72rem;
    color: #555;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ── Run prediction button ── */
/* Streamlit button override — add this alongside existing button CSS */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: background 0.15s;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.11) !important;
}

.pred-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.gradcam-caption {
    font-size: 0.72rem;
    color: #555;
    margin-top: 0.4rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)
            
st.title("Medical-X")
st.write("Welcome to the Medical-X application! This app allows you to analyze medical X-ray images and predict the chest disease using machine learning models.")

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
 
    # Logo
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🫁</div>
        <div>
            <div class="sb-logo-text">Med-X</div>
            <div class="sb-logo-sub">Chest X-ray AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Session state init
    if "page" not in st.session_state:
        st.session_state["page"] = "Analyze"
 
    # MAIN nav
    st.markdown('<div class="sb-section-label">MAIN</div>', unsafe_allow_html=True)
 
    if st.session_state["page"] == "Analyze":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button("⇄  Analyze X-ray", key="nav_analyze"):
        st.session_state["page"] = "Analyze"
    if st.session_state["page"] == "Analyze":
        st.markdown('</div>', unsafe_allow_html=True)
 
    if st.session_state["page"] == "History":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button("🕓  History", key="nav_history"):
        st.session_state["page"] = "History"
    if st.session_state["page"] == "History":
        st.markdown('</div>', unsafe_allow_html=True)
 
    # INFO nav
    st.markdown('<div class="sb-section-label">INFO</div>', unsafe_allow_html=True)
 
    if st.session_state["page"] == "About":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button("ⓘ  About model", key="nav_about"):
        st.session_state["page"] = "About"
    if st.session_state["page"] == "About":
        st.markdown('</div>', unsafe_allow_html=True)
 
    if st.session_state["page"] == "HowTo":
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button("☰  How to use", key="nav_howto"):
        st.session_state["page"] = "HowTo"
    if st.session_state["page"] == "HowTo":
        st.markdown('</div>', unsafe_allow_html=True)
 
    # Footer
    st.markdown("""
    <div class="sb-footer">
        <div class="sb-footer-row">Built by : <span>Harsh Prajapati</span></div>
        <div class="sb-footer-row">Model : <span>DenseNet121</span></div>
        <div class="sb-footer-row">Version : <span>1.0.0</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    
col1, col2, col3 = st.columns([1,1,1],gap="medium")
with col1:
    st.markdown("""
        <div style="background-color: #333130; padding: 10px; border-radius: 10px;">
            <h4>Disease Classes</h4>
            <h5 style="color: #4caf50;">14</h5>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style="background-color: #333130; padding: 10px; border-radius: 10px;">
            <h4>Training Images</h4>
            <h5 style="color: #4caf50;">123,000+</h5>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style="background-color:#333130; padding: 10px; border-radius: 10px;">
            <h4>Avg. ROC-AUC</h4>
            <h5 style="color: #4caf50;">0.95</h5>
        </div>
    """, unsafe_allow_html=True)

# Load model once — cache it so it doesn't reload on every rerun
@st.cache_resource
def get_model():
    return load_model("best_model.pth")

def run_inference(image: Image.Image, threshold=0.5):
    model = get_model()
    transform = get_transform()
    
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    
    with torch.inference_mode():
        logits = model(input_tensor)
        probs  = torch.sigmoid(logits).squeeze()
    
    probs_np = probs.cpu().numpy()
    scores = {ALL_LABELS[i]: float(probs_np[i]) for i in range(len(ALL_LABELS))}
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def real_gradcam(image: Image.Image) -> Image.Image:
    model = get_model()
    transform = get_transform()

    input_tensor = transform(image.convert("RGB")).unsqueeze(0)

    # Get top predicted class
    with torch.inference_mode():
        logits = model(input_tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    top_idx = int(probs.argmax())

    # Generate heatmap
    gcam    = GradCAM(model)
    heatmap = gcam.generate(input_tensor, class_idx=top_idx)

    # Overlay on original image
    img_cv  = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    result          = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    result_rgb      = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


col_left, col_right = st.columns([1, 1.1], gap="medium")

with col_left:

    # Upload card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
    <div class="upload-zone-icon">📂</div>
    <h4>Drop your X-ray here</h4>
    <p>PNG, JPG, JPEG · max 10 MB</p>
    </div>
    """, unsafe_allow_html=True)

   
    uploaded_file = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖼 Preview</div>', unsafe_allow_html=True)

    if uploaded_file:
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
            No image uploaded yet
        </div>
        <table class="meta-table">
            <tr><td class="meta-label">Filename</td><td class="meta-val">—</td></tr>
            <tr><td class="meta-label">File size</td><td class="meta-val">—</td></tr>
            <tr><td class="meta-label">Dimensions</td><td class="meta-val">—</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Run button
    run_clicked = st.button("⚙ Run prediction", disabled=(uploaded_file is None))
    
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    ph_left, ph_right = st.columns([1.4, 1])
    with ph_left:
        st.markdown('<div class="card-title">📋 Predictions</div>', unsafe_allow_html=True)
    with ph_right:
        if st.session_state.get("predictions"):
            st.markdown('<div class="findings-badge">Findings detected</div>',
                        unsafe_allow_html=True)

    if run_clicked and uploaded_file:
        image = Image.open(uploaded_file)
        with st.spinner("Running inference…"):
            scores = run_inference(image)
            gradcam_img = real_gradcam(image)
            st.session_state["gradcam"] = gradcam_img
        st.session_state["predictions"] = scores
    scores = st.session_state.get("predictions", {})

    if scores:
        st.markdown('<div class="sub-text">Confidence scores · top-5 by score</div>',
                    unsafe_allow_html=True)
        for name, score in list(scores.items())[:5]:
            pct = int(score * 100)
            if score >= 0.5:
                cls = "bar-fill-red"
                dot_color = "#f87171"
            elif score >= 0.25:
                cls = "bar-fill-amber"
                dot_color = "#fb923c"
            else:
                cls = "bar-fill-green"
                dot_color = "#34d399"
            st.markdown(f"""
            <div class="disease-row">
                <div class="pred-dot" style="background:{dot_color};"></div>
                <div class="disease-name">{name}</div>
                <div class="bar-track">
                    <div class="{cls}" style="width:{pct}%"></div>
                </div>
                <div class="bar-pct">{score:.2f}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="sub-text">+ {len(scores)-5} more classes</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="sub-text">Upload an image and click Run prediction.</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
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
        <p class="gradcam-caption">Overlay via st.image() with matplotlib · highlights model focus regions</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-empty">
            <span>🔬</span>
            Grad-CAM renders after prediction
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)