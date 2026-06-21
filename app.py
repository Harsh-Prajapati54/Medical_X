import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Med-X · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
)
# Global CSS

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

def run_inference(image):
    mock = {
        "Infiltration": 0.72, "Effusion": 0.58,
        "Atelectasis": 0.41,  "No Finding": 0.19,
        "Pneumonia": 0.12,    "Cardiomegaly": 0.08,
        "Mass": 0.06,         "Nodule": 0.05,
        "Consolidation": 0.04,"Emphysema": 0.03,
        "Pleural Thickening": 0.02, "Hernia": 0.01,
        "Edema": 0.01,        "Fibrosis": 0.01,
    }
    return dict(sorted(mock.items(), key=lambda x: x[1], reverse=True))

col_left, col_right = st.columns([1, 1.1], gap="medium")

with col_left:

    # Upload card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⬆ Upload X-ray image</div>', unsafe_allow_html=True)
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
        st.session_state["predictions"] = scores

    scores = st.session_state.get("predictions", {})

    if scores:
        st.markdown('<div class="sub-text">Confidence scores · top-5 by score</div>',
                    unsafe_allow_html=True)
        for name, score in list(scores.items())[:5]:
            pct = int(score * 100)
            if score >= 0.5:   cls = "bar-fill-red"
            elif score >= 0.25: cls = "bar-fill-amber"
            else:               cls = "bar-fill-green"
            st.markdown(f"""
            <div class="disease-row">
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
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="preview-empty">
            <span>🔬</span>
            Grad-CAM renders after prediction
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)