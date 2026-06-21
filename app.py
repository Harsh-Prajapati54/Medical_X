import streamlit as st
# Global CSS
st.markdown("""
            <style>
            [data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #2a2a2a !important;
}

.sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 20px 16px 16px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 8px;
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
.sb-logo-sub  { font-size: 15px; color: #666; margin-top: 1px; }
.sb-section-label {
    font-size: 20px;
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
    font-size: 16px !important;
    font-weight: 400 !important;
    text-align: center !important;
    padding: 9px 16px !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
}
</style>
""" ,unsafe_allow_html=True)
            
st.title("Medical-X")
st.write("Welcome to the Medical-X application! This app allows you to analyze medical X-ray images and predict the chest disease using machine learning models.")
st.set_page_config(
    page_title="Med-X · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
)
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🫁</div>
        <div>
            <div class="sb-logo-text">Med-X</div>
            <div class="sb-logo-sub">Chest X-ray AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state["page"] = "Analyze"    

    st.markdown('<div class="sb-section-label">Main</div>', unsafe_allow_html=True)

    if st.button("🔬  Analyze X-ray", key="nav_analyze"):
        st.session_state["page"] = "Analyze"

    if st.button("🕓  History", key="nav_history"):
        st.session_state["page"] = "History"

    st.markdown('<div class="sb-section-label">Info</div>', unsafe_allow_html=True)

    if st.button("📖  About model", key="nav_about"):
        st.session_state["page"] = "About"

    if st.button("❓  How to use", key="nav_howto"):
        st.session_state["page"] = "HowTo"
    
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