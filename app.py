import streamlit as st

st.title("Medical-X")
st.write("Welcome to the Medical-X application! This app allows you to analyze medical X-ray images and predict the chest disease using machine learning models.")

    
st.sidebar.radio( "Main ",["Home", "About"])

st.set_page_config(
    page_title="Med-X · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
)

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