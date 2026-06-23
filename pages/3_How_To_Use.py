import streamlit as st

st.set_page_config(page_title="How to Use · Med-X", layout="wide")

st.title("How to Use Med-X")
st.caption("A step-by-step guide to analyzing chest X-rays")

steps = [
    ("1", "📂", "Upload your X-ray",
     "Click **Browse files** or drag and drop a chest X-ray image onto the upload zone.\n\n"
     "**Accepted formats:** PNG, JPG, JPEG · **Max file size:** 10 MB\n\n"
     "For best results, use a standard PA (Posterior-Anterior) chest X-ray image."),

    ("2", "🔍", "Review the preview",
     "Once uploaded, the image appears in the **Preview** panel. "
     "Check that it's correctly oriented and the full chest is visible.\n\n"
     "File metadata (name, size, dimensions) is shown below the preview."),

    ("3", "🔬", "Run prediction",
     "Click the **Run prediction** button. The model runs DenseNet121 inference "
     "and computes Grad-CAM on the last dense block.\n\n"
     "This takes 3–8 seconds depending on hardware."),

    ("4", "📊", "Read the predictions",
     "The **Predictions** panel shows confidence scores per disease class (sigmoid output).\n\n"
     "- 🔴 **Red (≥ 0.50):** High confidence finding\n"
     "- 🟠 **Orange (0.30–0.49):** Moderate signal\n"
     "- 🟢 **Green (< 0.30):** Low / unlikely\n\n"
     "Top 5 classes are shown by score. A **Findings detected** badge appears if any class scores above 0.3."),

    ("5", "🌡️", "Interpret Grad-CAM",
     "The **Grad-CAM heatmap** overlays a heat map on the original X-ray, "
     "highlighting regions the model paid the most attention to.\n\n"
     "Red/orange zones = high attention. Green = low attention. "
     "This helps understand *why* the model flagged a region."),

    ("6", "📜", "Check history",
     "Every prediction is saved to the **History** page. "
     "You can review past scans, compare results, and track confidence scores over sessions."),
]

for num, icon, title, body in steps:
    with st.container():
        col_num, col_body = st.columns([0.08, 1])
        with col_num:
            st.markdown(f"""
            <div style="width:36px;height:36px;border-radius:50%;
                        background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.85rem;font-weight:700;color:#e0e0e0;margin-top:4px;">
                {num}
            </div>
            """, unsafe_allow_html=True)
        with col_body:
            st.markdown(f"#### {icon} {title}")
            st.markdown(body)
        st.markdown("---")

st.subheader("⚠️ Disclaimer")
st.warning(
    "Med-X is a **research and educational tool only**. "
    "It is not a certified medical device and should not be used for clinical diagnosis. "
    "Always consult a qualified radiologist or physician for medical decisions."
)

st.subheader("💡 Tips for best results")
st.markdown("""
- Use **original DICOM exports** converted to PNG (not photos of screens)
- Ensure the image is **not cropped** — full lung fields should be visible
- Avoid images with **watermarks or text overlays**
- The model was trained on adult chest X-rays — **paediatric X-rays** may show lower accuracy
""")