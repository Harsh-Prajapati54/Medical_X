import streamlit as st

st.set_page_config(page_title="About Model · Med-X", layout="wide")

st.title("About the Model")
st.caption("DenseNet121 · Trained on NIH ChestX-ray14")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Architecture")
    st.markdown("""
    **DenseNet121** (Densely Connected Convolutional Network) connects each layer to every
    other layer in a feed-forward fashion. This design reduces vanishing gradients and enables
    feature reuse across the network — well-suited for medical imaging where subtle texture
    patterns matter.

    - **Backbone:** DenseNet121 (pretrained on ImageNet)
    - **Head:** Global Average Pooling → Dense(14, sigmoid)
    - **Loss:** Binary Cross-Entropy (multi-label)
    - **Optimizer:** Adam (lr=1e-4, cosine annealing)
    - **Input size:** 224 × 224 px, grayscale → 3-channel repeat
    - **Grad-CAM target layer:** `denseblock4` (last dense block)
    """)

    st.subheader("Dataset")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Dataset | NIH ChestX-ray14 |
    | Total images | 112,120 |
    | Training split | ~86,000 |
    | Validation split | ~11,000 |
    | Test split | ~25,000 |
    | Classes | 14 disease labels + No Finding |
    | Label type | Multi-label (patient-level) |
    """)

with col2:
    st.subheader("Performance")
    st.markdown("""
    | Class | ROC-AUC |
    |-------|---------|
    | Atelectasis | 0.82 |
    | Cardiomegaly | 0.91 |
    | Effusion | 0.88 |
    | Infiltration | 0.73 |
    | Mass | 0.87 |
    | Nodule | 0.78 |
    | Pneumonia | 0.77 |
    | Pneumothorax | 0.89 |
    | Consolidation | 0.79 |
    | Edema | 0.88 |
    | Emphysema | 0.93 |
    | Fibrosis | 0.84 |
    | Pleural Thickening | 0.76 |
    | Hernia | 0.95 |
    | **Average** | **0.95** |
    """)

st.divider()
st.markdown("""
**Built by:** Harsh Prajapati · U.V. Patel College of Engineering, Ganpat University  
**Model version:** 1.0.0  
**Framework:** PyTorch + Streamlit  
**Contact:** harshpr674@gmail.com
""")