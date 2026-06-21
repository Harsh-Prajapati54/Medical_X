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
    
col4, col5 = st.columns([1,1],gap="medium")
with col4:
    st.markdown('<div style="background:#f8f9fa; padding:16px; border-radius:10px;">', unsafe_allow_html=True)
    
    st.subheader("Upload X-ray")
    uploaded_file = st.file_uploader("Choose file", type=["png","jpg","jpeg"])
    
    st.markdown('</div>', unsafe_allow_html=True)