import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
with open("corn_or_maize.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------------
# App Title & Description
# ------------------------------
st.title("🌽 Corn / Maize Leaf Disease Detection")
st.markdown(
    """
    **An ML-based image classification system** that detects diseases in corn leaves  
    using image processing and a trained **Decision Tree Classifier**.
    """
)

st.divider()

# ------------------------------
# Sidebar Info (Looks very professional)
# ------------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.markdown(
    """
    **Problem Type:** Image Classification  
    **Model Used:** Decision Tree Classifier  
    **Input:** Corn leaf image  
    **Output:** Disease class  
    """
)

st.sidebar.markdown("### 🌱 Disease Classes")
st.sidebar.markdown(
    """
    - Common Rust  
    - Gray Leaf Spot  
    - Blight  
    - Healthy  
    """
)

st.sidebar.markdown("### 🧠 Image Size Used")
st.sidebar.markdown("300 × 300 (RGB)")

# ------------------------------
# Image Upload Section
# ------------------------------
st.subheader("📤 Upload Corn Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------
# Prediction Section
# ------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image")

    # Preprocessing (must match training)
    img = np.array(image)
    img = cv2.resize(img, (300, 300))
    img = img.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img)
    predicted_class = prediction[0]

    st.subheader("🔍 Prediction Result")
    st.success(f"🩺 Detected Disease: **{predicted_class}**")



# ------------------------------
# Footer
# ------------------------------
st.divider()
st.markdown(
    """
    👨‍💻 **Developed by:** Mahesh Dugyani  
    🚀 **Deployed using:** Streamlit & Hugging Face Spaces  
    📌 **Domain:** Machine Learning | Image Classification
    """
)
