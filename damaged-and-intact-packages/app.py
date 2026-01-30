import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from io import BytesIO

# ========================
# Page configuration
# ========================
st.set_page_config(
    page_title="📦 Package Damage Detection",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📦 Package Damage Detection")
st.markdown(
    """
    Upload an image **or use your camera** to let the AI predict whether a package is  
    **Damaged** or **Intact**.
    """
)

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "model.savedmodel")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ========================
# Load model
# ========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FOLDER)

model = None
if not os.path.exists(MODEL_FOLDER):
    st.error("❌ Model folder not found. Please upload the full SavedModel folder.")
else:
    try:
        with st.spinner("⏳ Loading model..."):
            model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

# ========================
# Load labels
# ========================
class_names = []
if not os.path.exists(LABELS_PATH):
    st.error("❌ Labels file not found.")
else:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

# ========================
# Image input section
# ========================
st.markdown("## 📸 Choose Input Method")

uploaded_file = st.file_uploader(
    "📁 Upload an image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("### OR")

camera_image = st.camera_input("📷 Use your camera")

# Decide image source
image_source = None
source_label = ""

if uploaded_file is not None:
    image_source = uploaded_file
    source_label = "Uploaded Image"
elif camera_image is not None:
    image_source = camera_image
    source_label = "Camera Capture"

# ========================
# Prediction
# ========================
if image_source is not None and model is not None and class_names:
    try:
        # Read image
        image_bytes = image_source.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Display image
        st.image(image, caption=f"📸 {source_label}", use_column_width=True)

        # ========================
        # Preprocess image
        # ========================
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # ========================
        # Predict
        # ========================
        with st.spinner("🔍 Analyzing image..."):
            prediction = model.predict(image_array, verbose=0)

        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # ========================
        # Display result
        # ========================
        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            if "damaged" in class_name.lower():
                st.error("⚠️ DAMAGED")
            else:
                st.success("✅ INTACT")

        with col2:
            st.write(f"**Class:** {class_name}")
            st.write(f"**Confidence:** {confidence_score * 100:.2f}%")

        st.markdown("---")
        st.info("💡 Tip: Use clear, well-lit images for best results.")

    except Exception as e:
        st.error(f"❌ Failed to process image: {e}")

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("📦 AI-powered Package Damage Detection")
