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
    Upload an image of a package and let the AI predict whether it is **Damaged** or **Intact**.
    """
)

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "model.savedmodel")  # Folder containing saved_model.pb + variables/
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ========================
# Load model
# ========================
model = None
if not os.path.exists(MODEL_FOLDER):
    st.error(f"❌ Model folder not found at {MODEL_FOLDER}. Please upload the full SavedModel folder.")
else:
    try:
        with st.spinner("⏳ Loading model..."):
            model = tf.keras.models.load_model(MODEL_FOLDER)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

# ========================
# Load labels
# ========================
class_names = []
if not os.path.exists(LABELS_PATH):
    st.error(f"❌ Labels file not found at {LABELS_PATH}.")
else:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

# ========================
# Upload image
# ========================
uploaded_file = st.file_uploader(
    "Choose an image of the package...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None and class_names:
    try:
        # Read image
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Display uploaded image
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # ========================
        # Preprocess image
        # ========================
        input_size = (224, 224)  # Teachable Machine default
        image_resized = image.resize(input_size)
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # ========================
        # Make prediction
        # ========================
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # ========================
        # Display result nicely
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
        st.info("💡 Tip: Make sure the image shows the package clearly for best results.")

    except Exception as e:
        st.error(f"❌ Failed to process uploaded image: {e}")
