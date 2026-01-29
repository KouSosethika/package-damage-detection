import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page settings
st.set_page_config(page_title="Package Damage Detection", layout="centered")

st.title("📦 Package Damage Detection")
st.write("Upload an image of a package to check whether it is **Damaged** or **Intact**.")

# ===== Path handling =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT: MODEL_PATH must be the folder containing saved_model.pb + variables/
MODEL_PATH = os.path.join(BASE_DIR, "model.savedmodel")  # <-- folder name 'model.savedmodel'
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ===== Load model safely =====
if not os.path.exists(MODEL_PATH):
    st.error(f"Model folder not found at {MODEL_PATH}. Upload the full SavedModel folder.")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# ===== Load labels =====
if not os.path.exists(LABELS_PATH):
    st.error(f"Labels file not found at {LABELS_PATH}.")
else:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

# ===== Image uploader =====
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and os.path.exists(MODEL_PATH):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Output
    st.subheader("🔍 Prediction Result")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence:** {confidence_score * 100:.2f}%")

    if "damaged" in class_name.lower():
        st.error("⚠️ The package is DAMAGED")
    else:
        st.success("✅ The package is INTACT")
