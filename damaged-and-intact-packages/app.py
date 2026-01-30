import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="📦 Package Damage Detection",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Package Damage Detection (Live Camera)")
st.markdown(
    "Detect whether a package is **Damaged** or **Intact** using **live webcam** or image upload."
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

model = load_model()

# ========================
# Load labels
# ========================
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ========================
# Prediction function
# ========================
def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

# ========================
# Sidebar mode selector
# ========================
mode = st.sidebar.radio(
    "Choose Input Mode",
    ["📁 Upload Image", "📹 Live Camera"]
)

# ========================
# IMAGE UPLOAD MODE
# ========================
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        label, confidence = predict_image(image)

        st.markdown("### 🔍 Prediction")
        if "damaged" in label.lower():
            st.error(f"⚠️ DAMAGED ({confidence*100:.2f}%)")
        else:
            st.success(f"✅ INTACT ({confidence*100:.2f}%)")

# ========================
# LIVE CAMERA MODE
# ========================
else:
    st.markdown("### 📹 Live Webcam Detection")
    st.info("Point the camera at a package. Prediction updates in real time.")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="rgb24")
            pil_img = Image.fromarray(img)

            label, confidence = predict_image(pil_img)

            # Draw result text
            import cv2
            color = (255, 0, 0) if "damaged" in label.lower() else (0, 255, 0)
            text = f"{label} ({confidence*100:.1f}%)"

            cv2.putText(
                img,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(img, format="rgb24")

    webrtc_streamer(
        key="live-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("📦 AI-powered Package Damage Detection using Live Webcam")
