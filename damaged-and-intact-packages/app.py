import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ========================
# Page configuration
# ========================
st.set_page_config(
    page_title="📦 Package Damage Detection",
    page_icon="📦",
    layout="centered",
)

st.title("📦 Package Damage Detection")
st.markdown(
    "Detect whether a package is **Damaged** or **Intact** using image upload or **live webcam**."
)

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "model.savedmodel")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ========================
# Load model (ONCE)
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
def predict_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    index = np.argmax(preds)
    return class_names[index], float(preds[0][index])

# ========================
# Sidebar selector
# ========================
mode = st.sidebar.radio(
    "Select Input Mode",
    ["📁 Image Upload", "📹 Live Webcam"]
)

# ========================
# IMAGE UPLOAD MODE
# ========================
if mode == "📁 Image Upload":
    uploaded_file = st.file_uploader(
        "Upload a package image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        label, confidence = predict_image(image)

        st.markdown("### 🔍 Prediction Result")
        if "damaged" in label.lower():
            st.error(f"⚠️ DAMAGED — {confidence*100:.2f}%")
        else:
            st.success(f"✅ INTACT — {confidence*100:.2f}%")

# ========================
# LIVE WEBCAM MODE
# ========================
else:
    st.markdown("### 📹 Live Webcam Detection")
    st.info("Prediction updates every few frames for better stability.")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.label = "Detecting..."
            self.confidence = 0.0

        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="rgb24")

            # Predict every 10 frames (IMPORTANT)
            if self.frame_count % 10 == 0:
                pil_img = Image.fromarray(img)
                self.label, self.confidence = predict_image(pil_img)

            # Draw overlay
            color = (255, 0, 0) if "damaged" in self.label.lower() else (0, 255, 0)
            text = f"{self.label} ({self.confidence*100:.1f}%)"

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
        key="package-damage-live",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("📦 AI-powered Package Damage Detection • Live Webcam Enabled")
