import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.savedmodel")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# ========================
# Load model
# ========================
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.")

# ========================
# Load labels
# ========================
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ========================
# Open webcam
# ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("📹 Live webcam started. Press 'Q' to quit.")

# ========================
# Live loop
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    img = Image.fromarray(rgb_frame).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = prediction[0][index]

    # Display live result on frame
    text = f"{label}: {confidence*100:.1f}%"
    color = (0, 0, 255) if "damaged" in label.lower() else (0, 255, 0)

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("📦 Package Damage Detection - Live Webcam", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
