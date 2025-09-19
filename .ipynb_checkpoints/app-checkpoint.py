# app.py (final)
import os, json, traceback
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# No chdir needed: run this file from the project folder.

# --------------- Page setup ---------------
st.set_page_config(page_title="AI Soil Health Analyzer", page_icon="🌱")
st.title("AI Soil Health Analyzer")
st.caption("Upload a soil photo to classify its type and get quick crop suggestions.")

# --------------- Helpers ---------------
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model("soil_cnn_best.h5")
        with open("class_indices.json") as f:
            class_idx = json.load(f)
        labels = list(class_idx.keys())
        return model, labels, None
    except Exception:
        return None, None, traceback.format_exc()

def preprocess(img: Image.Image, size=(128,128)):
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# --------------- Load model ---------------
model, labels, err = load_artifacts()
if err:
    st.error("Failed to load model/labels. Ensure soil_cnn_best.h5 and class_indices.json are in the same folder as app.py.")
    st.code(err)
    st.stop()
else:
    st.success("Model loaded.")

# Default crop suggestions (edit per your classes)
CROP_SUGG = { lab: ["Wheat","Rice","Sugarcane"] for lab in labels }
if len(labels) >= 4:
    CROP_SUGG[labels[1]] = ["Maize","Potato","Pulses"]
    CROP_SUGG[labels[2]] = ["Millet","Gram","Cotton"]
    CROP_SUGG[labels[3]] = ["Horticulture","Cactus","Legumes"]

# --------------- Uploader / Samples ---------------
file = st.file_uploader("Upload soil image (JPG/PNG)", type=["jpg","jpeg","png"])

if os.path.isdir("samples"):
    samples = [f for f in os.listdir("samples") if f.lower().endswith((".jpg",".jpeg",".png"))]
else:
    samples = []

if samples:
    with st.expander("Or pick a sample from ./samples"):
        s = st.selectbox("Samples", ["-- select --"] + samples)
        if s != "-- select --":
            file = os.path.join("samples", s)

# --------------- Predict ---------------
if file:
    img = Image.open(file) if isinstance(file, str) else Image.open(file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess(img)[None, ...]
    y = model.predict(x, verbose=0)[0]
    top = int(np.argmax(y))
    pred = labels[top]
    conf = float(y[top])

    st.subheader(f"Prediction: {pred}")
    st.metric("Confidence", f"{conf*100:.1f}%")
    st.write("Suggested crops:", ", ".join(CROP_SUGG.get(pred, [])))

    st.write("Class probabilities")
    st.bar_chart({labels[i]: float(y[i]) for i in range(len(labels))})

# --------------- Footer ---------------
with st.expander("About"):
    st.write("CNN trained on soil images, input 128×128 RGB scaled to 0–1. For education and preliminary screening.")
