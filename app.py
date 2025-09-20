# app.py
import os, json, random
import numpy as np
import streamlit as st
from PIL import Image

# --------------- Page setup ---------------
st.set_page_config(page_title="AI Soil Health Analyzer (Mock)", page_icon="🌱")
st.title("🌱 AI Soil Health Analyzer (Mock)")
st.caption("Upload a soil photo to classify its type and get quick crop suggestions.")

# --------------- Fake Model & Labels ---------------
FAKE_LABELS = ["Alluvial", "Black", "Red", "Laterite", "Arid"]

def fake_predict(img_array):
    """Simulate a fake model prediction."""
    probs = np.random.dirichlet(np.ones(len(FAKE_LABELS)), size=1)[0]
    top_index = int(np.argmax(probs))
    pred = FAKE_LABELS[top_index]
    conf = float(probs[top_index])
    return pred, conf, dict(zip(FAKE_LABELS, map(float, probs)))

# Crop suggestions (you can edit this)
CROP_SUGG = {
    "Alluvial": ["Wheat", "Rice", "Sugarcane"],
    "Black": ["Cotton", "Soybean", "Sunflower"],
    "Red": ["Millet", "Groundnut", "Pulses"],
    "Laterite": ["Tea", "Coffee", "Cashew"],
    "Arid": ["Cactus", "Dates", "Barley"]
}

# --------------- Preprocessing (just to simulate real flow) ---------------
def preprocess(img: Image.Image, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# --------------- Upload / Sample Selection ---------------
file = st.file_uploader("Upload soil image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if os.path.isdir("samples"):
    samples = [f for f in os.listdir("samples") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    samples = []

if samples:
    with st.expander("Or pick a sample from ./samples"):
        s = st.selectbox("Samples", ["-- select --"] + samples)
        if s != "-- select --":
            file = os.path.join("samples", s)

# --------------- Prediction ---------------
if file:
    img = Image.open(file) if isinstance(file, str) else Image.open(file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    arr = preprocess(img)
    pred, conf, all_probs = fake_predict(arr)

    st.subheader(f"🧠 Predicted Soil Type: {pred}")
    st.metric("Confidence", f"{conf * 100:.1f}%")
    st.write("🌾 Suggested Crops:", ", ".join(CROP_SUGG.get(pred, [])))

    st.write("📊 Class Probabilities")
    st.bar_chart(all_probs)

# --------------- Footer ---------------
with st.expander("About"):
    st.write("🚨 This is a mock version for UI testing only. No real machine learning is used.")
