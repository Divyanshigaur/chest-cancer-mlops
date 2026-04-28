import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Page config
st.set_page_config(page_title="Chest Cancer Detection", layout="wide")

# ---------- PROFESSIONAL STYLING ----------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }

    .main-title {
        font-size: 32px;
        font-weight: 600;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 20px;
    }

    .section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }

    .result-box {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 8px;
        color: #1f4e79;
        font-weight: 500;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>Chest Cancer Detection System</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Model Configuration")

model_option = st.sidebar.selectbox(
    "Select Model",
    ("CNN", "MobileNet", "EfficientNet")
)

# Load model
@st.cache_resource
def load_model(model_name):
    if model_name == "MobileNet":
        return tf.keras.models.load_model("models/mobilenet.h5")
    elif model_name == "EfficientNet":
        return tf.keras.models.load_model("models/efficientnet.h5")
    else:
        return tf.keras.models.load_model("models/cnn.h5")

model = load_model(model_option)

IMG_SIZE = (224, 224)
classes = ["ACA", "Normal", "SCC"]

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Upload Histopathology Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("Diagnosis Result")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # 🔥 Split into sub-columns inside result section
        img_col, result_col = st.columns([1, 1])

        with img_col:
            st.image(image, caption="Input Image", width=250)  # 👈 FIXED SIZE

        with result_col:
            image_resized = image.resize(IMG_SIZE)
            image_array = np.array(image_resized)

            # Preprocessing
            if model_option == "EfficientNet":
                image_array = preprocess_input(image_array)
            else:
                image_array = image_array / 255.0

            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Result box
            st.markdown(f"""
            <div class='result-box'>
            <b>Prediction:</b> {classes[predicted_class]} <br>
            <b>Confidence:</b> {confidence:.2f}
            </div>
            """, unsafe_allow_html=True)

            # Probabilities
            st.subheader("Class Probabilities")
            probs = prediction[0]

            for i, prob in enumerate(probs):
                st.write(f"{classes[i]}: {prob:.2f}")
                st.progress(float(prob))

    else:
        st.info("Upload an image to generate prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer / Disclaimer
st.markdown("""
---
⚠️ **Disclaimer:** This system is for research and educational purposes only. It is not intended for clinical diagnosis.
""")