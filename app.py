import os
import sys
import pickle
import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="AI Defect Scanner", layout="centered")
st.title("🏭 Real-Time Factory Defect Detection Portal")

MODEL_PATH = "models/defect_detector.pkl"


@st.cache_resource
def load_detection_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_detection_model()

if model is None:
    st.error("⚠️ Active AI Model File Not Found! Run 'src/train.py' in your terminal first to generate it.")
else:
    st.success("🤖 Inspection AI System Status: Online and Ready.")
    
    uploaded_file = st.file_uploader("Insert factory sample image stream...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        display_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Incoming Pipeline Sample", use_container_width=True)
        
        # Format the incoming image to match what the AI expects
        resized_img = cv2.resize(opencv_img, (128, 128))
        normalized_img = resized_img.astype(np.float32) / 255.0
        input_flat = normalized_img.flatten().reshape(1, -1)
        
        # The model returns 1 for a normal item and -1 for a defect
        prediction = model.predict(input_flat)[0]
        
        # Calculate an anomaly score (lower means more abnormal)
        score = model.score_samples(input_flat)[0]
        
        st.markdown("### 📊 Inspection Diagnostics")
        st.metric(label="Calculated Structural Anomaly Score", value=f"{score:.5f}")
        
        if prediction == -1:
            st.error("🚨 DEFECT DETECTED: Item structural integrity outside tolerances.")
        else:
            st.success("✅ PASS: Sample conforms to production standards.")
