import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown

# --- 1. Page Config ---
st.set_page_config(page_title="Fish AI Classifier", layout="wide", page_icon="🐠")

# --- 2. Advanced CSS for Custom UI (เลียนแบบรูปที่คุณส่งมา) ---
st.markdown("""
    <style>
    /* บังคับพื้นหลังสีฟ้าอ่อน/เทา ตามรูป */
    .stApp {
        background-color: #F0F4F8;
    }
    
    /* สไตล์ของ Card (กล่องขาวๆ) */
    [data-testid="stVerticalBlock"] > div:has(div.card-box) {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* สไตล์หัวข้อ */
    .section-title {
        color: #8E9AAF;
        font-size: 14px;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    /* สไตล์ปุ่ม Classify Image */
    div.stButton > button:first-child {
        background-color: #E2E8F0;
        color: #94A3B8;
        border: none;
        border-radius: 10px;
        padding: 15px;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0072ff;
        color: white;
    }
    
    /* สไตล์กล่องดำ (About Model) */
    .about-box {
        background-color: #0F172A;
        color: #94A3B8;
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 14px;
    }
    
    /* ซ่อน Streamlit Header/Footer เพื่อความเนียน */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ฟังก์ชันโหลดโมเดลเดิม
@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    MODEL_PATH = 'fish_model_v3.h5'
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

# --- 3. Main Layout ---
col_left, col_right = st.columns([1, 1])

# --- ฝั่งซ้าย: IMAGE SOURCE ---
with col_left:
    st.markdown('<p class="section-title">IMAGE SOURCE</p>', unsafe_allow_html=True)
    
    # ส่วนอัปโหลด
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
        else:
            # จำลองปุ่ม Select Fish Photo
            st.info("⬆️ Select Fish Photo to start")
        
        btn_classify = st.button("Classify Image ❯")
        st.markdown('</div>', unsafe_allow_html=True)

    # ส่วน TECHNICAL INFO
    st.markdown('<p class="section-title">ⓘ TECHNICAL INFO</p>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        t1.write("**Architecture**")
        t2.write("CNN (Custom)")
        t1.write("**Backend**")
        t2.write("TensorFlow.py")
        t1.write("**Privacy**")
        t2.markdown("<span style='color:green'>100% Secure</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- ฝั่งขวา: CLASSIFICATION RESULTS ---
with col_right:
    st.markdown('<p class="section-title">CLASSIFICATION RESULTS</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-box" style="min-height: 400px; display: flex; align-items: center; justify-content: center; text-align: center;">', unsafe_allow_html=True)
        
        if uploaded_file and btn_classify:
            img = Image.open(uploaded_file).convert('RGB').resize((180, 180))
            img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
            pred = model.predict(img_array, verbose=0)
            res_idx = np.argmax(pred[0])
            species = CLASS_NAMES[res_idx]
            conf = np.max(pred[0]) * 100
            
            st.success(f"### {species}")
            st.write(f"Confidence: {conf:.2f}%")
        else:
            st.write("🔍\n\nUpload an image to see classification results")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ส่วน ABOUT THIS MODEL (กล่องดำ)
    st.markdown('<div class="about-box">', unsafe_allow_html=True)
    st.markdown("**ABOUT THIS MODEL**")
    st.write("This AI model is trained to recognize specific fish species using Deep Learning. Analysis runs on a secure environment without compromising your data.")
    st.markdown('</div>', unsafe_allow_html=True)
