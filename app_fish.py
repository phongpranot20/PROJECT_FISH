import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown

# --- 1. Page Config & Modern UI Theme ---
st.set_page_config(page_title="Fish AI Analysis", layout="wide", page_icon="🐠")

st.markdown("""
    <style>
    /* บังคับธีมสีขาวและพื้นหลังแอปสีฟ้าอ่อนแบบในรูป */
    .stApp { background-color: #F0F4F8; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label { color: #262730 !important; }
    
    /* สไตล์ของ Card (กล่องสีขาว) */
    .card-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* สไตล์หัวข้อเมนูสีเทาอ่อน */
    .section-title {
        color: #8E9AAF;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    
    /* ปุ่มวิเคราะห์สีน้ำเงิน Gradient */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.4);
    }
    
    /* กล่อง About Model สีเข้ม (ฝั่งขวาล่าง) */
    .about-box {
        background-color: #0F172A;
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 15px;
        font-size: 14px;
        margin-top: 1rem;
    }
    
    /* ซ่อนส่วนเกินของ Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. Configuration & Model Loading ---
HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'

# แมตช์ชื่อไฟล์รูปที่คุณอัปไว้หน้าแรก GitHub
FISH_INFO = {
    'Angelfish': 'angelfish.jpg',
    'Betta': 'betta.jpg',
    'Cichlidae': 'cichilde.jpg', 
    'Goldfish': 'goldfish.jpg',
    'Koifish': 'koifish.jpg',
    'Nenotetra': 'neontetra.jpg'
}
CLASS_NAMES = list(FISH_INFO.keys())

@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def save_to_csv(new_df):
    if not os.path.isfile(HISTORY_FILE):
        new_df.to_csv(HISTORY_FILE, index=False)
    else:
        old_df = pd.read_csv(HISTORY_FILE)
        pd.concat([old_df, new_df], ignore_index=True).to_csv(HISTORY_FILE, index=False)

model = load_my_model()

# --- 3. Main Layout (แบ่งซ้าย-ขวา ตามรูปตัวอย่าง) ---
col_left, col_right = st.columns([1, 1.3], gap="large")

# --- ฝั่งซ้าย: IMAGE SOURCE ---
with col_left:
    st.markdown('<p class="section-title">IMAGE SOURCE</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Select Fish Photo", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files:
            # พื้นที่เลื่อนดูรูป (Scrollable Preview)
            with st.container(height=300, border=True):
                img_cols = st.columns(3)
                for i, file in enumerate(uploaded_files):
                    with img_cols[i % 3]:
                        st.image(file, use_container_width=True)
        
        btn_classify = st.button("Classify Images ❯")
        st.markdown('</div>', unsafe_allow_html=True)

    # TECHNICAL INFO
    st.markdown('<p class="section-title">ⓘ TECHNICAL INFO</p>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        t1.write("**Backend**")
        t2.write("TensorFlow 2.16")
        t1.write("**Privacy**")
        t2.markdown("<span style='color:green'>100% Secure</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- ฝั่งขวา: CLASSIFICATION RESULTS & DASHBOARD ---
with col_right:
    st.markdown('<p class="section-title">CLASSIFICATION RESULTS</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        
        if uploaded_files and btn_classify:
            results = []
            status_text = st.empty()
            for i, file in enumerate(uploaded_files):
                status_text.text(f"🔍 Analyzing: {file.name}")
                img = Image.open(file).convert('RGB').resize((180, 180))
                img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
                pred = model.predict(img_array, verbose=0)
                res_idx = np.argmax(pred[0])
                results.append({
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Species': CLASS_NAMES[res_idx],
                    'Confidence': np.max(pred[0]) * 100
                })
            save_to_csv(pd.DataFrame(results))
            status_text.empty()
            st.success(f"Successfully analyzed {len(uploaded_files)} images!")

        # แสดง Dashboard ภายในกล่องผลลัพธ์
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            
            # กราฟวงกลม
            fig_pie = px.pie(df, names='Species', hole=0.5, title="Species Distribution")
            fig_pie.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # กราฟจุด (Scatter Plot)
            fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', title="Confidence History")
            fig_scatter.update_layout(height=250)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.write("🔍\n\nUpload and classify images to see results.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ABOUT THIS MODEL (กล่องสีเข้ม)
    st.markdown('<div class="about-box">', unsafe_allow_html=True)
    st.markdown("**ABOUT THIS MODEL**")
    st.write("Custom CNN model trained for 6 species. The classification provides real-time insights into your fish dataset distribution.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- ล่างสุด: HISTORY LOGS & SPECIES DETAILS ---
st.divider()
if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    
    # ส่วนโชว์รูปสายพันธุ์และจำนวนที่พบ
    st.markdown('<p class="section-title">📊 Species Overview</p>', unsafe_allow_html=True)
    grid = st.columns(6)
    for i, species in enumerate(CLASS_NAMES):
        count = len(df[df['Species'] == species])
        with grid[i]:
            img_path = FISH_INFO[species]
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.markdown(f"<center><b>{species}</b> ({count})</center>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ตารางประวัติ (History Logs อยู่ล่างสุดตามที่ขอ)
    st.markdown('<p class="section-title">📝 Detailed History Logs</p>', unsafe_allow_html=True)
    st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

# Sidebar Clear Button
if st.sidebar.button("🗑️ Clear All History"):
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    st.rerun()
