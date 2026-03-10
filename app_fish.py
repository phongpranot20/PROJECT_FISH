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
st.set_page_config(page_title="Fish Species Analysis", layout="wide", page_icon="🐠")

# --- 2. Custom CSS (Forced White Theme & Overlay Positioning) ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label { color: #262730 !important; }
    
    /* Analysis Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white !important; border: none; padding: 15px 30px;
        font-size: 22px; font-weight: bold; border-radius: 12px;
        width: 100%; transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    
    /* Container สำหรับซ้อนรูปปลาในกราฟ */
    .chart-container {
        position: relative;
        text-align: center;
        color: white;
    }
    .fish-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 120px; /* ปรับขนาดรูปปลาในวงกลม */
        z-index: 10;
        pointer-events: none; /* เพื่อให้ยังกดคลิกที่กราฟได้ */
    }
    </style>
    """, unsafe_allow_html=True)

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'

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

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear History Data"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.rerun()

# --- 4. Main UI ---
st.title("🐠 Fish Species Analysis")

uploaded_files = st.file_uploader("Select images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.subheader(f"📸 Image Preview ({len(uploaded_files)})")
    with st.container(height=350, border=True):
        cols = st.columns(6)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 6]:
                st.image(Image.open(file), caption=file.name, use_container_width=True)

    if st.button('🚀 START ANALYSIS NOW'):
        results = []
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img = Image.open(file).convert('RGB').resize((180, 180))
            img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
            pred = model.predict(img_array, verbose=0)
            res_idx = np.argmax(pred[0])
            species = CLASS_NAMES[res_idx]
            results.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Filename': file.name, 'Species': species, 'Confidence': np.max(pred[0]) * 100
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
        save_to_csv(pd.DataFrame(results))
        st.success("✅ Analysis Complete!")
        st.balloons()

st.divider()

# --- 5. Insight Dashboard (Fish in Circle) ---
if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    st.header("📊 Insight Dashboard")
    
    m1, m2 = st.columns(2)
    m1.metric("Total Analyzed", f"{len(df)} Images")
    m2.metric("Average Accuracy", f"{df['Confidence'].mean():.2f}%")

    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        # หาสายพันธุ์ที่พบมากที่สุดมาโชว์ในวงกลม
        most_found = df['Species'].mode()[0] if not df.empty else 'Betta'
        fish_img_path = FISH_INFO[most_found]
        
        # กราฟวงกลม
        fig_pie = px.pie(df, names='Species', hole=0.6, title="Species Distribution")
        fig_pie.update_layout(showlegend=False) # ซ่อน Legend เพื่อความคลีน
        
        # ใช้ HTML ซ้อนรูปปลา
        st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # ดึงรูปจาก GitHub (ถ้าไฟล์อยู่ในหน้าแรก GitHub จะใช้ชื่อไฟล์ตรงๆ ได้เลยในกรณีรัน Local 
        # แต่บน Cloud ต้องระบุ URL ของรูป หรือใช้เทคนิคดึงภาพมาแสดง)
        if os.path.exists(fish_img_path):
            st.image(fish_img_path, width=150) # แสดงรูปข้างล่างแทนหาก Overlay มีปัญหาในบาง Browser
            st.write(f"**Most Found: {most_found}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.write("### 🐟 Detected Species Details")
        grid = st.columns(3)
        for i, species in enumerate(CLASS_NAMES):
            count = len(df[df['Species'] == species])
            with grid[i % 3]:
                if os.path.exists(FISH_INFO[species]):
                    st.image(FISH_INFO[species], width=100)
                st.markdown(f"**{species}**")
                st.caption(f"Count: {count}")

    st.subheader("📝 Detailed Logs")
    st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
