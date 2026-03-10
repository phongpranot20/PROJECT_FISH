import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Fish Species Analysis", layout="wide", page_icon="🐠")

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📦 กำลังดาวน์โหลดโมเดล AI...'):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ โหลดไม่สำเร็จ: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            # ใช้ compile=False เพื่อแก้ปัญหา Unrecognized keyword arguments ในบางเวอร์ชัน
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"❌ โครงสร้างโมเดลขัดข้อง: {e}")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            return None
    return None

def save_to_csv(new_df):
    if not os.path.isfile(HISTORY_FILE):
        new_df.to_csv(HISTORY_FILE, index=False)
    else:
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.to_csv(HISTORY_FILE, index=False)
        except Exception:
            new_df.to_csv(HISTORY_FILE, index=False)

model = load_my_model()

st.title("🐠 Fish Species Analysis")
st.write("อัปโหลดไฟล์ภาพเพื่อวิเคราะห์สายพันธุ์และดู Dashboard สรุปผล")

if model is None:
    st.warning("⚠️ กำลังเชื่อมต่อกับโมเดล AI... (ตรวจสอบสิทธิ์การแชร์ไฟล์ใน Drive)")
else:
    uploaded_files = st.file_uploader("เลือกรูปภาพปลา...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 รูปภาพที่อัปโหลด ({len(uploaded_files)} รูป)")
        with st.container(height=300):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

        if st.button('🚀 เริ่มการวิเคราะห์', type="primary"):
            results = []
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"🔍 วิเคราะห์: {file.name}")
                img = Image.open(file).convert('RGB').resize((180, 180))
                img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
                
                pred = model.predict(img_array, verbose=0)
                res_idx = np.argmax(pred[0])
                results.append({
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Filename': file.name,
                    'Species': CLASS_NAMES[res_idx],
                    'Confidence': np.max(pred[0]) * 100
                })
                progress_bar.progress((i + 1) / len(uploaded_files))

            save_to_csv(pd.DataFrame(results))
            status_text.empty()
            progress_bar.empty()
            st.success("✅ วิเคราะห์เสร็จสิ้น!")
            st.balloons()

    st.divider()
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.header("📊 Dashboard")
        m1, m2 = st.columns(2)
        m1.metric("จำนวนรูปทั้งหมด", f"{len(df)} รูป")
        m2.metric("ความแม่นยำเฉลี่ย", f"{df['Confidence'].mean():.2f}%")
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.plotly_chart(px.pie(df, names='Species', hole=0.4), use_container_width=True)
        with c2:
            st.plotly_chart(px.scatter(df, x='Timestamp', y='Confidence', color='Species'), use_container_width=True)
            
        if st.sidebar.button("🗑️ ล้างประวัติทั้งหมด"):
            os.remove(HISTORY_FILE)
            st.rerun()
