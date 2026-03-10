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
    # ID ไฟล์โมเดลของคุณจากลิงก์ล่าสุด
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # ตรวจสอบและดาวน์โหลด
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📦 กำลังดาวน์โหลดโมเดล AI (ขนาดใหญ่) กรุณารอสักครู่...'):
                # ใช้ gdown แบบข้ามหน้ายืนยันไฟล์ขนาดใหญ่
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ ดาวน์โหลดไม่สำเร็จ: {e}")
            return None

    # ตรวจสอบความถูกต้องของไฟล์ที่ดาวน์โหลดมา
    if os.path.exists(MODEL_PATH):
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        
        # ถ้าไฟล์เล็กกว่า 5MB แสดงว่าโหลดมาผิด (ปกติโมเดล .h5 ของคุณควรจะใหญ่กว่านี้)
        if file_size_mb < 5:
            st.error(f"⚠️ ไฟล์ที่โหลดมาผิดพลาด (ขนาดเพียง {file_size_mb:.2f} MB) ระบบจะลบไฟล์เสียทิ้ง")
            os.remove(MODEL_PATH)
            st.info("💡 กรุณากดปุ่ม 'R' บนคีย์บอร์ด หรือ Refresh หน้าเว็บเพื่อลองใหม่อีกครั้ง")
            return None
            
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ โครงสร้างโมเดลผิดพลาด: {e}")
            os.remove(MODEL_PATH) # ลบไฟล์ที่เสียเพื่อป้องกัน Error ค้าง
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
    st.warning("⚠️ กำลังรอการติดตั้งโมเดล... หากค้างนานเกินไป กรุณาตรวจสอบลิงก์ Google Drive")
else:
    # --- ส่วนอัปโหลด ---
    uploaded_files = st.file_uploader("เลือกรูปภาพปลา (อัปโหลดได้หลายรูป)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 รูปภาพที่อัปโหลด ({len(uploaded_files)} รูป)")
        with st.container(height=300):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    img_display = Image.open(file)
                    st.image(img_display, caption=file.name, use_container_width=True)

        if st.button('🚀 เริ่มการวิเคราะห์', type="primary"):
            results = []
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner('AI กำลังวิเคราะห์...'):
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"🔍 วิเคราะห์: {file.name}")
                    img = Image.open(file).convert('RGB')
                    processed_img = img.resize((180, 180))
                    img_array = tf.keras.utils.img_to_array(processed_img)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    pred = model.predict(img_array, verbose=0)
                    res_idx = np.argmax(pred[0])
                    confidence = np.max(pred[0]) * 100
                    
                    results.append({
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Filename': file.name,
                        'Species': CLASS_NAMES[res_idx],
                        'Confidence': confidence
                    })
                    progress_bar.progress((i + 1) / len(uploaded_files))

            save_to_csv(pd.DataFrame(results))
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ วิเคราะห์เสร็จสิ้น!")
            st.balloons()

    st.divider()

    # --- ส่วน Dashboard ---
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            st.header("📊 Dashboard")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("จำนวนรูปทั้งหมด", f"{len(df)} รูป")
            with m2:
                st.metric("ความแม่นยำเฉลี่ย", f"{df['Confidence'].mean():.2f}%")

            c1, c2 = st.columns([1, 1.2])
            with c1:
                fig_pie = px.pie(df, names='Species', title="สัดส่วนที่วิเคราะห์ได้", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', title="กราฟความมั่นใจรายครั้ง")
                st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander("📝 ดูประวัติแบบตาราง"):
                st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

            if st.sidebar.button("🗑️ ล้างประวัติทั้งหมด"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                    st.rerun()
        except Exception:
            st.error("ไม่สามารถแสดงผล Dashboard ได้")
