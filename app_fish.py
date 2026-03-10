import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown  # สำหรับโหลดโมเดลจาก Google Drive

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Fish Species Analysis", layout="wide", page_icon="🐠")

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

@st.cache_resource
def load_my_model():
    # ID ไฟล์ใหม่ที่คุณส่งมา
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📦 กำลังดาวน์โหลดโมเดล AI จาก Google Drive (ครั้งแรกเท่านั้น)...'):
                # fuzzy=True ช่วยให้จัดการกับลิงก์ที่มีการยืนยันการสแกนไวรัสได้ดีขึ้น
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
            return None
            
    if os.path.exists(MODEL_PATH):
        # ตรวจสอบเบื้องต้นว่าไม่ใช่ไฟล์เสีย (ปกติโมเดล .h5 ควรมีขนาด > 1MB)
        if os.path.getsize(MODEL_PATH) < 1000000:
            st.error("⚠️ ไฟล์ที่โหลดมามีขนาดเล็กเกินไป อาจเกิดจาก Link ไม่ถูกต้อง หรือสิทธิ์การเข้าถึงไม่ได้เปิดเป็น Public")
            os.remove(MODEL_PATH) # ลบทิ้งเพื่อให้โหลดใหม่รอบหน้า
            return None
            
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ ไฟล์โมเดลไม่ถูกต้องหรือเสียหาย: {e}")
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
    st.error("⚠️ ระบบไม่พร้อมใช้งาน กรุณาตรวจสอบสถานะโมเดลด้านบน")
else:
    # --- ส่วนอัปโหลด ---
    uploaded_files = st.file_uploader("เลือกรูปภาพปลา (Multiple)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 รูปภาพที่อัปโหลด ({len(uploaded_files)} รูป)")
        with st.container(height=300):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    img_display = Image.open(file)
                    st.image(img_display, caption=file.name, use_container_width=True)

        st.info(f"📁 พร้อมประมวลผลไฟล์ทั้งหมด {len(uploaded_files)} รายการ")
        
        if st.button('🚀 เริ่มการวิเคราะห์', type="primary"):
            results = []
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner('กำลังวิเคราะห์...'):
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
            st.header("📊 Dashboard สรุปผล")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("จำนวนรูปทั้งหมด", f"{len(df)} รูป")
            with m2:
                st.metric("ความแม่นยำเฉลี่ย", f"{df['Confidence'].mean():.2f}%")

            st.write("") 

            c1, c2 = st.columns([1, 1.2])
            with c1:
                fig_pie = px.pie(df, names='Species', title="สัดส่วนปลาที่วิเคราะห์แล้ว", hole=0.4)
                fig_pie.update_layout(font=dict(size=16), title_font=dict(size=20))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', 
                                    hover_data=['Filename'], title="ระดับความมั่นใจรายครั้ง")
                fig_scatter.update_layout(
                    height=500, 
                    font=dict(size=14),
                    xaxis_title="วัน/เวลา",
                    yaxis_title="ความมั่นใจ (%)"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander("📝 ดูตารางประวัติ"):
                st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

            if st.sidebar.button("🗑️ ล้างประวัติทั้งหมด"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                    st.rerun()

        except Exception as e:
            st.error("พบปัญหาในการแสดงผล Dashboard")
