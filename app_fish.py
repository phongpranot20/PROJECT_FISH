import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Fish Species Analysis", layout="wide", page_icon="🐠")

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
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
    st.error("❌ ไม่พบโมเดล กรุณาตรวจสอบไฟล์ .h5 ในโฟลเดอร์เดียวกับสคริปต์")
else:
    # --- ส่วนอัปโหลด ---
    uploaded_files = st.file_uploader("เลือกรูปภาพปลา (Multiple)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"Loading Image... ({len(uploaded_files)} รูป)")
        with st.container(height=300):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    img_display = Image.open(file)
                    st.image(img_display, caption=file.name, use_container_width=True)

        st.info(f"📁 Ready to process all {len(uploaded_files)} รายการ")
        
        if st.button('🚀 Start The Analysis', type="primary"):
            results = []
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner('กำลังวิเคราะห์รูปภาพ...'):
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"กำลังประมวลผล: {file.name} ({i+1}/{len(uploaded_files)})")
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

    # --- ส่วน Dashboard (ลบสายพันธุ์ที่พบมากที่สุดออกแล้ว) ---
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            st.header("📊 Dashboard สรุปผลการวิเคราะห์สะสม")
            
            # แสดงเฉพาะ Metric พื้นฐานที่จำเป็น
            m1, m2 = st.columns(2)
            with m1:
                st.metric("จำนวนรูปภาพทั้งหมด", f"{len(df)} รูป")
            with m2:
                st.metric("ความแม่นยำเฉลี่ย", f"{df['Confidence'].mean():.2f}%")

            st.write("") # เว้นวรรคเล็กน้อย

            c1, c2 = st.columns([1, 1.2])
            with c1:
                fig_pie = px.pie(df, names='Species', title="สัดส่วนสายพันธุ์ปลาทั้งหมด", hole=0.4)
                fig_pie.update_layout(font=dict(size=18), title_font=dict(size=22))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                fig_line = px.scatter(df, x='Timestamp', y='Confidence', color='Species', 
                                    hover_data=['Filename'],
                                    title="ระดับความมั่นใจในแต่ละการวิเคราะห์")
                fig_line.update_layout(
                    height=500, 
                    font=dict(size=16),
                    title_font=dict(size=22),
                    xaxis_title="เวลา",
                    yaxis_title="ความมั่นใจ (%)"
                )
                st.plotly_chart(fig_line, use_container_width=True)

            with st.expander("📝 ดูตารางข้อมูลประวัติทั้งหมด"):
                st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

            if st.sidebar.button("🗑️ ล้างประวัติข้อมูลทั้งหมด"):
                os.remove(HISTORY_FILE)
                st.rerun()

        except Exception as e:
            st.error("พบปัญหาในการอ่านข้อมูลประวัติ")
            if st.sidebar.button("🔧 ซ่อมแซมไฟล์ประวัติ"):
                os.remove(HISTORY_FILE)
                st.rerun()