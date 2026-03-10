import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown

# --- 1. Page Config & Forced Light Theme ---
st.set_page_config(
    page_title="Fish Species Analysis", 
    layout="wide", 
    page_icon="🐠",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (Theme & Layout) ---
st.markdown("""
    <style>
    /* บังคับพื้นหลังสีฟ้าอ่อน/เทา ตามสไตล์ Minimal */
    .stApp {
        background-color: #F0F4F8;
    }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label {
        color: #262730 !important;
    }
    
    /* สไตล์ของ Card (กล่องขาวๆ) */
    .card-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* สไตล์หัวข้อเมนู */
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
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        width: 100%;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
    }
    
    /* กล่อง About Model สีเข้ม */
    .about-box {
        background-color: #0F172A;
        color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 15px;
        font-size: 14px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
# แมตช์ชื่อไฟล์รูปใน Root (หน้าแรก) ของ GitHub
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
        try:
            with st.spinner('📦 Loading AI Engine...'):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Connection Error: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
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

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ App Controls")
    if st.button("🗑️ Clear History Data", use_container_width=True):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.rerun()

# --- 3. Main Interface Layout ---
st.title("🐠 Fish Species Analysis")

# แบ่งเป็น 2 คอลัมน์ (ซ้าย: Image Source, ขวา: Results)
col_left, col_right = st.columns([1, 1.3], gap="large")

if model is None:
    st.warning("⚠️ AI Model is not ready. Please verify Google Drive permissions.")
else:
    # --- ฝั่งซ้าย: IMAGE SOURCE ---
    with col_left:
        st.markdown('<p class="section-title">IMAGE SOURCE</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Select images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

            if uploaded_files:
                st.subheader(f"📸 Preview ({len(uploaded_files)})")
                with st.container(height=300, border=True):
                    img_cols = st.columns(3)
                    for idx, file in enumerate(uploaded_files):
                        with img_cols[idx % 3]:
                            st.image(Image.open(file), use_container_width=True)

            if st.button('🚀 START ANALYSIS NOW'):
                results = []
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"🔍 Analyzing: {file.name}")
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
                st.success("✅ Analysis Complete!")
                st.balloons()
            st.markdown('</div>', unsafe_allow_html=True)

        # TECHNICAL INFO (ฝั่งซ้ายล่าง)
        st.markdown('<p class="section-title">ⓘ TECHNICAL INFO</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            t1.write("**Architecture**")
            t2.write("MobileNetV2")
            t1.write("**Backend**")
            t2.write("TensorFlow")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- ฝั่งขวา: CLASSIFICATION RESULTS ---
    with col_right:
        st.markdown('<p class="section-title">CLASSIFICATION RESULTS</p>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            if os.path.exists(HISTORY_FILE):
                df = pd.read_csv(HISTORY_FILE)
                
                # Metrics สั้นๆ
                m1, m2 = st.columns(2)
                m1.metric("Total Analyzed", f"{len(df)}")
                m2.metric("Avg Confidence", f"{df['Confidence'].mean():.1f}%")

                # กราฟวงกลม
                fig_pie = px.pie(df, names='Species', hole=0.5, title="Species Distribution")
                fig_pie.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.divider()

                # กราฟกระจาย (Confidence History)
                fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', title="Confidence History")
                fig_scatter.update_layout(height=250)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.write("🔍 Upload an image to see classification results")
            st.markdown('</div>', unsafe_allow_html=True)

        # ABOUT THIS MODEL (ฝั่งขวาล่าง)
        st.markdown('<div class="about-box">', unsafe_allow_html=True)
        st.markdown("**ABOUT THIS MODEL**")
        st.write("This model is optimized for fish species recognition. It can identify 6 different categories with high precision without sending data to a server.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- ล่างสุด: SPECIES OVERVIEW & HISTORY LOGS ---
st.divider()
if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    
    # ส่วนโชว์รูปปลาทั้ง 6 แบบที่อัปไว้ (Example Species)
    st.markdown('<p class="section-title">Example Species</p>', unsafe_allow_html=True)
    grid = st.columns(6)
    for i, species in enumerate(CLASS_NAMES):
        count = len(df[df['Species'] == species])
        with grid[i]:
            img_path = FISH_INFO[species]
            if os.path.exists(img_path):
                # โชว์รูปปลาแต่ละพันธุ์ที่คุณอัปโหลดไว้หน้าแรก
                st.image(img_path, caption=f"{species} ({count})", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ตารางประวัติ (History Logs อยู่ล่างสุดตามที่ขอ)
    st.subheader("📝 Detailed History Logs")
    st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
