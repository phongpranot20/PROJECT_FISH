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
st.set_page_config(
    page_title="Fish Species Analysis", 
    layout="wide", 
    page_icon="🐠",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (เน้นแก้ให้รูปเท่ากันและ Card สูงเท่ากัน) ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    
    /* บังคับให้ Column และ Card สูงเท่ากัน */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 0px !important;
        overflow: hidden;
    }

    /* บังคับรูปภาพให้สูงเท่ากันและ Crop ส่วนเกิน (object-fit) */
    [data-testid="stImage"] img {
        height: 200px !important;
        object-fit: cover !important;
        width: 100% !important;
    }

    /* ตกแต่งตัวอักษรใน Card */
    .species-title { 
        font-weight: bold; 
        font-size: 1.1rem; 
        margin-top: 12px; 
        padding: 0 10px;
        color: #262730;
    }
    .species-sub { 
        font-style: italic; 
        color: #888; 
        font-size: 0.85rem; 
        margin-bottom: 15px;
        padding: 0 10px;
    }

    /* ปุ่มวิเคราะห์ */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white !important;
        border: none;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

HISTORY_FILE = 'summary_log_v4.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Neontetra']

# --- 3. ระบบโหลด Model (ใส่ระบบ Auto-download กลับเข้าไปกันเหนียว) ---
@st.cache_resource
def load_my_model():
    # ID ไฟล์ใน Google Drive (จากโค้ดเดิมของคุณ)
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # ถ้าไม่มีไฟล์ หรือไฟล์เสีย ให้โหลดใหม่
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        try:
            with st.spinner('📦 Downloading AI Model... Please wait...'):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Download Error: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except:
            return None
    return None

def save_to_csv(new_df):
    if not os.path.isfile(HISTORY_FILE):
        new_df.to_csv(HISTORY_FILE, index=False)
    else:
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            pd.concat([old_df, new_df], ignore_index=True).to_csv(HISTORY_FILE, index=False)
        except:
            new_df.to_csv(HISTORY_FILE, index=False)

model = load_my_model()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ App Controls")
    if st.button("🗑️ Clear History Data", use_container_width=True):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.rerun()

# --- Main Interface ---
st.title("🐠 Fish Species Analysis")

# --- SECTION: Example Species ---
st.header("Example Species")
st.write("Reference images from your local directory.")

examples = [
    {"name": "Goldfish", "sci": "Carassius auratus", "file": "goldfish.jpg"},
    {"name": "Betta Fish", "sci": "Betta splendens", "file": "betta.jpg"},
    {"name": "Cichlide", "sci": "Cichlidae family", "file": "cichilde.jpg"},
    {"name": "Koi", "sci": "Cyprinus rubrofuscus", "file": "koifish.jpg"},
    {"name": "Neon Tetra", "sci": "Paracheirodon innesi", "file": "neontetra.jpg"},
    {"name": "Angelfish", "sci": "Pterophyllum", "file": "angelfish.jpg"}
]

cols = st.columns(6)
for idx, ex in enumerate(examples):
    with cols[idx]:
        with st.container(border=True):
            if os.path.exists(ex['file']):
                st.image(ex['file'], use_container_width=True)
            else:
                st.warning(f"No {ex['file']}")
            
            st.markdown(f"<div class='species-title'>{ex['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='species-sub'>{ex['sci']}</div>", unsafe_allow_html=True)

st.divider()

if model is None:
    st.error("⚠️ AI Model could not be loaded. Please check your internet connection for the first-time download.")
else:
    # --- Upload Section ---
    uploaded_files = st.file_uploader("Upload fish images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 Image Preview ({len(uploaded_files)})")
        with st.container(height=300, border=True):
            p_cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with p_cols[idx % 6]:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

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

    st.divider()

    # --- Dashboard Section ---
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.header("📊 Insight Dashboard")
        
        m1, m2 = st.columns(2)
        m1.metric("Total Analyzed", f"{len(df)} Images")
        m2.metric("Average Accuracy", f"{df['Confidence'].mean():.2f}%")

        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_pie = px.pie(df, names='Species', title="Species Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', 
                                    hover_data=['Filename'], title="Confidence Levels Over Time")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📝 History Logs")
        st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
