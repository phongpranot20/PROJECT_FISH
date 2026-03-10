import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
from datetime import datetime
import gdown

# --- 1. Page Config ---
st.set_page_config(
    page_title="Fish Species Analysis", 
    layout="wide", 
    page_icon="🐠",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (ปรับสีพื้นหลังแอป และกล่อง Card สีขาวตามรูปที่ 2) ---
st.markdown("""
    <style>
    /* ✅ 1. สีพื้นหลังของแอป (สีฟ้าอ่อน) */
    .stApp { 
        background-color: #F0F8FF; 
    }
    
    /* ✅ 2. ปรับแต่งกล่อง Card ให้เป็นสีขาวและขอบมน (ตามรูปที่ 2) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        background-color: white !important; /* พื้นหลังกล่องสีขาว */
        border-radius: 20px !important;    /* ขอบมนมากขึ้นตามรูป */
        padding: 0px !important;
        border: none !important;           /* เอาเส้นขอบออก */
        box-shadow: 0 10px 25px rgba(0,0,0,0.05) !important; /* เพิ่มเงาให้นุ่มนวล */
        overflow: hidden;
        transition: transform 0.3s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px); /* เวลาเอาเมาส์ชี้ให้กล่องลอยขึ้นนิดนึง */
    }

    /* บังคับรูปภาพให้สูงเท่ากันเป๊ะ */
    [data-testid="stImage"] img {
        height: 180px !important; 
        width: 100% !important;
        object-fit: cover !important; 
    }

    /* จัดแต่งตัวอักษรด้านล่างรูป */
    .species-title { 
        font-weight: bold; 
        font-size: 1.1rem; 
        margin-top: 15px; 
        padding: 0 15px; 
        color: #1E1E1E;
    }
    .species-sub { 
        font-style: italic; 
        color: #8E8E93; 
        font-size: 0.85rem; 
        padding: 0 15px 20px 15px;
    }

    /* ปรับแต่งส่วนหัวข้อหลัก */
    .stApp h1, .stApp h2 {
        color: #262730 !important;
        font-weight: 800 !important;
    }

    /* ปุ่ม Start Analysis */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white !important;
        border: none; 
        padding: 15px; 
        font-size: 20px; 
        font-weight: bold;
        border-radius: 15px; 
        width: 100%; 
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ตั้งค่าไฟล์และโมเดล ---
HISTORY_FILE = 'summary_log_v4.csv' 
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Neontetra']

@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        try:
            with st.spinner('📦 Downloading AI Model...'):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except: return None
    if os.path.exists(MODEL_PATH):
        try: return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except: return None
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

# --- 4. Sidebar ---
with st.sidebar:
    st.header("⚙️ App Controls")
    if st.button("🗑️ Clear History Data", use_container_width=True):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.rerun()

# --- 5. Main Interface ---
st.title("🐠 Fish Species Analysis")

# --- SECTION: Example Species ---
st.header("Example Species")

examples = [
    {"name": "Goldfish", "sci": "Carassius auratus", "file": "goldfish.jpg"},
    {"name": "Betta Fish", "sci": "Betta splendens", "file": "betta.jpg"},
    {"name": "Cichlide", "sci": "Cichlidae family", "file": "cichilde.jpg"},
    {"name": "Koi", "sci": "Cyprinus rubrofuscus", "file": "koifish.jpg"},
    {"name": "Neon Tetra", "sci": "Paracheirodon innesi", "file": "neontetra.jpg"},
    {"name": "Angelfish", "sci": "Pterophyllum", "file": "anglefish.jpg"} 
]

cols = st.columns(6)
for idx, ex in enumerate(examples):
    with cols[idx]:
        # ใช้ container ครอบเพื่อให้เกิด Card สีขาวที่ตั้งค่าไว้ใน CSS
        with st.container(border=True):
            if os.path.exists(ex['file']):
                st.image(ex['file'], use_container_width=True)
            else:
                st.write(f"🚫 {ex['file']} missing")
            
            st.markdown(f"<div class='species-title'>{ex['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='species-sub'>{ex['sci']}</div>", unsafe_allow_html=True)

st.divider()

# --- 6. Upload & Analysis ---
if model is None:
    st.error("⚠️ AI Model is not ready.")
else:
    uploaded_files = st.file_uploader("Upload fish images for analysis...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 Image Preview")
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
                    'Confidence': float(np.max(pred[0]) * 100)
                })
                progress_bar.progress((i + 1) / len(uploaded_files))

            save_to_csv(pd.DataFrame(results))
            status_text.empty()
            progress_bar.empty()
            st.success("✅ Analysis Complete!")
            st.rerun()

    # --- 7. History Logs ---
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        if not df.empty:
            st.header("📝 History Logs")
            display_cols = [c for c in ['Timestamp', 'Filename', 'Species', 'Confidence'] if c in df.columns]
            st.dataframe(
                df[display_cols].sort_values(by='Timestamp', ascending=False), 
                use_container_width=True
            )
