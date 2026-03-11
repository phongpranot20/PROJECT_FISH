import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px  # สำหรับสร้างกราฟ
from datetime import datetime
import gdown

# --- 1. Page Config ---
st.set_page_config(
    page_title="Fish Species Analysis", 
    layout="wide", 
    page_icon="🐠",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (พื้นหลังฟ้าอ่อน + กล่องขาวมนมีเงา) ---
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF !important; }
    [data-testid="column"] { display: flex; flex-direction: column; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1; display: flex; flex-direction: column;
        background-color: white !important; 
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
        padding: 0px !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stImage"] img {
        height: 180px !important; 
        width: 100% !important;
        object-fit: cover !important;
        border-radius: 20px 20px 0 0 !important;
    }
    .species-title { font-weight: bold; font-size: 1.1rem; margin-top: 15px; padding: 0 15px; color: #1E1E1E !important; }
    .species-sub { font-style: italic; color: #8E8E93 !important; font-size: 0.85rem; padding: 0 15px 20px 15px; }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important; padding: 15px !important; font-size: 18px !important;
        font-weight: bold !important; border-radius: 12px !important; width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Path & Model Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, 'summary_log_v4.csv') 
MODEL_PATH = os.path.join(BASE_DIR, 'fish_model_v3.h5')
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Neontetra']

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
        url = f'https://drive.google.com/uc?id={file_id}'
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

# --- Example Species ---
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
        with st.container(border=True):
            full_img_path = os.path.join(BASE_DIR, ex['file'])
            if os.path.exists(full_img_path):
                st.image(full_img_path, use_container_width=True)
            else:
                st.write(f"🚫 {ex['file']} missing")
            st.markdown(f"<div class='species-title'>{ex['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='species-sub'>{ex['sci']}</div>", unsafe_allow_html=True)

st.divider()

# --- 6. Upload Section ---
uploaded_files = st.file_uploader("Upload fish images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
if uploaded_files:
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

st.divider()

# --- 7. Dashboard (เอากราฟวงกลมและกราฟจุดกลับมา) ---
st.header("📊 Insight Dashboard")
if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    if not df.empty:
        # Metrics Row
        m1, m2 = st.columns(2)
        m1.metric("Total Analyzed", f"{len(df)} Images")
        if 'Confidence' in df.columns:
            m2.metric("Average Accuracy", f"{df['Confidence'].mean():.2f}%")

        # กราฟ Row
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            if 'Species' in df.columns:
                # กราฟวงกลม (Pie Chart)
                fig_pie = px.pie(df, names='Species', title="Species Distribution", 
                                hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            if 'Timestamp' in df.columns and 'Confidence' in df.columns:
                # กราฟจุด (Scatter Plot)
                fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species',
                                        title="Confidence Levels over Time",
                                        hover_data=['Filename'])
                st.plotly_chart(fig_scatter, use_container_width=True)

        # History Logs Table
        st.subheader("📝 History Logs")
        display_cols = [c for c in ['Timestamp', 'Filename', 'Species', 'Confidence'] if c in df.columns]
        st.dataframe(df[display_cols].sort_values(by='Timestamp', ascending=False), use_container_width=True)
