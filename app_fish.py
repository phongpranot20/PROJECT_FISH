import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import gdown

# --- Page Configuration ---
st.set_page_config(page_title="Fish Species Analysis", layout="wide", page_icon="🐠")

# --- Custom UI Styling (ปุ่มเด่น Gradient) ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white; border: none; padding: 15px 30px;
        font-size: 20px; font-weight: bold; border-radius: 10px;
        width: 100%; transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.01);
        background: linear-gradient(to right, #0072ff, #00c6ff);
    }
    </style>
    """, unsafe_allow_html=True)

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        try:
            with st.spinner('📦 Loading AI Model...'):
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

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear History Data"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.rerun()

# --- Main Page ---
st.title("🐠 Fish Species Analysis")
st.write("Upload images to identify fish species and track results.")

if model is None:
    st.warning("⚠️ AI Model is not ready. Please check Google Drive link.")
else:
    # --- 1. Upload & Preview Section (มีที่เลื่อนแบบเดิม) ---
    uploaded_files = st.file_uploader("Select images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 Image Preview ({len(uploaded_files)})")
        
        # เพิ่มที่เลื่อน (Scrollable Container) ตรงนี้ครับ
        with st.container(height=350, border=True):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

        # ปุ่ม Start Analysis เด่นๆ
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

    # --- 2. Dashboard Section (แจ้งแบบเดิม) ---
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.header("📊 Insight Dashboard")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed", f"{len(df)} Images")
        m2.metric("Avg. Confidence", f"{df['Confidence'].mean():.2f}%")
        m3.metric("Species Found", f"{df['Species'].nunique()} Types")

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
