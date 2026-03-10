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

# --- Custom UI Styling ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white; border: none; padding: 12px 20px;
        font-size: 20px; font-weight: bold; border-radius: 10px;
        width: 100%; transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02); background: linear-gradient(to right, #0072ff, #00c6ff);
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
    
    # ถ้าไม่มีไฟล์ หรือไฟล์เสีย (ขนาดเล็กเกินไป) ให้โหลดใหม่
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        try:
            with st.spinner('📦 Downloading AI Model from Drive (Large File)...'):
                # ใช้ fuzzy=True เพื่อข้ามหน้ายืนยันไวรัสของ Google
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Download Error: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            # ใช้ compile=False เพื่อความชัวร์ว่าโหลดขึ้นแน่นอน
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"❌ Model signature not found. The file might be corrupted.")
            st.info("💡 Try clicking 'Re-download Model' in the sidebar.")
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
    st.header("⚙️ App Controls")
    if st.button("🔄 Re-download Model"):
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()
    if st.button("🗑️ Clear History"):
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.rerun()

# --- Main Page ---
st.title("🐠 Fish Species Analysis")
st.write("Upload fish images for instant species identification.")

if model is None:
    st.warning("⚠️ Waiting for a valid model file. Check sidebar for Re-download option.")
else:
    uploaded_files = st.file_uploader("Select images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 Selected Images ({len(uploaded_files)})")
        with st.container(height=250):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(Image.open(file), use_container_width=True)

        if st.button('🚀 START ANALYSIS'):
            results = []
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name}")
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
    # Dashboard ... (โค้ดส่วน Dashboard เหมือนเดิมครับ)
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.header("📊 Insight Dashboard")
        m1, m2 = st.columns(2)
        m1.metric("Total Analyzed", len(df))
        m2.metric("Avg. Confidence", f"{df['Confidence'].mean():.2f}%")
        st.plotly_chart(px.pie(df, names='Species', hole=0.4), use_container_width=True)
