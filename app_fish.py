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

HISTORY_FILE = 'fish_prediction_history.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Nenotetra']

@st.cache_resource
def load_my_model():
    file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(MODEL_PATH):
        try:
            with st.spinner('📦 Downloading AI Model...'):
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"❌ Download Failed: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            # Using compile=False to avoid Unrecognized keyword arguments issues
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"❌ Model Structure Error: {e}")
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

# --- Header ---
st.title("🐠 Fish Species Analysis")
st.write("Upload images to analyze fish species and view the analysis dashboard.")

if model is None:
    st.warning("⚠️ Waiting for AI Model... (Please check Drive file sharing permissions)")
else:
    # --- Upload Section ---
    uploaded_files = st.file_uploader("Choose fish images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader(f"📸 Uploaded Images ({len(uploaded_files)})")
        with st.container(height=300):
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

        if st.button('🚀 Start Analysis', type="primary"):
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
            st.success("✅ Analysis Completed!")
            st.balloons()

    st.divider()

    # --- Dashboard Section ---
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.header("📊 Dashboard")
        m1, m2 = st.columns(2)
        m1.metric("Total Images", f"{len(df)}")
        m2.metric("Avg. Confidence", f"{df['Confidence'].mean():.2f}%")
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_pie = px.pie(df, names='Species', title="Species Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_scatter = px.scatter(df, x='Timestamp', y='Confidence', color='Species', title="Prediction Confidence Over Time")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with st.expander("📝 View History Logs"):
            st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)

        if st.sidebar.button("🗑️ Clear All History"):
            os.remove(HISTORY_FILE)
            st.rerun()
