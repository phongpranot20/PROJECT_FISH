import os
import io
import time
from datetime import datetime
from typing import List

import numpy as np
from PIL import Image
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(
    title="AquaAI - Fish Species Classifier",
    description="High-precision deep learning classifier for fish species identification",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = 'analysis_logs_v2.csv'
MODEL_PATH = 'fish_model_v3.h5'
CLASS_NAMES = ['Angelfish', 'Betta', 'Cichlidae', 'Goldfish', 'Koifish', 'Neontetra']

SPECIES_METADATA = [
    {
        "id": "goldfish",
        "name": "Goldfish",
        "scientific": "Carassius auratus",
        "family": "Cyprinidae",
        "image": "/images/goldfish.jpg",
        "sample_file": "goldfish.jpg",
        "description": "One of the most commonly kept aquarium fish, native to East Asia."
    },
    {
        "id": "betta",
        "name": "Betta Fish",
        "scientific": "Betta splendens",
        "family": "Osphronemidae",
        "image": "/images/betta.jpg",
        "sample_file": "betta.jpg",
        "description": "Known as Siamese fighting fish, renowned for vivid colors and elaborate fins."
    },
    {
        "id": "cichlide",
        "name": "Cichlide",
        "scientific": "Cichlidae family",
        "family": "Cichlidae",
        "image": "/images/cichilde.jpg",
        "sample_file": "cichilde.jpg",
        "description": "A diverse family of tropical freshwater fish with distinct territorial behaviors."
    },
    {
        "id": "koifish",
        "name": "Koi Fish",
        "scientific": "Cyprinus rubrofuscus",
        "family": "Cyprinidae",
        "image": "/images/koifish.jpg",
        "sample_file": "koifish.jpg",
        "description": "Ornamental varieties of carp kept for decorative purposes in outdoor ponds."
    },
    {
        "id": "neontetra",
        "name": "Neon Tetra",
        "scientific": "Paracheirodon innesi",
        "family": "Characidae",
        "image": "/images/neontetra.jpg",
        "sample_file": "neontetra.jpg",
        "description": "Small, iridescent freshwater fish native to blackwater and clearwater streams."
    },
    {
        "id": "angelfish",
        "name": "Angelfish",
        "scientific": "Pterophyllum",
        "family": "Cichlidae",
        "image": "/images/anglefish.jpg",
        "sample_file": "anglefish.jpg",
        "description": "Distinguished by laterally compressed bodies and elongated triangular dorsal fins."
    }
]

# Lazy-loaded model instance
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        import gdown
        file_id = '1mvtOAcFbM2PFxDVv5jtDnqI7-ZCsRhO6'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            print("[INFO] Downloading AI Model...")
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"Failed to download model: {e}")
            return None

    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            print("[INFO] Loading TensorFlow model...")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("[INFO] Model loaded successfully!")
            return _model
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None
    return None

def save_log(result_records: list):
    """Appends prediction records to CSV log file."""
    if not result_records:
        return
    new_df = pd.DataFrame(result_records)
    if not os.path.isfile(HISTORY_FILE):
        new_df.to_csv(HISTORY_FILE, index=False)
    else:
        try:
            old_df = pd.read_csv(HISTORY_FILE)
            pd.concat([old_df, new_df], ignore_index=True).to_csv(HISTORY_FILE, index=False)
        except Exception:
            new_df.to_csv(HISTORY_FILE, index=False)

def predict_single_image(image: Image.Image, filename: str) -> dict:
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model is not loaded or ready.")

    import tensorflow as tf
    img_rgb = image.convert('RGB').resize((180, 180))
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img_rgb), 0)
    
    start_time = time.time()
    preds = model.predict(img_array, verbose=0)[0]
    latency_ms = round((time.time() - start_time) * 1000, 1)

    top_idx = int(np.argmax(preds))
    top_species = CLASS_NAMES[top_idx]
    top_confidence = round(float(np.max(preds) * 100), 2)

    # Detailed scores for all classes
    all_scores = [
        {"species": name, "confidence": round(float(score * 100), 2)}
        for name, score in zip(CLASS_NAMES, preds)
    ]
    all_scores.sort(key=lambda x: x["confidence"], reverse=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "filename": filename,
        "species": top_species,
        "confidence": top_confidence,
        "latency_ms": latency_ms,
        "scores": all_scores
    }

# API Endpoints
@app.get("/api/species")
def get_species_list():
    return {"species": SPECIES_METADATA}

@app.get("/api/model-status")
def get_model_status():
    model = get_model()
    has_model = model is not None
    model_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    return {
        "ready": has_model,
        "model_file": MODEL_PATH,
        "model_size_mb": round(model_size / (1024 * 1024), 2),
        "classes": CLASS_NAMES
    }

@app.post("/api/predict")
async def predict_uploaded_files(files: List[UploadFile] = File(...)):
    results = []
    log_entries = []

    for file in files:
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents))
            res = predict_single_image(pil_img, file.filename)
            results.append(res)
            log_entries.append({
                'Timestamp': res['timestamp'],
                'Filename': res['filename'],
                'Species': res['species'],
                'Confidence': res['confidence']
            })
        except Exception as e:
            results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": file.filename,
                "error": str(e)
            })

    if log_entries:
        save_log(log_entries)

    return {"results": results}

class SamplePredictRequest(BaseModel):
    sample_file: str

@app.post("/api/predict-sample")
def predict_sample_image(req: SamplePredictRequest):
    if not os.path.exists(req.sample_file):
        raise HTTPException(status_code=404, detail="Sample image file not found.")

    pil_img = Image.open(req.sample_file)
    res = predict_single_image(pil_img, req.sample_file)
    save_log([{
        'Timestamp': res['timestamp'],
        'Filename': res['filename'],
        'Species': res['species'],
        'Confidence': res['confidence']
    }])
    return res

@app.get("/api/history")
def get_history_logs():
    if not os.path.exists(HISTORY_FILE):
        return {"logs": []}
    try:
        df = pd.read_csv(HISTORY_FILE)
        if df.empty:
            return {"logs": []}
        df = df.sort_values(by='Timestamp', ascending=False)
        return {"logs": df.to_dict(orient="records")}
    except Exception as e:
        return {"logs": [], "error": str(e)}

@app.delete("/api/history")
def clear_history_logs():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"message": "History cleared successfully."}

@app.get("/api/stats")
def get_statistics():
    if not os.path.exists(HISTORY_FILE):
        return {
            "total_analyzed": 0,
            "avg_confidence": 0,
            "species_counts": {},
            "timeline": []
        }
    try:
        df = pd.read_csv(HISTORY_FILE)
        if df.empty:
            return {
                "total_analyzed": 0,
                "avg_confidence": 0,
                "species_counts": {},
                "timeline": []
            }
        
        species_counts = df['Species'].value_counts().to_dict()
        avg_confidence = round(float(df['Confidence'].mean()), 2)
        total_analyzed = int(len(df))

        # Recent 20 items for timeline
        recent_df = df.tail(20)
        timeline = recent_df[['Timestamp', 'Species', 'Confidence']].to_dict(orient='records')

        return {
            "total_analyzed": total_analyzed,
            "avg_confidence": avg_confidence,
            "species_counts": species_counts,
            "timeline": timeline
        }
    except Exception as e:
        return {"error": str(e)}

# Serve sample images from root folder
@app.get("/images/{image_name}")
def serve_image(image_name: str):
    # Sanitize path to prevent traversal
    safe_name = os.path.basename(image_name)
    file_path = os.path.join(os.path.dirname(__file__), safe_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

# Serve frontend static assets
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
