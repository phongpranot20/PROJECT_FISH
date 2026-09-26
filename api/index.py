import os
import io
import time
from datetime import datetime
from typing import List

import numpy as np
from PIL import Image
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "fish_model_v3.tflite")
SAMPLES_DIR = os.path.join(CURRENT_DIR, "samples")
HISTORY_FILE = "/tmp/analysis_logs_v2.csv"

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

app = FastAPI(
    title="AquaAI - Fish Species Classifier",
    description="High-precision deep learning classifier for fish species identification (Vercel Serverless Ready)",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_interpreter = None

def get_interpreter():
    global _interpreter
    if _interpreter is not None:
        return _interpreter

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading TFLite model from {MODEL_PATH}...")
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        print("[INFO] Model loaded successfully!")
        return _interpreter

    return None

def save_log(result_records: list):
    if not result_records:
        return
    new_df = pd.DataFrame(result_records)
    try:
        if not os.path.isfile(HISTORY_FILE):
            new_df.to_csv(HISTORY_FILE, index=False)
        else:
            old_df = pd.read_csv(HISTORY_FILE)
            pd.concat([old_df, new_df], ignore_index=True).to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        print(f"[WARN] Unable to write log file: {e}")

def predict_single_image(image: Image.Image, filename: str) -> dict:
    interpreter = get_interpreter()
    if interpreter is None:
        raise HTTPException(status_code=503, detail="AI Model is not ready.")

    img_rgb = image.convert('RGB').resize((180, 180))
    input_data = np.expand_dims(np.array(img_rgb, dtype=np.float32), axis=0)

    start_time = time.time()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    latency_ms = round((time.time() - start_time) * 1000, 1)

    top_idx = int(np.argmax(preds))
    top_species = CLASS_NAMES[top_idx]
    top_confidence = round(float(np.max(preds) * 100), 2)

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

@app.get("/")
@app.get("/api")
def root_check():
    return {
        "status": "AquaAI API Online",
        "version": "2.2.0",
        "model": "TensorFlow Lite (XNNPACK)"
    }

@app.get("/api/species")
def get_species_list():
    return {"species": SPECIES_METADATA}

@app.get("/api/model-status")
def get_model_status():
    interp = get_interpreter()
    has_model = interp is not None
    model_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    return {
        "ready": has_model,
        "model_file": os.path.basename(MODEL_PATH),
        "model_size_mb": round(model_size / (1024 * 1024), 2),
        "format": "TensorFlow Lite",
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
    fname = os.path.basename(req.sample_file)
    sample_path = os.path.join(SAMPLES_DIR, fname)
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(CURRENT_DIR), fname)

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample image {fname} not found.")

    pil_img = Image.open(sample_path)
    res = predict_single_image(pil_img, fname)
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
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass
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
