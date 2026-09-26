@echo off
echo Starting AquaAI Fish Classifier Web Server...
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
pause
