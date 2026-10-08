@echo off
setlocal

cd /d "%~dp0.."

set "PASSPORT_OCR_SERVICE_URL=http://127.0.0.1:8124/ocr/passport"
set "PASSPORT_OCR_SERVICE_TIMEOUT_SECONDS=180"
set "INFERENCE_SKIP_OCR_AUTO_ROTATE=false"

python scripts\run_api.py
