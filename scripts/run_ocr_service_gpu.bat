@echo off
setlocal

cd /d "%~dp0.."

if exist "C:\venvs\passport-ocr-gpu\Scripts\activate.bat" (
    call "C:\venvs\passport-ocr-gpu\Scripts\activate.bat"
) else (
    echo OCR virtual environment not found: C:\venvs\passport-ocr-gpu
    exit /b 1
)

set "PY_SITE=C:\venvs\passport-ocr-gpu\Lib\site-packages"
set "PATH=%PY_SITE%\nvidia\cudnn\bin;%PY_SITE%\nvidia\cuda_runtime\bin;%PY_SITE%\nvidia\cuda_nvrtc\bin;%PY_SITE%\nvidia\cublas\bin;%PY_SITE%\nvidia\cufft\bin;%PY_SITE%\nvidia\curand\bin;%PY_SITE%\nvidia\cusolver\bin;%PY_SITE%\nvidia\cusparse\bin;%PY_SITE%\nvidia\nvjitlink\bin;%PATH%"

set "PADDLE_OCR_DEVICE=gpu"
set "INFERENCE_SKIP_OCR_AUTO_ROTATE=true"
set "OCR_SERVICE_HOST=127.0.0.1"
set "OCR_SERVICE_PORT=8124"

python scripts\run_ocr_service.py
