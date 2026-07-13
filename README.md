# Backend Layout

This backend is organized so the reusable code lives under `app/` and runnable commands live under `scripts/`.

## Structure

- `app/config/`: environment loading and application settings
- `app/database/`: SQL Server connection helpers
- `app/models/`: lightweight domain models
- `app/repositories/`: database access helpers
- `app/services/`: OCR and image-processing logic
- `app/scripts/`: command implementations
- `scripts/`: thin entrypoints for running commands directly

## Common Commands

```powershell
python .\Backend\main.py
python .\Backend\scripts\start_backend_app.py
python .\Backend\scripts\check_db.py
python .\Backend\scripts\ocr_to_db.py
python .\Backend\scripts\rename_images.py
python .\Backend\scripts\mask_mlz_images.py
python .\Backend\scripts\crop_mlz_images.py
python .\Backend\scripts\generate_layoutlm_json.py
python .\Backend\scripts\import_images_flat.py
python .\Backend\scripts\run_api.py
```

Use `python .\Backend\main.py` when you just want to start the backend API with the host and port from `.env`.

## OCR Runtime

This project now uses `PaddleOCR` instead of Tesseract.

For CPU setup on Windows, install the inference runtime first, then install backend requirements:

```powershell
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r .\Backend\requirements.txt
```

OCR behavior is configured through `.env`:

- `OCR_LANGUAGE`
- `PADDLE_OCR_VERSION`
- `PADDLE_OCR_DEVICE`
- `PADDLE_PDX_MODEL_SOURCE`
- `PADDLE_TEXT_DETECTION_MODEL_DIR`
- `PADDLE_TEXT_RECOGNITION_MODEL_DIR`
- `OCR_IMAGE_INPUT_DIR`
- `RENAME_IMAGE_INPUT_DIR`
- `IMPORT_SOURCE_IMAGE_INPUT_DIR`
- `IMPORT_TARGET_IMAGE_OUTPUT_DIR`
- `MLZ_MASK_INPUT_DIR`
- `MLZ_MASK_OUTPUT_DIR`
- `MLZ_CROP_INPUT_DIR`
- `MLZ_CROP_OUTPUT_DIR`
- `MASK_REVIEW_IMAGE_DIR`
- `MASK_REVIEW_ERROR_DIR`
- `MASK_REVIEW_STATE_PATH`

## MLZ/MRZ Prep For Phase 4

Use these batch scripts to prepare a second image set without the passport MRZ strip:

```powershell
python .\Backend\scripts\mask_mlz_images.py
python .\Backend\scripts\crop_mlz_images.py
```

Both commands read their input/output folders from `.env`, including the source `metadata.jsonl`. Output images keep the old name and append `_mask` or `_crop`, and each output folder gets its own `metadata.jsonl` with `personal_number=""`. You can also override them with:

```powershell
python .\Backend\scripts\mask_mlz_images.py --input D:\input --output D:\masked --metadata D:\input\metadata.jsonl --recursive
python .\Backend\scripts\crop_mlz_images.py --input D:\input --output D:\cropped --metadata D:\input\metadata.jsonl --recursive
```

Frontend now also supports a dedicated mask-review flow:

- load next unreviewed image from `MASK_REVIEW_IMAGE_DIR`
- `approved`: keep image and mark as reviewed in `review_state.json`
- `rejected`: move image to `MASK_REVIEW_ERROR_DIR` and remove its line from `metadata.jsonl`

Review progress resumes automatically from `MASK_REVIEW_STATE_PATH`.

Backend runtime is also configured through `.env`:

- `API_HOST`
- `API_PORT`
- `APP_LOG_DIR`

## LayoutLMV3 Prep

Before using the LayoutLM review flow, run the SQL patch once:

```powershell
sqlcmd -S .\SQLEXPRESS -d HOCHIEU -E -i .\Backend\sql\add_layoutlm_columns.sql
```

Generate the first-pass LayoutLM JSON from OCR + reviewed passport fields:

```powershell
python .\Backend\scripts\generate_layoutlm_json.py
python .\Backend\scripts\generate_layoutlm_json.py --record-id 1
```
