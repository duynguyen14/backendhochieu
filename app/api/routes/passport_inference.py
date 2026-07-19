from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_passport_inference_api_key
from app.services.ocr_field_matcher import build_ocr_field_matches, serialize_field_matches_for_api
from app.services.passport_portrait_service import detect_passport_portrait
from app.services.passport_inference_service import (
    decode_base64_image_payload,
    get_inference_image_path,
    run_passport_inference,
    store_inference_upload,
)


router = APIRouter(tags=["passport-inference"])


class PassportInferenceUploadPayload(BaseModel):
    api_key: str = Field(..., min_length=1)
    base64: str = Field(..., min_length=1)
    file_name: str = Field(default="passport_upload.jpg", min_length=1)


def _guess_content_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".bmp":
        return "image/bmp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _serialize_face_image(portrait: dict[str, object]) -> dict[str, object]:
    portrait_image_path = Path(str(portrait.get("portrait_image_path") or "")).expanduser()
    if not portrait_image_path.exists():
        return {
            "detected": False,
            "file_name": "",
            "content_type": "",
            "base64": "",
        }

    file_bytes = portrait_image_path.read_bytes()
    return {
        "detected": True,
        "file_name": portrait_image_path.name,
        "content_type": _guess_content_type(portrait_image_path),
        "base64": base64.b64encode(file_bytes).decode("ascii"),
    }


def _build_empty_face_image() -> dict[str, object]:
    return {
        "detected": False,
        "file_name": "",
        "content_type": "",
        "base64": "",
    }


def _to_percent(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round((value / total) * 100, 4)


def _serialize_overlay_words(words: list[dict[str, object]], image_width: float, image_height: float) -> list[dict[str, object]]:
    return [
        {
            "id": str(word.get("id") or ""),
            "text": str(word.get("text") or ""),
            "confidence": float(word.get("confidence") or 0),
            "line_id": str(word.get("line_id") or ""),
            "order": int(word.get("order") or 0),
            "rotation": float(word.get("rotation") or 0),
            "boundingBox": {
                "top": _to_percent(float(word["bbox"]["top"]), image_height),
                "left": _to_percent(float(word["bbox"]["left"]), image_width),
                "width": _to_percent(float(word["bbox"]["width"]), image_width),
                "height": _to_percent(float(word["bbox"]["height"]), image_height),
            },
        }
        for word in words
    ]


def _serialize_bbox(bbox: dict[str, object], image_width: float, image_height: float) -> dict[str, dict[str, float] | float]:
    left = float(bbox.get("left") or 0)
    top = float(bbox.get("top") or 0)
    width = float(bbox.get("width") or 0)
    height = float(bbox.get("height") or 0)
    return {
        "pixels": {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        },
        "percent": {
            "left": _to_percent(left, image_width),
            "top": _to_percent(top, image_height),
            "width": _to_percent(width, image_width),
            "height": _to_percent(height, image_height),
        },
    }


@router.get("/passport-inference/images/{image_id}", name="get_passport_inference_image")
def get_passport_inference_image(image_id: str):
    image_path = get_inference_image_path(image_id)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Inference image not found")

    return FileResponse(
        path=image_path,
        filename=image_path.name,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/passport-inference/portraits/{image_id}", name="get_passport_inference_portrait_image")
def get_passport_inference_portrait_image(image_id: str):
    image_path = get_inference_image_path(image_id)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Inference image not found")

    portrait = detect_passport_portrait(image_path)
    portrait_image_path = Path(str(portrait.get("portrait_image_path") or ""))
    if not portrait_image_path.exists():
        raise HTTPException(status_code=404, detail="Portrait crop not found")

    return FileResponse(
        path=portrait_image_path,
        filename=portrait_image_path.name,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.post("/passport-portrait/upload")
async def upload_passport_portrait_only(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    try:
        file_bytes = await file.read()
        image_id, image_path = store_inference_upload(file_bytes, file.filename)
        portrait = detect_passport_portrait(image_path, use_ocr_fallback=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Passport portrait detection failed: {exc}") from exc
    finally:
        await file.close()

    image_width = float(portrait.get("image_width") or 0)
    image_height = float(portrait.get("image_height") or 0)

    return {
        "status": "success",
        "data": {
            "image_id": image_id,
            "image_name": Path(file.filename or image_path.name).name,
            "image_path": str(image_path),
            "image_url": str(request.url_for("get_passport_inference_image", image_id=image_id)),
            "detected": bool(portrait.get("detected")),
            "face_bbox": _serialize_bbox(portrait.get("face_bbox", {}), image_width, image_height),
            "portrait_bbox": _serialize_bbox(portrait.get("portrait_bbox", {}), image_width, image_height),
            "portrait_image_path": str(portrait.get("portrait_image_path") or ""),
            "portrait_image_url": (
                str(request.url_for("get_passport_inference_portrait_image", image_id=image_id))
                if str(portrait.get("portrait_image_path") or "")
                else ""
            ),
            "image_width": image_width,
            "image_height": image_height,
        },
    }


@router.post("/passport-inference/upload")
async def upload_passport_inference(request: Request, payload: PassportInferenceUploadPayload):
    configured_api_key = get_passport_inference_api_key()
    if not configured_api_key:
        raise HTTPException(status_code=500, detail="PASSPORT_INFERENCE_API_KEY is not configured")
    if payload.api_key != configured_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        file_bytes, resolved_file_name = decode_base64_image_payload(payload.base64, payload.file_name)
        result = run_passport_inference(file_bytes, resolved_file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Passport inference failed: {exc}") from exc

    overlay = result["overlay"]
    image_width = float(overlay.get("image_width") or 0)
    image_height = float(overlay.get("image_height") or 0)
    image_url = str(request.url_for("get_passport_inference_image", image_id=result["image_id"]))
    field_matches = build_ocr_field_matches(result.get("editable_fields"), overlay)
    try:
        portrait = detect_passport_portrait(Path(str(result["image_path"])), overlay=overlay)
        face_image = _serialize_face_image(portrait)
    except Exception:
        face_image = _build_empty_face_image()

    return {
        "status": "success",
        "data": {
            "image_id": result["image_id"],
            "image_name": result["image_name"],
            "image_url": image_url,
            "editable_fields": result["editable_fields"],
            "donut_raw_text": result["donut_raw_text"],
            "donut_json": result["donut_json"],
            "task_prompt": result["task_prompt"],
            "performance": result["performance"],
            "face_image": face_image,
            "overlay": {
                "image_path": result["image_path"],
                "image_url": image_url,
                "image_width": image_width,
                "image_height": image_height,
                "rotation_applied": float(overlay.get("rotation_applied") or 0),
                "words": _serialize_overlay_words(overlay.get("words", []), image_width, image_height),
                "field_matches": serialize_field_matches_for_api(field_matches, image_width, image_height),
            },
        },
    }
