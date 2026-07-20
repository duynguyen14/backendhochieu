from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
import threading

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import (
    get_inference_donut_concurrency,
    get_inference_ocr_concurrency,
    get_log_dir,
    get_passport_inference_api_key,
)
from app.services.ocr_field_matcher import build_ocr_field_matches, serialize_field_matches_for_api
from app.services.passport_portrait_service import detect_passport_portrait
from app.services.passport_inference_service import (
    build_passport_inference_result,
    decode_base64_image_payload,
    get_inference_image_path,
    prepare_passport_inference,
    run_passport_donut_stage,
    run_passport_ocr_stage,
    store_inference_upload,
)


router = APIRouter(tags=["passport-inference"])
_INFERENCE_REQUEST_LOG_LOCK = Lock()
_INFERENCE_OCR_STAGE_LIMIT = threading.Semaphore(get_inference_ocr_concurrency())
_INFERENCE_DONUT_STAGE_LIMIT = threading.Semaphore(get_inference_donut_concurrency())


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


def _get_inference_request_log_file_path(current_time: datetime) -> Path:
    date_folder = get_log_dir() / current_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    return date_folder / "passport_inference_requests.txt"


def _append_inference_request_log(
    *,
    request: Request,
    payload: PassportInferenceUploadPayload,
    status: str,
    detail: str,
    image_id: str = "",
    cache_hit: bool | None = None,
) -> None:
    current_time = datetime.now()
    log_file_path = _get_inference_request_log_file_path(current_time)
    client_ip = request.client.host if request.client else ""
    request_summary = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_ip": client_ip,
        "path": str(request.url.path),
        "method": request.method,
        "status": status,
        "detail": detail,
        "file_name": payload.file_name,
        "base64_length": len(payload.base64 or ""),
        "image_id": image_id,
    }
    if cache_hit is not None:
        request_summary["cache_hit"] = cache_hit

    with _INFERENCE_REQUEST_LOG_LOCK:
        with log_file_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(request_summary, ensure_ascii=False) + "\n")


def _safe_append_inference_request_log(**kwargs: object) -> None:
    try:
        _append_inference_request_log(**kwargs)
    except Exception:
        return


def _run_ocr_stage_limited(image_path: Path) -> dict[str, object]:
    with _INFERENCE_OCR_STAGE_LIMIT:
        return run_passport_ocr_stage(image_path)


def _run_donut_stage_limited(image_path: Path) -> dict[str, object]:
    with _INFERENCE_DONUT_STAGE_LIMIT:
        return run_passport_donut_stage(image_path)


def _build_face_image_payload(image_path: Path, overlay: dict[str, object]) -> dict[str, object]:
    portrait = detect_passport_portrait(image_path, overlay=overlay)
    return _serialize_face_image(portrait)


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
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail="PASSPORT_INFERENCE_API_KEY is not configured",
        )
        raise HTTPException(status_code=500, detail="PASSPORT_INFERENCE_API_KEY is not configured")
    if payload.api_key != configured_api_key:
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail="Invalid API key",
        )
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        total_started = perf_counter()
        file_bytes, resolved_file_name = decode_base64_image_payload(payload.base64, payload.file_name)
        image_id, image_path, cached_result = prepare_passport_inference(file_bytes, resolved_file_name)
        if cached_result is not None:
            result = cached_result
        else:
            ocr_started = perf_counter()
            overlay = await asyncio.to_thread(_run_ocr_stage_limited, image_path)
            ocr_duration_ms = round((perf_counter() - ocr_started) * 1000, 2)

            donut_started = perf_counter()
            donut_result = await asyncio.to_thread(_run_donut_stage_limited, image_path)
            donut_duration_ms = round((perf_counter() - donut_started) * 1000, 2)
            total_duration_ms = round((perf_counter() - total_started) * 1000, 2)

            result = build_passport_inference_result(
                image_id=image_id,
                image_path=image_path,
                file_name=resolved_file_name,
                overlay=overlay,
                donut_result=donut_result,
                ocr_duration_ms=ocr_duration_ms,
                donut_duration_ms=donut_duration_ms,
                total_duration_ms=total_duration_ms,
            )
    except ValueError as exc:
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail=f"Passport inference failed: {exc}",
        )
        raise HTTPException(status_code=500, detail=f"Passport inference failed: {exc}") from exc

    overlay = result["overlay"]
    image_width = float(overlay.get("image_width") or 0)
    image_height = float(overlay.get("image_height") or 0)
    image_url = str(request.url_for("get_passport_inference_image", image_id=result["image_id"]))
    field_matches = build_ocr_field_matches(result.get("editable_fields"), overlay)
    try:
        face_image = await asyncio.to_thread(
            _build_face_image_payload,
            Path(str(result["image_path"])),
            overlay,
        )
    except Exception:
        face_image = _build_empty_face_image()

    _safe_append_inference_request_log(
        request=request,
        payload=payload,
        status="success",
        detail="Passport inference completed",
        image_id=str(result.get("image_id") or ""),
        cache_hit=bool(result.get("performance", {}).get("cache_hit")),
    )

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
