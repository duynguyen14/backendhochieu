from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
import threading
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import (
    get_inference_donut_concurrency,
    get_inference_ocr_concurrency,
    get_log_dir,
    get_passport_inference_api_key,
)
from app.services.ocr_field_matcher import (
    apply_high_confidence_ocr_date_overrides,
    build_ocr_field_matches,
    serialize_field_matches_for_api,
)
from app.services.passport_face_match_service import get_passport_face_match_runtime_info, verify_passport_face_match
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


class PassportFaceVerifyPayload(BaseModel):
    api_key: str = Field(..., min_length=1)
    passport_face_base64: str = Field(..., min_length=1)
    passport_face_file_name: str = Field(default="passport_face.jpg", min_length=1)
    uploaded_face_base64: str = Field(..., min_length=1)
    uploaded_face_file_name: str = Field(default="uploaded_face.jpg", min_length=1)


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


def _get_inference_stage_log_file_path(current_time: datetime) -> Path:
    date_folder = get_log_dir() / current_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    return date_folder / "passport_inference_stage_trace.txt"


def _append_inference_stage_log(
    *,
    request_id: str,
    stage: str,
    detail: str = "",
    request: Request | None = None,
    image_id: str = "",
    file_name: str = "",
    elapsed_ms: float | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    current_time = datetime.now()
    client_ip = request.client.host if request and request.client else ""
    log_item: dict[str, object] = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "request_id": request_id,
        "client_ip": client_ip,
        "path": str(request.url.path) if request else "",
        "method": request.method if request else "",
        "stage": stage,
        "detail": detail,
        "image_id": image_id,
        "file_name": file_name,
        "thread": threading.current_thread().name,
    }
    if elapsed_ms is not None:
        log_item["elapsed_ms"] = round(float(elapsed_ms), 2)
    if extra:
        log_item["extra"] = extra

    console_parts = [
        f"[passport-upload][{log_item['timestamp']}][{request_id}]",
        str(stage),
    ]
    if detail:
        console_parts.append(str(detail))
    if image_id:
        console_parts.append(f"image_id={image_id}")
    if elapsed_ms is not None:
        console_parts.append(f"elapsed_ms={round(float(elapsed_ms), 2)}")
    print(" ".join(console_parts), flush=True)

    log_file_path = _get_inference_stage_log_file_path(current_time)
    with _INFERENCE_REQUEST_LOG_LOCK:
        with log_file_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def _safe_append_inference_stage_log(**kwargs: object) -> None:
    try:
        _append_inference_stage_log(**kwargs)
    except Exception:
        return


def _append_inference_request_log(
    *,
    request: Request,
    payload: BaseModel,
    status: str,
    detail: str,
    image_id: str = "",
    cache_hit: bool | None = None,
    response_data: dict[str, object] | None = None,
) -> None:
    current_time = datetime.now()
    log_file_path = _get_inference_request_log_file_path(current_time)
    client_ip = request.client.host if request.client else ""
    payload_data = payload.model_dump() if isinstance(payload, BaseModel) else {}
    file_name = str(
        payload_data.get("file_name")
        or payload_data.get("passport_face_file_name")
        or payload_data.get("uploaded_face_file_name")
        or ""
    )
    raw_base64_value = str(
        payload_data.get("base64")
        or payload_data.get("passport_face_base64")
        or payload_data.get("uploaded_face_base64")
        or ""
    )
    request_summary = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_ip": client_ip,
        "path": str(request.url.path),
        "method": request.method,
        "status": status,
        "detail": detail,
        "file_name": file_name,
        "base64_length": len(raw_base64_value),
        "image_id": image_id,
    }
    if cache_hit is not None:
        request_summary["cache_hit"] = cache_hit
    if response_data is not None:
        request_summary["response_data"] = response_data

    with _INFERENCE_REQUEST_LOG_LOCK:
        with log_file_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(request_summary, ensure_ascii=False) + "\n")


def _safe_append_inference_request_log(**kwargs: object) -> None:
    try:
        _append_inference_request_log(**kwargs)
    except Exception:
        return


def _run_ocr_stage_limited(image_path: Path, *, request_id: str, image_id: str, file_name: str) -> dict[str, object]:
    wait_started = perf_counter()
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="ocr_wait_semaphore",
        image_id=image_id,
        file_name=file_name,
    )
    _INFERENCE_OCR_STAGE_LIMIT.acquire()
    wait_duration_ms = (perf_counter() - wait_started) * 1000
    try:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="ocr_start",
            image_id=image_id,
            file_name=file_name,
            elapsed_ms=wait_duration_ms,
        )
        run_started = perf_counter()
        result = run_passport_ocr_stage(image_path)
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="ocr_done",
            image_id=image_id,
            file_name=file_name,
            elapsed_ms=(perf_counter() - run_started) * 1000,
            extra={
                "word_count": len(result.get("words", [])),
                "line_count": len(result.get("lines", [])),
            },
        )
        return result
    except Exception as exc:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="ocr_error",
            detail=str(exc),
            image_id=image_id,
            file_name=file_name,
        )
        raise
    finally:
        _INFERENCE_OCR_STAGE_LIMIT.release()
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="ocr_release_semaphore",
            image_id=image_id,
            file_name=file_name,
        )


def _run_donut_stage_limited(image_path: Path, *, request_id: str, image_id: str, file_name: str) -> dict[str, object]:
    wait_started = perf_counter()
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="donut_wait_semaphore",
        image_id=image_id,
        file_name=file_name,
    )
    _INFERENCE_DONUT_STAGE_LIMIT.acquire()
    wait_duration_ms = (perf_counter() - wait_started) * 1000
    try:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="donut_start",
            image_id=image_id,
            file_name=file_name,
            elapsed_ms=wait_duration_ms,
        )
        run_started = perf_counter()
        result = run_passport_donut_stage(image_path)
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="donut_done",
            image_id=image_id,
            file_name=file_name,
            elapsed_ms=(perf_counter() - run_started) * 1000,
            extra={
                "editable_field_count": len(result.get("editable_fields", {})),
            },
        )
        return result
    except Exception as exc:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="donut_error",
            detail=str(exc),
            image_id=image_id,
            file_name=file_name,
        )
        raise
    finally:
        _INFERENCE_DONUT_STAGE_LIMIT.release()
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="donut_release_semaphore",
            image_id=image_id,
            file_name=file_name,
        )


def _build_face_image_payload(image_path: Path, overlay: dict[str, object]) -> dict[str, object]:
    portrait = detect_passport_portrait(image_path, overlay=overlay)
    return _serialize_face_image(portrait)


def _build_source_image_payload(image_path: Path) -> dict[str, object]:
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.exists():
        return {
            "file_name": "",
            "content_type": "",
            "base64": "",
        }

    file_bytes = resolved_path.read_bytes()
    return {
        "file_name": resolved_path.name,
        "content_type": _guess_content_type(resolved_path),
        "base64": base64.b64encode(file_bytes).decode("ascii"),
    }


def _build_loggable_response_data(data: dict[str, object]) -> dict[str, object]:
    loggable_data = dict(data)

    source_image_base64 = str(loggable_data.pop("image_base64", "") or "")
    if source_image_base64:
        loggable_data["image_base64_length"] = len(source_image_base64)

    raw_face_image = loggable_data.get("face_image")
    if isinstance(raw_face_image, dict):
        face_image = dict(raw_face_image)
        base64_value = str(face_image.pop("base64", "") or "")
        face_image["base64_length"] = len(base64_value)
        loggable_data["face_image"] = face_image

    raw_overlay = loggable_data.get("overlay")
    if isinstance(raw_overlay, dict):
        overlay = dict(raw_overlay)
        overlay_base64 = str(overlay.pop("image_base64", "") or "")
        if overlay_base64:
            overlay["image_base64_length"] = len(overlay_base64)
        loggable_data["overlay"] = overlay

    return loggable_data


def _build_loggable_face_match_response(data: dict[str, object]) -> dict[str, object]:
    loggable_data = dict(data)

    for field_name in ("passport_face", "uploaded_face"):
        raw_face = loggable_data.get(field_name)
        if not isinstance(raw_face, dict):
            continue

        face_data = dict(raw_face)
        for base64_field_name in ("base64", "aligned_face_base64"):
            base64_value = str(face_data.pop(base64_field_name, "") or "")
            if base64_value:
                face_data[f"{base64_field_name}_length"] = len(base64_value)
        loggable_data[field_name] = face_data

    return loggable_data


def _validate_inference_api_key(*, configured_api_key: str, provided_api_key: str, request: Request, payload: BaseModel) -> None:
    if not configured_api_key:
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail="PASSPORT_INFERENCE_API_KEY is not configured",
        )
        raise HTTPException(status_code=500, detail="PASSPORT_INFERENCE_API_KEY is not configured")
    if provided_api_key != configured_api_key:
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail="Invalid API key",
        )
        raise HTTPException(status_code=401, detail="Invalid API key")


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


@router.post("/passport-interface/upload")
@router.post("/passport-inference/upload")
async def upload_passport_inference(request: Request, payload: PassportInferenceUploadPayload):
    request_id = str(getattr(request.state, "passport_upload_request_id", "") or uuid4().hex[:12])
    request_started = perf_counter()
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="request_received",
        request=request,
        file_name=payload.file_name,
        extra={
            "base64_length": len(str(payload.base64 or "")),
        },
    )
    configured_api_key = get_passport_inference_api_key()
    _validate_inference_api_key(
        configured_api_key=configured_api_key,
        provided_api_key=payload.api_key,
        request=request,
        payload=payload,
    )
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="api_key_valid",
        request=request,
        file_name=payload.file_name,
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )

    try:
        total_started = perf_counter()
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="decode_base64_start",
            request=request,
            file_name=payload.file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        file_bytes, resolved_file_name = decode_base64_image_payload(payload.base64, payload.file_name)
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="decode_base64_done",
            request=request,
            file_name=resolved_file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
            extra={
                "file_size_bytes": len(file_bytes),
            },
        )
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="prepare_inference_start",
            request=request,
            file_name=resolved_file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        image_id, image_path, cached_result = prepare_passport_inference(file_bytes, resolved_file_name)
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="prepare_inference_done",
            request=request,
            image_id=image_id,
            file_name=resolved_file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
            extra={
                "cache_hit": cached_result is not None,
                "image_path": str(image_path),
            },
        )
        if cached_result is not None:
            _safe_append_inference_stage_log(
                request_id=request_id,
                stage="cache_hit",
                request=request,
                image_id=image_id,
                file_name=resolved_file_name,
                elapsed_ms=(perf_counter() - request_started) * 1000,
            )
            result = cached_result
        else:
            ocr_started = perf_counter()
            overlay = await asyncio.to_thread(
                _run_ocr_stage_limited,
                image_path,
                request_id=request_id,
                image_id=image_id,
                file_name=resolved_file_name,
            )
            ocr_duration_ms = round((perf_counter() - ocr_started) * 1000, 2)

            donut_started = perf_counter()
            donut_result = await asyncio.to_thread(
                _run_donut_stage_limited,
                image_path,
                request_id=request_id,
                image_id=image_id,
                file_name=resolved_file_name,
            )
            donut_duration_ms = round((perf_counter() - donut_started) * 1000, 2)
            total_duration_ms = round((perf_counter() - total_started) * 1000, 2)

            _safe_append_inference_stage_log(
                request_id=request_id,
                stage="build_result_start",
                request=request,
                image_id=image_id,
                file_name=resolved_file_name,
                elapsed_ms=(perf_counter() - request_started) * 1000,
            )
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
            _safe_append_inference_stage_log(
                request_id=request_id,
                stage="build_result_done",
                request=request,
                image_id=image_id,
                file_name=resolved_file_name,
                elapsed_ms=(perf_counter() - request_started) * 1000,
            )
    except ValueError as exc:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="request_error",
            detail=str(exc),
            request=request,
            file_name=payload.file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="request_error",
            detail=str(exc),
            request=request,
            file_name=payload.file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        _safe_append_inference_request_log(
            request=request,
            payload=payload,
            status="error",
            detail=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="request_error",
            detail=f"Passport inference failed: {exc}",
            request=request,
            file_name=payload.file_name,
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
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
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="source_image_payload_start",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )
    source_image = _build_source_image_payload(Path(str(result["image_path"])))
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="source_image_payload_done",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
        extra={
            "image_base64_length": len(str(source_image.get("base64") or "")),
        },
    )
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="field_match_start",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )
    editable_fields = apply_high_confidence_ocr_date_overrides(result.get("editable_fields"), overlay)
    field_matches = build_ocr_field_matches(editable_fields, overlay)
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="field_match_done",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )
    try:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="face_crop_start",
            request=request,
            image_id=str(result["image_id"]),
            file_name=str(result["image_name"]),
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        face_image = await asyncio.to_thread(
            _build_face_image_payload,
            Path(str(result["image_path"])),
            overlay,
        )
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="face_crop_done",
            request=request,
            image_id=str(result["image_id"]),
            file_name=str(result["image_name"]),
            elapsed_ms=(perf_counter() - request_started) * 1000,
            extra={
                "detected": bool(face_image.get("detected")),
                "face_base64_length": len(str(face_image.get("base64") or "")),
            },
        )
    except Exception as exc:
        _safe_append_inference_stage_log(
            request_id=request_id,
            stage="face_crop_error",
            detail=str(exc),
            request=request,
            image_id=str(result["image_id"]),
            file_name=str(result["image_name"]),
            elapsed_ms=(perf_counter() - request_started) * 1000,
        )
        face_image = _build_empty_face_image()

    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="response_build_start",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )
    response_data = {
        "image_id": result["image_id"],
        "image_name": result["image_name"],
        "image_url": image_url,
        "image_content_type": source_image["content_type"],
        "image_base64": source_image["base64"],
        "editable_fields": editable_fields,
        "donut_raw_text": result["donut_raw_text"],
        "donut_json": result["donut_json"],
        "task_prompt": result["task_prompt"],
        "performance": result["performance"],
        "face_image": face_image,
        "overlay": {
            "image_path": result["image_path"],
            "image_url": image_url,
            "image_content_type": source_image["content_type"],
            "image_base64": source_image["base64"],
            "image_width": image_width,
            "image_height": image_height,
            "rotation_applied": float(overlay.get("rotation_applied") or 0),
            "words": _serialize_overlay_words(overlay.get("words", []), image_width, image_height),
            "field_matches": serialize_field_matches_for_api(field_matches, image_width, image_height),
        },
    }
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="response_build_done",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
    )

    _safe_append_inference_request_log(
        request=request,
        payload=payload,
        status="success",
        detail="Passport inference completed",
        image_id=str(result.get("image_id") or ""),
        cache_hit=bool(result.get("performance", {}).get("cache_hit")),
        response_data=_build_loggable_response_data(response_data),
    )
    _safe_append_inference_stage_log(
        request_id=request_id,
        stage="request_completed",
        request=request,
        image_id=str(result["image_id"]),
        file_name=str(result["image_name"]),
        elapsed_ms=(perf_counter() - request_started) * 1000,
        extra={
            "cache_hit": bool(result.get("performance", {}).get("cache_hit")),
        },
    )

    return {
        "status": "success",
        "data": response_data,
    }


@router.post("/passport-face-match/verify")
async def verify_passport_face(request: Request, payload: PassportFaceVerifyPayload):
    configured_api_key = get_passport_inference_api_key()
    _validate_inference_api_key(
        configured_api_key=configured_api_key,
        provided_api_key=payload.api_key,
        request=request,
        payload=payload,
    )

    try:
        result = await asyncio.to_thread(
            verify_passport_face_match,
            passport_face_base64=payload.passport_face_base64,
            passport_face_file_name=payload.passport_face_file_name,
            uploaded_face_base64=payload.uploaded_face_base64,
            uploaded_face_file_name=payload.uploaded_face_file_name,
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
            detail=f"Passport face match failed: {exc}",
        )
        raise HTTPException(status_code=500, detail=f"Passport face match failed: {exc}") from exc

    _safe_append_inference_request_log(
        request=request,
        payload=payload,
        status="success",
        detail="Passport face match completed",
        response_data=_build_loggable_face_match_response(result),
    )

    return {
        "status": "success",
        "data": result,
    }


@router.get("/passport-face-match/runtime")
def get_passport_face_match_runtime():
    return {
        "status": "success",
        "data": get_passport_face_match_runtime_info(),
    }
