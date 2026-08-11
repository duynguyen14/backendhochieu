from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.code_values import router as code_values_router
from app.api.routes.document_type_classifier import router as document_type_classifier_router
from app.api.routes.mask_review import router as mask_review_router
from app.api.routes.passport_inference import router as passport_inference_router
from app.api.routes.passport_records import router as passport_records_router
from app.config import get_frontend_allowed_origins, get_log_dir
from app.services.document_type_classifier_service import preload_document_type_classifier_runtime
from app.services.passport_face_match_service import preload_passport_face_match_runtime
from app.services.ocr_service import preload_ocr_runtime
from app.services.passport_inference_service import preload_passport_inference_runtime
from app.services.passport_portrait_service import preload_passport_portrait_runtime


_UPLOAD_TRACE_PATHS = {"/api/passport-inference/upload", "/api/passport-interface/upload"}
_UPLOAD_TRACE_LOG_LOCK = threading.Lock()


def _get_upload_trace_log_file_path(current_time: datetime):
    date_folder = get_log_dir() / current_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    return date_folder / "passport_inference_stage_trace.txt"


def _append_upload_trace_log(
    *,
    request_id: str,
    stage: str,
    request: Request,
    detail: str = "",
    elapsed_ms: float | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    current_time = datetime.now()
    client_ip = request.client.host if request.client else ""
    log_item: dict[str, object] = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "request_id": request_id,
        "client_ip": client_ip,
        "path": str(request.url.path),
        "method": request.method,
        "stage": stage,
        "detail": detail,
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
    if elapsed_ms is not None:
        console_parts.append(f"elapsed_ms={round(float(elapsed_ms), 2)}")
    print(" ".join(console_parts), flush=True)

    log_file_path = _get_upload_trace_log_file_path(current_time)
    with _UPLOAD_TRACE_LOG_LOCK:
        with log_file_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def _safe_append_upload_trace_log(**kwargs: object) -> None:
    try:
        _append_upload_trace_log(**kwargs)
    except Exception:
        return


allowed_origins = get_frontend_allowed_origins()

app = FastAPI(title="Passport OCR Review API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=None if "*" in allowed_origins else r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(passport_records_router, prefix="/api")
app.include_router(passport_inference_router, prefix="/api")
app.include_router(code_values_router, prefix="/api")
app.include_router(mask_review_router, prefix="/api")
app.include_router(document_type_classifier_router, prefix="/api")
app.include_router(document_type_classifier_router, include_in_schema=False)


@app.middleware("http")
async def trace_passport_upload_request(request: Request, call_next):
    if request.url.path not in _UPLOAD_TRACE_PATHS:
        return await call_next(request)

    request_id = uuid4().hex[:12]
    request.state.passport_upload_request_id = request_id
    started = perf_counter()
    _safe_append_upload_trace_log(
        request_id=request_id,
        stage="asgi_request_enter",
        request=request,
        extra={
            "content_length": request.headers.get("content-length", ""),
            "content_type": request.headers.get("content-type", ""),
            "user_agent": request.headers.get("user-agent", ""),
        },
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        _safe_append_upload_trace_log(
            request_id=request_id,
            stage="asgi_request_exception",
            request=request,
            detail=str(exc),
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        raise

    _safe_append_upload_trace_log(
        request_id=request_id,
        stage="asgi_request_exit",
        request=request,
        elapsed_ms=(perf_counter() - started) * 1000,
        extra={
            "status_code": response.status_code,
        },
    )
    return response


@app.on_event("startup")
async def preload_backend_runtime() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Preloading OCR, Donut, portrait detection, face match, and document type runtimes")
    await asyncio.to_thread(preload_ocr_runtime, fast_mode=True, include_orientation=True)
    await asyncio.to_thread(preload_passport_inference_runtime)
    await asyncio.to_thread(preload_passport_portrait_runtime)
    await asyncio.to_thread(preload_document_type_classifier_runtime)
    try:
        await asyncio.to_thread(preload_passport_face_match_runtime)
    except Exception as exc:  # pragma: no cover
        logger.warning("Face match runtime preload skipped: %s", exc)
    logger.info("Finished preloading OCR, Donut, portrait detection, face match, and document type runtimes")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
