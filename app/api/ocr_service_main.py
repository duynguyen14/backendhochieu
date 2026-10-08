from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.ocr_service import preload_ocr_runtime, run_ocr_with_boxes


app = FastAPI(title="Passport OCR Worker API", version="1.0.0")


class PassportOcrPayload(BaseModel):
    image_path: str = Field(..., min_length=1)
    auto_rotate: bool = False
    fast_mode: bool = True


@app.on_event("startup")
async def preload_ocr_worker_runtime() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Preloading Passport OCR worker runtime")
    await asyncio.to_thread(preload_ocr_runtime, fast_mode=True, include_orientation=False)
    logger.info("Finished preloading Passport OCR worker runtime")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr/passport")
async def run_passport_ocr(payload: PassportOcrPayload) -> dict[str, Any]:
    image_path = Path(payload.image_path).expanduser().resolve()
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    started = perf_counter()
    try:
        overlay = await asyncio.to_thread(
            run_ocr_with_boxes,
            image_path,
            auto_rotate=payload.auto_rotate,
            fast_mode=payload.fast_mode,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"OCR worker failed: {exc}") from exc

    return {
        "status": "success",
        "data": overlay,
        "performance": {
            "ocr_worker_duration_ms": round((perf_counter() - started) * 1000, 2),
        },
    }
