from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.document_type_classifier_service import detect_document_type_from_base64


router = APIRouter(tags=["document-type-classifier"])


class DocumentTypeDetectPayload(BaseModel):
    base64: str | None = Field(default=None)
    dataBase64: str | None = Field(default=None)
    image_base64: str | None = Field(default=None)
    file_name: str | None = Field(default="document_upload.jpg")


@router.post("/file-type-detect-base64")
@router.post("/detect-file-base64")
@router.post("/external/file-type-detect")
async def detect_document_type(payload: DocumentTypeDetectPayload) -> dict[str, object]:
    base64_payload = payload.base64 or payload.dataBase64 or payload.image_base64
    if not base64_payload:
        raise HTTPException(status_code=400, detail="base64 is required")

    try:
        return await asyncio.to_thread(
            detect_document_type_from_base64,
            base64_payload,
            payload.file_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
