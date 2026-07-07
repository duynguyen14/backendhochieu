from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.image_path_service import resolve_record_image_path
from app.services.passport_review_service import (
    PASSPORT_FIELD_KEYS,
    get_passport_record_detail,
    list_passport_records,
    save_passport_record_review,
)
from app.services.layoutlm_service import build_layoutlm_for_record, save_layoutlm_review
from app.services.ocr_service import run_ocr_with_boxes


router = APIRouter(tags=["passport-records"])


class EditableFieldsPayload(BaseModel):
    passport_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    passport_number: str = ""
    sex: str = ""
    date_of_birth: str = ""
    place_of_birth: str = ""
    nationality_current: str = ""
    nationality_at_birth: str = ""
    date_of_issue: str = ""
    date_of_expiry: str = ""
    issuing_authority: str = ""
    personal_number: str = ""


class LayoutLmReviewPayload(BaseModel):
    file_name: str
    tokens: list[str]
    bboxes: list[list[int]]
    ner_tags: list[str]


class UpdatePassportRecordPayload(BaseModel):
    editable_fields: EditableFieldsPayload
    status: str = Field(default="reviewed")
    layoutlm_review: LayoutLmReviewPayload | None = None


@router.get("/passport-records")
def get_passport_records(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None),
):
    return list_passport_records(page=page, page_size=page_size, status=status)


@router.get("/passport-records/{record_id}")
def get_passport_record(record_id: int):
    record = get_passport_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.put("/passport-records/{record_id}")
def update_passport_record(record_id: int, payload: UpdatePassportRecordPayload):
    if payload.status not in {"pending", "ocr_done", "reviewing", "reviewed", "error"}:
        raise HTTPException(status_code=400, detail="Invalid status")

    if payload.layoutlm_review is not None:
        if (
            len(payload.layoutlm_review.tokens) != len(payload.layoutlm_review.bboxes)
            or len(payload.layoutlm_review.tokens) != len(payload.layoutlm_review.ner_tags)
        ):
            raise HTTPException(status_code=400, detail="LayoutLM payload lengths do not match")

    editable_fields = {
        key: getattr(payload.editable_fields, key, "")
        for key in PASSPORT_FIELD_KEYS
    }

    record = save_passport_record_review(
        record_id,
        editable_fields=editable_fields,
        status=payload.status,
        layoutlm_payload=payload.layoutlm_review.model_dump() if payload.layoutlm_review is not None else None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.get("/passport-records/{record_id}/image")
def get_passport_record_image(record_id: int):
    record = get_passport_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    image_path = resolve_record_image_path(record["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        path=image_path,
        filename=image_path.name,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/passport-records/{record_id}/ocr-overlay")
def get_passport_record_ocr_overlay(record_id: int):
    record = get_passport_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    image_path = resolve_record_image_path(record["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    overlay = run_ocr_with_boxes(image_path)
    return {
        "record_id": record_id,
        "image_path": str(image_path),
        **overlay,
    }


@router.post("/passport-records/{record_id}/layoutlm/generate")
def generate_layoutlm_payload(record_id: int):
    generated = build_layoutlm_for_record(record_id)
    if generated is None:
        raise HTTPException(status_code=404, detail="Record not found")

    record = get_passport_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found after LayoutLM generation")
    return record


@router.put("/passport-records/{record_id}/layoutlm-review")
def update_layoutlm_review(record_id: int, payload: LayoutLmReviewPayload):
    if len(payload.tokens) != len(payload.bboxes) or len(payload.tokens) != len(payload.ner_tags):
        raise HTTPException(status_code=400, detail="LayoutLM payload lengths do not match")

    saved = save_layoutlm_review(
        record_id,
        payload.model_dump(),
    )
    if saved is None:
        raise HTTPException(status_code=404, detail="Record not found")

    record = get_passport_record_detail(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found after LayoutLM review update")
    return record
