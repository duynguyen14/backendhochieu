from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.mask_review_service import (
    get_mask_review_image_path,
    get_mask_review_session,
    get_mask_review_session_for_file,
    save_mask_review_decision,
)


router = APIRouter(tags=["mask-review"])


class MaskReviewDecisionPayload(BaseModel):
    file_name: str
    decision: Literal["approved", "rejected"]


@router.get("/mask-review")
def fetch_mask_review_session(request: Request, response: Response, file_name: str = ""):
    _disable_response_cache(response)
    if file_name:
        return get_mask_review_session_for_file(request, file_name)
    return get_mask_review_session(request)


@router.get("/mask-review/images/{file_name}", name="get_mask_review_image")
def fetch_mask_review_image(file_name: str):
    try:
        image_path = get_mask_review_image_path(file_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path=image_path,
        filename=image_path.name,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.post("/mask-review/decision")
def submit_mask_review_decision(payload: MaskReviewDecisionPayload, request: Request, response: Response):
    _disable_response_cache(response)
    try:
        return save_mask_review_decision(
            file_name=payload.file_name,
            decision=payload.decision,
            request=request,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _disable_response_cache(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
