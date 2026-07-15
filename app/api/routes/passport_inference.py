from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.services.ocr_field_matcher import build_ocr_field_matches, serialize_field_matches_for_api
from app.services.passport_inference_service import (
    get_inference_image_path,
    run_passport_inference,
)


router = APIRouter(tags=["passport-inference"])


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


@router.post("/passport-inference/upload")
async def upload_passport_inference(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    try:
        file_bytes = await file.read()
        result = run_passport_inference(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Passport inference failed: {exc}") from exc
    finally:
        await file.close()

    overlay = result["overlay"]
    image_width = float(overlay.get("image_width") or 0)
    image_height = float(overlay.get("image_height") or 0)
    image_url = str(request.url_for("get_passport_inference_image", image_id=result["image_id"]))
    field_matches = build_ocr_field_matches(result.get("editable_fields"), overlay)

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
