from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.database import get_connection
from app.repositories import (
    count_records_with_status,
    get_next_record_id,
    get_previous_record_id,
    get_record_by_id,
    list_records_paginated,
    update_layoutlm_json,
    update_layoutlm_reviewed_json,
    update_reviewed_record,
)
from app.services.layoutlm_service import get_layoutlm_detail, normalize_layoutlm_payload
from app.services.ocr_service import ensure_image_orientation


PASSPORT_FIELD_KEYS = [
    "passport_type",
    "issuing_country",
    "surname",
    "given_names",
    "passport_number",
    "sex",
    "date_of_birth",
    "place_of_birth",
    "nationality_current",
    "nationality_at_birth",
    "date_of_issue",
    "date_of_expiry",
    "issuing_authority",
    "personal_number",
]


def _safe_load_json(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}

    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    return loaded if isinstance(loaded, dict) else {}


def _get_gt_parse(payload: str | None) -> dict[str, str]:
    loaded = _safe_load_json(payload)
    gt_parse: Any = loaded.get("gt_parse")

    if gt_parse is None and "ground_truth" in loaded:
        ground_truth_payload = loaded.get("ground_truth")
        if isinstance(ground_truth_payload, str):
            ground_truth_loaded = _safe_load_json(ground_truth_payload)
            gt_parse = ground_truth_loaded.get("gt_parse", {})
        else:
            gt_parse = {}

    if not isinstance(gt_parse, dict):
        return {}
    return {key: str(value) if value is not None else "" for key, value in gt_parse.items()}


def _resolve_editable_fields(extracted_json: str | None, reviewed_json: str | None) -> dict[str, str]:
    source_fields = _get_gt_parse(reviewed_json) if reviewed_json else _get_gt_parse(extracted_json)

    resolved = {key: "" for key in PASSPORT_FIELD_KEYS}
    for key in PASSPORT_FIELD_KEYS:
        resolved[key] = source_fields.get(key, "")

    return resolved


def _serialize_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""


def _build_display_name(fields: dict[str, str]) -> str:
    surname = fields.get("surname", "").strip()
    given_names = fields.get("given_names", "").strip()
    return " ".join(part for part in [surname, given_names] if part)


def _rotate_point(x: int, y: int, width: int, height: int, angle: int) -> tuple[int, int]:
    normalized_angle = angle % 360
    if normalized_angle == 90:
        return y, width - x
    if normalized_angle == 180:
        return width - x, height - y
    if normalized_angle == 270:
        return height - y, x
    return x, y


def _rotate_bbox(bbox: list[int], width: int, height: int, angle: int) -> list[int]:
    if len(bbox) < 4:
        return bbox

    left, top, right, bottom = [int(value) for value in bbox[:4]]
    corners = [
        _rotate_point(left, top, width, height, angle),
        _rotate_point(right, top, width, height, angle),
        _rotate_point(right, bottom, width, height, angle),
        _rotate_point(left, bottom, width, height, angle),
    ]
    rotated_x = [point[0] for point in corners]
    rotated_y = [point[1] for point in corners]
    return [
        min(rotated_x),
        min(rotated_y),
        max(rotated_x),
        max(rotated_y),
    ]


def _rotate_layoutlm_payload_bboxes(payload: str | None, width: int, height: int, angle: int) -> str | None:
    loaded = _safe_load_json(payload)
    bboxes = loaded.get("bboxes")
    if not isinstance(bboxes, list):
        return payload

    loaded["bboxes"] = [
        _rotate_bbox(bbox, width, height, angle) if isinstance(bbox, list) else bbox
        for bbox in bboxes
    ]
    return json.dumps(loaded, ensure_ascii=False)


def _sync_record_orientation(record_id: int) -> None:
    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return

        image_path = Path(str(row.image_path))
        if not image_path.exists():
            return

        orientation = ensure_image_orientation(image_path)
        if not orientation["rotated"]:
            return

        angle = int(orientation["angle"])
        original_width = int(orientation["original_width"])
        original_height = int(orientation["original_height"])

        if row.layoutlm_json:
            rotated_layoutlm_json = _rotate_layoutlm_payload_bboxes(
                str(row.layoutlm_json),
                original_width,
                original_height,
                angle,
            )
            if rotated_layoutlm_json is not None:
                update_layoutlm_json(
                    cursor,
                    record_id=record_id,
                    layoutlm_json=rotated_layoutlm_json,
                )

        if row.layoutlm_reviewed_json:
            rotated_layoutlm_reviewed_json = _rotate_layoutlm_payload_bboxes(
                str(row.layoutlm_reviewed_json),
                original_width,
                original_height,
                angle,
            )
            if rotated_layoutlm_reviewed_json is not None:
                update_layoutlm_reviewed_json(
                    cursor,
                    record_id=record_id,
                    layoutlm_reviewed_json=rotated_layoutlm_reviewed_json,
                )

        connection.commit()


def list_passport_records(*, page: int, page_size: int, status: str | None = None) -> dict[str, Any]:
    with get_connection() as connection:
        cursor = connection.cursor()
        total = count_records_with_status(cursor, status)
        rows = list_records_paginated(cursor, page=page, page_size=page_size, status=status)

    items = []
    for row in rows:
        editable_fields = _resolve_editable_fields(row.extracted_json, row.reviewed_json)
        items.append(
            {
                "id": int(row.id),
                "image_path": str(row.image_path),
                "image_name": Path(str(row.image_path)).name,
                "status": str(row.status),
                "created_at": _serialize_datetime(row.created_at),
                "updated_at": _serialize_datetime(row.updated_at),
                "full_name": _build_display_name(editable_fields),
                "passport_number": editable_fields.get("passport_number", ""),
                "date_of_birth": editable_fields.get("date_of_birth", ""),
                "has_reviewed_json": bool(row.reviewed_json),
                "has_layoutlm_json": bool(row.layoutlm_json),
                "has_layoutlm_reviewed_json": bool(row.layoutlm_reviewed_json),
            }
        )

    total_pages = max(1, (total + page_size - 1) // page_size) if page_size > 0 else 1
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": total_pages,
        },
    }


def get_passport_record_detail(record_id: int) -> dict[str, Any] | None:
    _sync_record_orientation(record_id)

    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return None

        previous_record_id = get_previous_record_id(cursor, record_id)
        next_record_id = get_next_record_id(cursor, record_id)

    layoutlm_detail = get_layoutlm_detail(record_id)

    response = {
        "id": int(row.id),
        "image_path": str(row.image_path),
        "image_name": Path(str(row.image_path)).name,
        "status": str(row.status),
        "created_at": _serialize_datetime(row.created_at),
        "updated_at": _serialize_datetime(row.updated_at),
        "error_message": str(row.error_message) if row.error_message is not None else "",
        "raw_ocr_text": str(row.raw_ocr_text) if row.raw_ocr_text is not None else "",
        "extracted_json": _safe_load_json(row.extracted_json),
        "reviewed_json": _safe_load_json(row.reviewed_json),
        "editable_fields": _resolve_editable_fields(row.extracted_json, row.reviewed_json),
        "previous_record_id": previous_record_id,
        "next_record_id": next_record_id,
    }
    if layoutlm_detail:
        response.update(layoutlm_detail)

    return response


def save_passport_record_review(
    record_id: int,
    *,
    editable_fields: dict[str, str],
    status: str,
    layoutlm_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return None

        file_name = Path(str(row.image_path)).name
        gt_parse_payload = {
            key: str(editable_fields.get(key, "") or "")
            for key in PASSPORT_FIELD_KEYS
        }
        payload = {
            "file_name": file_name,
            "ground_truth": json.dumps(
                {"gt_parse": gt_parse_payload},
                ensure_ascii=False,
            ),
        }
        reviewed_json = json.dumps(payload, ensure_ascii=False)

        update_reviewed_record(
            cursor,
            record_id=record_id,
            reviewed_json=reviewed_json,
            status=status,
        )

        if layoutlm_payload is not None:
            normalized_layoutlm_payload = normalize_layoutlm_payload(file_name, layoutlm_payload)
            update_layoutlm_reviewed_json(
                cursor,
                record_id=record_id,
                layoutlm_reviewed_json=json.dumps(normalized_layoutlm_payload, ensure_ascii=False),
            )
        connection.commit()

    return get_passport_record_detail(record_id)
