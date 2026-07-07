from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from app.database import get_connection
from app.repositories import (
    get_record_by_id,
    list_records_for_layoutlm,
    update_layoutlm_json,
    update_layoutlm_reviewed_json,
)
from app.services.ocr_service import run_ocr_with_boxes


LAYOUTLM_FIELD_TAGS: list[tuple[str, str]] = [
    ("passport_type", "PASSPORT_TYPE"),
    ("issuing_country", "ISSUING_COUNTRY"),
    ("surname", "SURNAME"),
    ("given_names", "GIVEN_NAMES"),
    ("passport_number", "PASSPORT_NUMBER"),
    ("sex", "SEX"),
    ("date_of_birth", "DOB"),
    ("place_of_birth", "POB"),
    ("nationality_current", "NATIONALITY_CURRENT"),
    ("nationality_at_birth", "NATIONALITY_AT_BIRTH"),
    ("date_of_issue", "DATE_OF_ISSUE"),
    ("date_of_expiry", "DATE_OF_EXPIRY"),
    ("issuing_authority", "PLACE_OF_ISSUE"),
    ("personal_number", "PERSONAL_NUMBER"),
]

TOKEN_PATTERN = re.compile(r"[A-Z0-9<]+(?:[/-][A-Z0-9<]+)*")


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
            gt_parse = _safe_load_json(ground_truth_payload).get("gt_parse", {})

    if not isinstance(gt_parse, dict):
        return {}

    return {key: str(value) if value is not None else "" for key, value in gt_parse.items()}


def _resolve_source_fields(row: Any) -> dict[str, str]:
    source = _get_gt_parse(row.reviewed_json) if row.reviewed_json else _get_gt_parse(row.extracted_json)
    return {field_key: source.get(field_key, "").strip() for field_key, _ in LAYOUTLM_FIELD_TAGS}


def _normalize_match_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9<]", "", value.upper())


def _tokenize_text(text: str) -> list[str]:
    return [match.group(0).upper() for match in TOKEN_PATTERN.finditer(text.upper())]


def _tokenize_with_boxes(text: str, bbox: dict[str, int]) -> list[dict[str, Any]]:
    normalized_text = text.upper()
    matches = list(TOKEN_PATTERN.finditer(normalized_text))
    if not matches:
        return []

    total_length = max(1, len(normalized_text))
    left = int(bbox["left"])
    width = int(bbox["width"])
    top = int(bbox["top"])
    height = int(bbox["height"])

    tokens: list[dict[str, Any]] = []
    for match in matches:
        start_ratio = match.start() / total_length
        end_ratio = match.end() / total_length
        token_left = left + round(width * start_ratio)
        token_right = left + round(width * end_ratio)

        tokens.append(
            {
                "text": match.group(0).upper(),
                "bbox": [
                    token_left,
                    top,
                    max(token_left + 1, token_right),
                    top + height,
                ],
            }
        )

    return tokens


def _date_variants(value: str) -> list[str]:
    if not value:
        return []

    parsed: datetime | None = None
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            parsed = datetime.strptime(value, pattern)
            break
        except ValueError:
            continue

    if parsed is None:
        return [value]

    return [
        parsed.strftime("%Y-%m-%d"),
        parsed.strftime("%d/%m/%Y"),
        parsed.strftime("%d-%m-%Y"),
        parsed.strftime("%d%m%Y"),
        parsed.strftime("%y%m%d"),
    ]


def _field_token_variants(field_key: str, raw_value: str) -> list[list[str]]:
    value = raw_value.strip()
    if not value:
        return []

    candidates = _date_variants(value) if field_key in {"date_of_birth", "date_of_issue", "date_of_expiry"} else [value]
    variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for candidate in candidates:
        token_sequence = tuple(_tokenize_text(candidate))
        if token_sequence and token_sequence not in seen:
            seen.add(token_sequence)
            variants.append(list(token_sequence))

    return sorted(variants, key=len, reverse=True)


def _find_sequence_start(tokens: list[dict[str, Any]], normalized_tokens: list[str], target_tokens: list[str]) -> int | None:
    sequence_length = len(target_tokens)
    if sequence_length == 0 or sequence_length > len(tokens):
        return None

    normalized_target = [_normalize_match_token(token) for token in target_tokens]

    for start_index in range(len(tokens) - sequence_length + 1):
        window = normalized_tokens[start_index:start_index + sequence_length]
        if any(tokens[start_index + offset]["ner_tag"] != "O" for offset in range(sequence_length)):
            continue

        if window == normalized_target:
            return start_index

    return None


def _build_layoutlm_payload(file_name: str, words: list[dict[str, Any]], source_fields: dict[str, str]) -> dict[str, Any]:
    token_entries: list[dict[str, Any]] = []

    for word in words:
        token_entries.extend(_tokenize_with_boxes(word["text"], word["bbox"]))

    if not token_entries:
        return {
            "file_name": file_name,
            "tokens": [],
            "bboxes": [],
            "ner_tags": [],
        }

    for token_entry in token_entries:
        token_entry["ner_tag"] = "O"

    normalized_tokens = [_normalize_match_token(token_entry["text"]) for token_entry in token_entries]

    for field_key, tag_name in LAYOUTLM_FIELD_TAGS:
        for variant in _field_token_variants(field_key, source_fields.get(field_key, "")):
            start_index = _find_sequence_start(token_entries, normalized_tokens, variant)
            if start_index is None:
                continue

            for offset, _ in enumerate(variant):
                token_entries[start_index + offset]["ner_tag"] = f"{'B' if offset == 0 else 'I'}-{tag_name}"
            break

    return {
        "file_name": file_name,
        "tokens": [token_entry["text"] for token_entry in token_entries],
        "bboxes": [token_entry["bbox"] for token_entry in token_entries],
        "ner_tags": [token_entry["ner_tag"] for token_entry in token_entries],
    }


def build_layoutlm_payload(
    file_name: str,
    words: list[dict[str, Any]],
    source_fields: dict[str, str],
) -> dict[str, Any]:
    return _build_layoutlm_payload(file_name, words, source_fields)


def _layoutlm_payload_to_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = payload.get("tokens", [])
    bboxes = payload.get("bboxes", [])
    ner_tags = payload.get("ner_tags", [])
    items: list[dict[str, Any]] = []

    for index, (token, bbox, ner_tag) in enumerate(zip(tokens, bboxes, ner_tags, strict=False)):
        if not isinstance(token, str) or not isinstance(bbox, list) or len(bbox) < 4:
            continue

        left, top, right, bottom = [int(value) for value in bbox[:4]]
        items.append(
            {
                "id": index,
                "text": token,
                "label": str(ner_tag),
                "bbox": {
                    "left": left,
                    "top": top,
                    "width": max(0, right - left),
                    "height": max(0, bottom - top),
                },
            }
        )

    return items


def _items_to_layoutlm_payload(file_name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_items = sorted(
        items,
        key=lambda item: (
            int(item["bbox"]["top"]),
            int(item["bbox"]["left"]),
        ),
    )

    return {
        "file_name": file_name,
        "tokens": [str(item["text"]) for item in ordered_items],
        "bboxes": [
            [
                int(item["bbox"]["left"]),
                int(item["bbox"]["top"]),
                int(item["bbox"]["left"]) + int(item["bbox"]["width"]),
                int(item["bbox"]["top"]) + int(item["bbox"]["height"]),
            ]
            for item in ordered_items
        ],
        "ner_tags": [str(item["label"]) for item in ordered_items],
    }


def normalize_layoutlm_payload(file_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _items_to_layoutlm_payload(file_name, _layoutlm_payload_to_items(payload))


def _get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        corrected = ImageOps.exif_transpose(image)
        return corrected.width, corrected.height


def _extract_numeric_stem(file_name: str) -> int | None:
    stem = Path(file_name).stem.strip()
    if not stem.isdigit():
        return None
    return int(stem)


def _build_layoutlm_for_rows(cursor: Any, rows: list[Any]) -> dict[str, int]:
    updated = 0
    errors = 0

    for row in rows:
        image_path = Path(str(row.image_path))
        if not image_path.exists():
            errors += 1
            continue

        try:
            source_fields = _resolve_source_fields(row)
            overlay = run_ocr_with_boxes(image_path)
            payload = _build_layoutlm_payload(image_path.name, overlay["words"], source_fields)
            update_layoutlm_json(
                cursor,
                record_id=int(row.id),
                layoutlm_json=json.dumps(payload, ensure_ascii=False),
            )
            updated += 1
        except Exception:
            errors += 1

    return {"updated": updated, "errors": errors}


def build_layoutlm_for_record(record_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return None

        image_path = Path(str(row.image_path))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        source_fields = _resolve_source_fields(row)
        overlay = run_ocr_with_boxes(image_path)
        payload = _build_layoutlm_payload(image_path.name, overlay["words"], source_fields)
        update_layoutlm_json(
            cursor,
            record_id=record_id,
            layoutlm_json=json.dumps(payload, ensure_ascii=False),
        )
        connection.commit()

    return get_layoutlm_detail(record_id)


def build_layoutlm_for_all_records() -> dict[str, int]:
    with get_connection() as connection:
        cursor = connection.cursor()
        rows = list_records_for_layoutlm(cursor)
        summary = _build_layoutlm_for_rows(cursor, rows)
        connection.commit()

    return summary


def build_layoutlm_for_image_index_range(start_index: int, end_index: int) -> dict[str, int]:
    with get_connection() as connection:
        cursor = connection.cursor()
        rows = [
            row
            for row in list_records_for_layoutlm(cursor)
            if (numeric_stem := _extract_numeric_stem(Path(str(row.image_path)).name)) is not None
            and start_index <= numeric_stem <= end_index
        ]
        summary = _build_layoutlm_for_rows(cursor, rows)
        connection.commit()

    return summary


def save_layoutlm_review(record_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return None

        file_name = Path(str(row.image_path)).name
        normalized_payload = normalize_layoutlm_payload(file_name, payload)
        update_layoutlm_reviewed_json(
            cursor,
            record_id=record_id,
            layoutlm_reviewed_json=json.dumps(normalized_payload, ensure_ascii=False),
        )
        connection.commit()

    return get_layoutlm_detail(record_id)


def get_layoutlm_detail(record_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        cursor = connection.cursor()
        row = get_record_by_id(cursor, record_id)
        if row is None:
            return None

    image_path = Path(str(row.image_path))
    image_width, image_height = 0, 0
    if image_path.exists():
        image_width, image_height = _get_image_size(image_path)

    generated_payload = _safe_load_json(row.layoutlm_json)
    reviewed_payload = _safe_load_json(row.layoutlm_reviewed_json)
    active_payload = reviewed_payload if row.layoutlm_reviewed_json else generated_payload

    return {
        "layoutlm_json": generated_payload,
        "layoutlm_reviewed_json": reviewed_payload,
        "layoutlm_active_json": active_payload,
        "layoutlm_items": _layoutlm_payload_to_items(active_payload),
        "layoutlm_available_labels": ["O"]
        + [f"{prefix}-{tag}" for _, tag in LAYOUTLM_FIELD_TAGS for prefix in ("B", "I")],
        "layoutlm_image_width": image_width,
        "layoutlm_image_height": image_height,
        "has_layoutlm_json": bool(row.layoutlm_json),
        "has_layoutlm_reviewed_json": bool(row.layoutlm_reviewed_json),
    }
