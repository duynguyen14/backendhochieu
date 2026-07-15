from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from app.services.ocr_service import build_empty_passport_json, normalize_date


_NON_ALNUM_PATTERN = re.compile(r"[^0-9A-Z]+")
_MULTISPACE_PATTERN = re.compile(r"\s+")
_DIGIT_PATTERN = re.compile(r"\d+")
_DATE_COMPACT_PATTERN = re.compile(r"^\d{8}$")

_FIELD_MATCH_THRESHOLDS: dict[str, float] = {
    "passport_type": 0.99,
    "issuing_country": 0.99,
    "surname": 0.74,
    "given_names": 0.72,
    "passport_number": 0.92,
    "sex": 0.99,
    "date_of_birth": 0.95,
    "place_of_birth": 0.72,
    "nationality_current": 0.99,
    "nationality_at_birth": 0.99,
    "date_of_issue": 0.95,
    "date_of_expiry": 0.95,
    "issuing_authority": 0.72,
    "personal_number": 0.9,
}


def build_ocr_field_matches(
    editable_fields: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    normalized_fields = build_empty_passport_json()
    if editable_fields:
        for field_key in normalized_fields:
            value = editable_fields.get(field_key, "")
            normalized_fields[field_key] = str(value or "")

    if not overlay:
        return {
            field_key: _build_empty_match(field_key, value)
            for field_key, value in normalized_fields.items()
        }

    candidates = _build_candidates(overlay)
    matches: dict[str, dict[str, Any]] = {}
    for field_key, expected_value in normalized_fields.items():
        matches[field_key] = _find_best_match(field_key, expected_value, candidates)

    return matches


def serialize_field_matches_for_api(
    field_matches: dict[str, dict[str, Any]],
    image_width: float,
    image_height: float,
) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for field_key, match in field_matches.items():
        bbox = match.get("bbox") or {}
        serialized[field_key] = {
            "field_key": field_key,
            "expected_value": str(match.get("expected_value") or ""),
            "text": str(match.get("text") or ""),
            "score": round(float(match.get("score") or 0.0), 4),
            "matched": bool(match.get("matched")),
            "match_type": str(match.get("match_type") or "none"),
            "source": str(match.get("source") or ""),
            "word_ids": [str(value) for value in match.get("word_ids", [])],
            "line_ids": [str(value) for value in match.get("line_ids", [])],
            "boundingBox": {
                "top": _to_percent(float(bbox.get("top") or 0), image_height),
                "left": _to_percent(float(bbox.get("left") or 0), image_width),
                "width": _to_percent(float(bbox.get("width") or 0), image_width),
                "height": _to_percent(float(bbox.get("height") or 0), image_height),
            },
        }
    return serialized


def _to_percent(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round((value / total) * 100, 4)


def _build_empty_match(field_key: str, expected_value: str) -> dict[str, Any]:
    return {
        "field_key": field_key,
        "expected_value": str(expected_value or ""),
        "text": "",
        "score": 0.0,
        "matched": False,
        "match_type": "none",
        "source": "",
        "word_ids": [],
        "line_ids": [],
        "bbox": {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
            "width": 0,
            "height": 0,
        },
    }


def _build_candidates(overlay: dict[str, Any]) -> list[dict[str, Any]]:
    words = overlay.get("words", [])
    lines = overlay.get("lines", [])
    word_by_line: dict[str, list[dict[str, Any]]] = {}

    for word in words:
        line_id = str(word.get("line_id") or "")
        if not line_id:
            continue
        word_by_line.setdefault(line_id, []).append(word)

    for grouped_words in word_by_line.values():
        grouped_words.sort(key=lambda item: int(item.get("order") or 0))

    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[str, ...], str]] = set()

    for line in lines:
        line_id = str(line.get("id") or "")
        if not line_id:
            continue

        line_words = word_by_line.get(line_id, [])
        if line_words:
            max_span_length = min(8, max(1, len(line_words)))
            for start_index in range(len(line_words)):
                for span_length in range(1, max_span_length + 1):
                    end_index = start_index + span_length
                    if end_index > len(line_words):
                        break
                    span_words = line_words[start_index:end_index]
                    _append_candidate(
                        candidates,
                        seen_keys,
                        text=" ".join(str(word.get("text") or "").strip() for word in span_words).strip(),
                        words=span_words,
                        line_ids=[line_id],
                        source="word_span",
                    )

        _append_candidate(
            candidates,
            seen_keys,
            text=str(line.get("text") or "").strip(),
            words=line_words,
            line_ids=[line_id],
            source="line",
            fallback_bbox=line.get("bbox"),
        )

    if not candidates:
        for word in words:
            _append_candidate(
                candidates,
                seen_keys,
                text=str(word.get("text") or "").strip(),
                words=[word],
                line_ids=[str(word.get("line_id") or "")] if word.get("line_id") else [],
                source="word",
            )

    return candidates


def _append_candidate(
    candidates: list[dict[str, Any]],
    seen_keys: set[tuple[str, tuple[str, ...], str]],
    *,
    text: str,
    words: list[dict[str, Any]],
    line_ids: list[str],
    source: str,
    fallback_bbox: dict[str, Any] | None = None,
) -> None:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return

    word_ids = [str(word.get("id") or "") for word in words if str(word.get("id") or "")]
    candidate_key = (normalized_text, tuple(word_ids), source)
    if candidate_key in seen_keys:
        return

    bbox = _merge_bboxes([word.get("bbox") for word in words], fallback_bbox=fallback_bbox)
    if bbox is None:
        return

    seen_keys.add(candidate_key)
    candidates.append(
        {
            "text": text.strip(),
            "normalized_text": normalized_text,
            "compact_text": _compact_text(text),
            "word_ids": word_ids,
            "line_ids": [line_id for line_id in line_ids if line_id],
            "bbox": bbox,
            "source": source,
        }
    )


def _merge_bboxes(
    raw_bboxes: list[dict[str, Any] | None],
    *,
    fallback_bbox: dict[str, Any] | None = None,
) -> dict[str, int] | None:
    bboxes = [bbox for bbox in raw_bboxes if isinstance(bbox, dict)]
    if not bboxes and isinstance(fallback_bbox, dict):
        bboxes = [fallback_bbox]
    if not bboxes:
        return None

    left = min(int(bbox.get("left") or 0) for bbox in bboxes)
    top = min(int(bbox.get("top") or 0) for bbox in bboxes)
    right = max(int(bbox.get("right") or (int(bbox.get("left") or 0) + int(bbox.get("width") or 0))) for bbox in bboxes)
    bottom = max(int(bbox.get("bottom") or (int(bbox.get("top") or 0) + int(bbox.get("height") or 0))) for bbox in bboxes)

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def _find_best_match(
    field_key: str,
    expected_value: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    cleaned_expected_value = str(expected_value or "").strip()
    if not cleaned_expected_value:
        return _build_empty_match(field_key, cleaned_expected_value)

    threshold = _FIELD_MATCH_THRESHOLDS.get(field_key, 0.75)
    best_candidate: dict[str, Any] | None = None
    best_score = 0.0
    best_match_type = "none"

    for candidate in candidates:
        score, match_type = _score_candidate(field_key, cleaned_expected_value, candidate)
        if score <= 0:
            continue
        if score > best_score:
            best_candidate = candidate
            best_score = score
            best_match_type = match_type

    if best_candidate is None:
        return _build_empty_match(field_key, cleaned_expected_value)

    return {
        "field_key": field_key,
        "expected_value": cleaned_expected_value,
        "text": str(best_candidate.get("text") or ""),
        "score": best_score,
        "matched": best_score >= threshold,
        "match_type": best_match_type,
        "source": str(best_candidate.get("source") or ""),
        "word_ids": list(best_candidate.get("word_ids") or []),
        "line_ids": list(best_candidate.get("line_ids") or []),
        "bbox": dict(best_candidate.get("bbox") or {}),
    }


def _score_candidate(
    field_key: str,
    expected_value: str,
    candidate: dict[str, Any],
) -> tuple[float, str]:
    if field_key in {"passport_type", "sex"}:
        return _score_code_value(field_key, expected_value, candidate)

    if field_key in {"issuing_country", "nationality_current", "nationality_at_birth"}:
        return _score_country_code(expected_value, candidate)

    if field_key in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
        return _score_date_value(expected_value, candidate)

    if field_key in {"passport_number", "personal_number"}:
        return _score_identifier_value(expected_value, candidate)

    return _score_text_value(expected_value, candidate)


def _score_code_value(
    field_key: str,
    expected_value: str,
    candidate: dict[str, Any],
) -> tuple[float, str]:
    expected_variants = _code_variants(field_key, expected_value)
    candidate_compact = str(candidate.get("compact_text") or "")
    candidate_normalized = str(candidate.get("normalized_text") or "")
    if candidate_compact in expected_variants or candidate_normalized in expected_variants:
        return 1.0, "exact"
    return 0.0, "none"


def _score_country_code(expected_value: str, candidate: dict[str, Any]) -> tuple[float, str]:
    expected_code = _compact_text(expected_value)
    if len(expected_code) != 3:
        return 0.0, "none"

    candidate_compact = str(candidate.get("compact_text") or "")
    if candidate_compact == expected_code:
        return 1.0, "exact"

    if candidate_compact.startswith(expected_code) or expected_code in candidate_compact.split():
        return 0.92, "contains"

    return 0.0, "none"


def _score_date_value(expected_value: str, candidate: dict[str, Any]) -> tuple[float, str]:
    expected_variants = _date_variants(expected_value)
    if not expected_variants:
        return 0.0, "none"

    raw_text = str(candidate.get("text") or "")
    normalized_text = str(candidate.get("normalized_text") or "")
    compact_text = str(candidate.get("compact_text") or "")
    candidate_variants = _date_variants(raw_text) | _date_variants(normalized_text) | _date_variants(compact_text)

    if expected_variants & candidate_variants:
        return 1.0, "date"

    return 0.0, "none"


def _score_identifier_value(expected_value: str, candidate: dict[str, Any]) -> tuple[float, str]:
    expected_compact = _compact_text(expected_value)
    candidate_compact = str(candidate.get("compact_text") or "")
    if not expected_compact or not candidate_compact:
        return 0.0, "none"

    if candidate_compact == expected_compact:
        return 1.0, "exact"

    if candidate_compact.startswith(expected_compact) or expected_compact.startswith(candidate_compact):
        ratio = len(candidate_compact) / max(len(expected_compact), 1)
        return min(0.96, 0.88 + ratio * 0.08), "prefix"

    sequence_ratio = SequenceMatcher(None, expected_compact, candidate_compact).ratio()
    return sequence_ratio * 0.9, "fuzzy"


def _score_text_value(expected_value: str, candidate: dict[str, Any]) -> tuple[float, str]:
    expected_normalized = _normalize_text(expected_value)
    expected_compact = _compact_text(expected_value)
    candidate_normalized = str(candidate.get("normalized_text") or "")
    candidate_compact = str(candidate.get("compact_text") or "")
    if not expected_normalized or not candidate_normalized:
        return 0.0, "none"

    if candidate_normalized == expected_normalized or candidate_compact == expected_compact:
        return 1.0, "exact"

    sequence_ratio = SequenceMatcher(None, expected_normalized, candidate_normalized).ratio()
    compact_ratio = SequenceMatcher(None, expected_compact, candidate_compact).ratio()
    token_overlap = _token_overlap_score(expected_normalized, candidate_normalized)

    contains_bonus = 0.0
    if len(expected_compact) >= 4 and (expected_compact in candidate_compact or candidate_compact in expected_compact):
        contains_bonus = 1.0

    score = max(sequence_ratio, compact_ratio) * 0.5 + token_overlap * 0.35 + contains_bonus * 0.15
    match_type = "fuzzy"
    if contains_bonus >= 1.0 and max(sequence_ratio, compact_ratio) >= 0.65:
        match_type = "contains"

    return min(score, 0.99), match_type


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = [token for token in left.split(" ") if token]
    right_tokens = [token for token in right.split(" ") if token]
    if not left_tokens or not right_tokens:
        return 0.0

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    intersection = len(left_set & right_set)
    if intersection == 0:
        return 0.0

    precision = intersection / len(right_set)
    recall = intersection / len(left_set)
    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def _code_variants(field_key: str, value: str) -> set[str]:
    compact = _compact_text(value)
    if not compact:
        return set()

    if field_key == "passport_type":
        if compact.startswith("PO"):
            return {"PO"}
        if compact.startswith("PD"):
            return {"PD"}
        if compact.startswith("P"):
            return {"P", "P<"}

    if field_key == "sex":
        if compact in {"M", "MALE", "NAM"}:
            return {"M", "MALE", "NAM"}
        if compact in {"F", "FEMALE", "NU"}:
            return {"F", "FEMALE", "NU"}
        if compact in {"X", "OTHER", "UNKNOWN"}:
            return {"X", "OTHER", "UNKNOWN"}

    return {compact}


def _date_variants(value: str) -> set[str]:
    cleaned_value = str(value or "").strip()
    if not cleaned_value:
        return set()

    variants: set[str] = set()
    normalized_input = cleaned_value.replace(".", "/").replace(" ", "")
    normalized_date = _safe_normalize_date(normalized_input)
    if normalized_date:
        variants.add(normalized_date)
        variants.add(normalized_date.replace("-", ""))

        try:
            parsed = datetime.strptime(normalized_date, "%Y-%m-%d")
        except ValueError:
            parsed = None

        if parsed is not None:
            variants.add(parsed.strftime("%d/%m/%Y"))
            variants.add(parsed.strftime("%d-%m-%Y"))
            variants.add(parsed.strftime("%d%m%Y"))
            variants.add(parsed.strftime("%y%m%d"))

    digit_groups = _DIGIT_PATTERN.findall(cleaned_value)
    joined_digits = "".join(digit_groups)
    if _DATE_COMPACT_PATTERN.fullmatch(joined_digits):
        variants.add(joined_digits)

    return {variant for variant in variants if variant}


def _safe_normalize_date(value: str) -> str:
    try:
        normalized = normalize_date(value)
    except Exception:
        return ""

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        return normalized
    return ""


def _normalize_text(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = ascii_text.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.upper()
    ascii_text = ascii_text.replace("<", " ")
    ascii_text = _NON_ALNUM_PATTERN.sub(" ", ascii_text)
    ascii_text = _MULTISPACE_PATTERN.sub(" ", ascii_text).strip()
    return ascii_text


def _compact_text(value: str) -> str:
    return _normalize_text(value).replace(" ", "")
