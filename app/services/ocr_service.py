from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import cv2
from PIL import Image, ImageOps

from app.config import (
    get_ocr_auto_rotate_and_overwrite,
    get_ocr_language,
    get_paddle_doc_orientation_model_dir,
    get_paddle_model_source,
    get_paddle_ocr_device,
    get_paddle_ocr_version,
    get_paddle_text_detection_model_dir,
    get_paddle_text_recognition_model_dir,
    get_paddle_textline_orientation_model_dir,
    get_paddle_use_doc_orientation_classify,
    get_paddle_use_textline_orientation,
)
from app.database import get_connection
from app.models import ExistingRecord
from app.repositories import insert_error_record, insert_record, load_existing_records, update_layoutlm_json


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
PASSPORT_NUMBER_PATTERN = re.compile(r"\b[A-Z][0-9A-Z]{6,8}\b")
WORD_SPLIT_PATTERN = re.compile(r"\S+")

_OCR_PIPELINE: Any | None = None
_OCR_PIPELINE_LOCK = threading.Lock()
_OCR_FAST_PIPELINE: Any | None = None
_OCR_FAST_PIPELINE_LOCK = threading.Lock()
_DOC_PREPROCESSOR_PIPELINE: Any | None = None
_DOC_PREPROCESSOR_LOCK = threading.Lock()
_IMAGE_ORIENTATION_CHECK_CACHE: dict[str, tuple[int, int]] = {}
_IMAGE_ORIENTATION_CACHE_LOCK = threading.Lock()


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\").lower()


def normalize_text(raw_text: str) -> str:
    cleaned = raw_text.replace("\x0c", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def normalize_date(date_text: str) -> str:
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            parsed = datetime.strptime(date_text, pattern)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_text


def build_empty_passport_json() -> dict[str, str]:
    return {
        "passport_type": "",
        "issuing_country": "",
        "surname": "",
        "given_names": "",
        "passport_number": "",
        "sex": "",
        "date_of_birth": "",
        "place_of_birth": "",
        "nationality_current": "",
        "nationality_at_birth": "",
        "date_of_issue": "",
        "date_of_expiry": "",
        "issuing_authority": "",
        "personal_number": "",
    }


def guess_structured_fields(raw_text: str) -> dict[str, str]:
    text = normalize_text(raw_text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper_text = text.upper()
    fields = build_empty_passport_json()
    first_line = lines[0].upper() if lines else ""

    passport_match = PASSPORT_NUMBER_PATTERN.search(upper_text)
    if passport_match:
        fields["passport_number"] = passport_match.group(0)

    dates = [normalize_date(match.group(0)) for match in DATE_PATTERN.finditer(text)]
    if dates:
        fields["date_of_birth"] = dates[0]
    if len(dates) > 1:
        fields["date_of_issue"] = dates[1]
    if len(dates) > 2:
        fields["date_of_expiry"] = dates[2]

    sex_match = re.search(r"\b(M|F|X)\b", upper_text)
    if sex_match:
        fields["sex"] = sex_match.group(1)

    country_match = re.search(r"\b([A-Z]{3})\b", upper_text)
    if country_match:
        fields["issuing_country"] = country_match.group(1)
        fields["nationality_current"] = country_match.group(1)

    if first_line:
        name_parts = first_line.split()
        if name_parts:
            fields["surname"] = name_parts[0]
        if len(name_parts) > 1:
            fields["given_names"] = " ".join(name_parts[1:])

    return fields


def _load_paddleocr_class():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR is not installed. Install paddlepaddle first, then install paddleocr. "
            "Example CPU setup: `python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/` "
            "and `python -m pip install paddleocr==3.2.0`."
        ) from exc

    return PaddleOCR


def _load_doc_preprocessor_class():
    try:
        from paddleocr import DocPreprocessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR DocPreprocessor is not available. Reinstall paddleocr and try again."
        ) from exc

    return DocPreprocessor


def _read_local_model_name(model_dir: Path) -> str | None:
    inference_config_path = model_dir / "inference.yml"
    if not inference_config_path.exists():
        return None

    try:
        config_text = inference_config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"(?m)^\s*model_name:\s*([^\s#]+)\s*$", config_text)
    if not match:
        return None

    return match.group(1).strip()


def _build_pipeline_kwargs(*, fast_mode: bool = False) -> dict[str, Any]:
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = get_paddle_model_source()

    kwargs: dict[str, Any] = {
        "lang": get_ocr_language(),
        "ocr_version": get_paddle_ocr_version(),
        "device": get_paddle_ocr_device(),
        "use_doc_orientation_classify": False if fast_mode else get_paddle_use_doc_orientation_classify(),
        "use_doc_unwarping": False,
        "use_textline_orientation": False if fast_mode else get_paddle_use_textline_orientation(),
        "return_word_box": True,
    }

    doc_orientation_model_dir = get_paddle_doc_orientation_model_dir()
    if doc_orientation_model_dir and not fast_mode:
        kwargs["doc_orientation_classify_model_dir"] = str(doc_orientation_model_dir)

    detection_model_dir = get_paddle_text_detection_model_dir()
    if detection_model_dir:
        detection_model_name = _read_local_model_name(detection_model_dir)
        if detection_model_name:
            kwargs["text_detection_model_name"] = detection_model_name
        kwargs["text_detection_model_dir"] = str(detection_model_dir)

    recognition_model_dir = get_paddle_text_recognition_model_dir()
    if recognition_model_dir:
        recognition_model_name = _read_local_model_name(recognition_model_dir)
        if recognition_model_name:
            kwargs["text_recognition_model_name"] = recognition_model_name
        kwargs["text_recognition_model_dir"] = str(recognition_model_dir)

    textline_orientation_model_dir = get_paddle_textline_orientation_model_dir()
    if textline_orientation_model_dir and not fast_mode:
        textline_orientation_model_name = _read_local_model_name(textline_orientation_model_dir)
        if textline_orientation_model_name:
            kwargs["textline_orientation_model_name"] = textline_orientation_model_name
        kwargs["textline_orientation_model_dir"] = str(textline_orientation_model_dir)

    return kwargs


def _build_doc_preprocessor_kwargs() -> dict[str, Any]:
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = get_paddle_model_source()

    kwargs: dict[str, Any] = {
        "device": get_paddle_ocr_device(),
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": False,
    }

    doc_orientation_model_dir = get_paddle_doc_orientation_model_dir()
    if doc_orientation_model_dir:
        kwargs["doc_orientation_classify_model_dir"] = str(doc_orientation_model_dir)

    return kwargs


def _get_ocr_pipeline():
    global _OCR_PIPELINE

    if _OCR_PIPELINE is not None:
        return _OCR_PIPELINE

    with _OCR_PIPELINE_LOCK:
        if _OCR_PIPELINE is None:
            PaddleOCR = _load_paddleocr_class()
            _OCR_PIPELINE = PaddleOCR(**_build_pipeline_kwargs())

    return _OCR_PIPELINE


def _get_fast_ocr_pipeline():
    global _OCR_FAST_PIPELINE

    if _OCR_FAST_PIPELINE is not None:
        return _OCR_FAST_PIPELINE

    with _OCR_FAST_PIPELINE_LOCK:
        if _OCR_FAST_PIPELINE is None:
            PaddleOCR = _load_paddleocr_class()
            _OCR_FAST_PIPELINE = PaddleOCR(**_build_pipeline_kwargs(fast_mode=True))

    return _OCR_FAST_PIPELINE


def _get_doc_preprocessor_pipeline():
    global _DOC_PREPROCESSOR_PIPELINE

    if _DOC_PREPROCESSOR_PIPELINE is not None:
        return _DOC_PREPROCESSOR_PIPELINE

    with _DOC_PREPROCESSOR_LOCK:
        if _DOC_PREPROCESSOR_PIPELINE is None:
            DocPreprocessor = _load_doc_preprocessor_class()
            _DOC_PREPROCESSOR_PIPELINE = DocPreprocessor(**_build_doc_preprocessor_kwargs())

    return _DOC_PREPROCESSOR_PIPELINE


def preload_ocr_runtime(*, fast_mode: bool = True, include_orientation: bool = True) -> None:
    if include_orientation:
        _get_doc_preprocessor_pipeline()
    if fast_mode:
        _get_fast_ocr_pipeline()
    else:
        _get_ocr_pipeline()


def _get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.width, corrected_image.height


def _read_orientation_cache_marker(image_path: Path) -> tuple[int, int]:
    stat = image_path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _is_orientation_check_cached(image_path: Path) -> bool:
    cache_key = normalize_path(image_path)
    try:
        marker = _read_orientation_cache_marker(image_path)
    except OSError:
        return False

    with _IMAGE_ORIENTATION_CACHE_LOCK:
        return _IMAGE_ORIENTATION_CHECK_CACHE.get(cache_key) == marker


def _update_orientation_cache(image_path: Path) -> None:
    cache_key = normalize_path(image_path)
    try:
        marker = _read_orientation_cache_marker(image_path)
    except OSError:
        return

    with _IMAGE_ORIENTATION_CACHE_LOCK:
        _IMAGE_ORIENTATION_CHECK_CACHE[cache_key] = marker


def _save_corrected_image(image_path: Path, corrected_image: Any) -> None:
    temp_path = image_path.with_name(f"{image_path.stem}.__rotating__{image_path.suffix}")
    write_params: list[int] = []
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        write_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    try:
        if not cv2.imwrite(str(temp_path), corrected_image, write_params):
            raise RuntimeError(f"Could not save corrected image to temporary file: {temp_path}")
        temp_path.replace(image_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _normalize_orientation_angle(raw_angle: Any) -> int:
    try:
        angle = int(raw_angle)
    except (TypeError, ValueError):
        return 0

    return angle % 360


def _rotate_and_overwrite_image(
    image_path: Path,
    *,
    angle: int,
    corrected_image: Any,
) -> dict[str, Any]:
    original_width, original_height = _get_image_size(image_path)

    if angle not in {90, 180, 270} or corrected_image is None:
        _update_orientation_cache(image_path)
        return {
            "rotated": False,
            "angle": 0,
            "original_width": original_width,
            "original_height": original_height,
            "current_width": original_width,
            "current_height": original_height,
        }

    _save_corrected_image(image_path, corrected_image)
    _update_orientation_cache(image_path)
    current_width, current_height = _get_image_size(image_path)

    return {
        "rotated": True,
        "angle": angle,
        "original_width": original_width,
        "original_height": original_height,
        "current_width": current_width,
        "current_height": current_height,
    }


def _build_no_rotation_result(image_path: Path) -> dict[str, Any]:
    width, height = _get_image_size(image_path)
    return {
        "rotated": False,
        "angle": 0,
        "original_width": width,
        "original_height": height,
        "current_width": width,
        "current_height": height,
    }


def ensure_image_orientation(image_path: Path) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not get_ocr_auto_rotate_and_overwrite() or _is_orientation_check_cached(image_path):
        return _build_no_rotation_result(image_path)

    pipeline = _get_doc_preprocessor_pipeline()
    try:
        result = pipeline.predict(str(image_path))
    except Exception as exc:  # pragma: no cover
        print(f"Orientation auto-rotate skipped for {image_path}: {exc}")
        _update_orientation_cache(image_path)
        return _build_no_rotation_result(image_path)

    if not result:
        _update_orientation_cache(image_path)
        return _build_no_rotation_result(image_path)

    item = result[0]
    angle = _normalize_orientation_angle(item.get("angle"))
    corrected_image = item.get("output_img")
    return _rotate_and_overwrite_image(
        image_path,
        angle=angle,
        corrected_image=corrected_image,
    )


def _sort_ocr_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []

    heights = [max(1, entry["box"][3] - entry["box"][1]) for entry in entries]
    line_tolerance = max(10, int(median(heights) * 0.6))
    entries_by_top = sorted(entries, key=lambda entry: (entry["box"][1], entry["box"][0]))

    lines: list[list[dict[str, Any]]] = []
    current_line: list[dict[str, Any]] = []
    current_line_top: int | None = None

    for entry in entries_by_top:
        top = int(entry["box"][1])

        if current_line_top is None or abs(top - current_line_top) <= line_tolerance:
            current_line.append(entry)
            if current_line_top is None:
                current_line_top = top
            continue

        lines.append(sorted(current_line, key=lambda item: item["box"][0]))
        current_line = [entry]
        current_line_top = top

    if current_line:
        lines.append(sorted(current_line, key=lambda item: item["box"][0]))

    return [entry for line in lines for entry in line]


def _normalize_polygon(raw_polygon: Any) -> list[list[int]]:
    if raw_polygon is None:
        return []

    polygon: list[list[int]] = []
    for point in raw_polygon:
        try:
            x_value = int(point[0])
            y_value = int(point[1])
        except (TypeError, ValueError, IndexError):
            continue

        polygon.append([x_value, y_value])

    return polygon


def _bbox_from_polygon(polygon: list[list[int]]) -> list[int] | None:
    if not polygon:
        return None

    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    return [min(x_values), min(y_values), max(x_values), max(y_values)]


def _normalize_box(raw_box: Any) -> list[int] | None:
    if raw_box is None:
        return None

    box = [int(value) for value in raw_box]
    if len(box) < 4:
        return None

    return box[:4]


def _sort_polygon_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []

    sortable_entries = [
        {
            **entry,
            "box": [int(value) for value in entry["box"][:4]],
        }
        for entry in entries
    ]

    return _sort_ocr_entries(sortable_entries)


def _calculate_rotation(polygon: list[list[int]]) -> float:
    if len(polygon) < 2:
        return 0.0

    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _build_rectangle_polygon(left: int, top: int, right: int, bottom: int) -> list[list[int]]:
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ]


def _segment_line_into_words(line_entry: dict[str, Any]) -> list[dict[str, Any]]:
    line_text = str(line_entry["text"])
    line_box = [int(value) for value in line_entry["box"][:4]]
    line_confidence = float(line_entry["confidence"])
    line_id = str(line_entry["line_id"])
    line_rotation = float(line_entry.get("rotation", 0.0))

    # Disable naive word splitting. Treat the whole line as a single "word" bounding box.
    return [
        {
            "text": line_text,
            "confidence": line_confidence,
            "polygon": line_entry.get("polygon", []),
            "box": line_box,
            "line_id": line_id,
            "rotation": line_rotation,
        }
    ]


def _extract_ocr_result(image_path: Path, *, auto_rotate: bool = True, fast_mode: bool = False) -> dict[str, Any]:
    if auto_rotate:
        ensure_image_orientation(image_path)
    pipeline = _get_fast_ocr_pipeline() if fast_mode else _get_ocr_pipeline()
    result = pipeline.predict(str(image_path))
    if not result:
        width, height = _get_image_size(image_path)
        return {
            "image_width": width,
            "image_height": height,
            "rotation_applied": 0,
            "lines": [],
            "words": [],
        }

    item = result[0]
    raw_line_entries: list[dict[str, Any]] = []

    for source_index, (text, score, raw_polygon, raw_box) in enumerate(
        zip(
            item.get("rec_texts", []),
            item.get("rec_scores", []),
            item.get("rec_polys", []),
            item.get("rec_boxes", []),
            strict=False,
        ),
        start=1,
    ):
        normalized_text = normalize_text(str(text))
        if not normalized_text:
            continue

        polygon = _normalize_polygon(raw_polygon)
        # Prefer Paddle's native rectangular box output. It is typically more stable for
        # frontend text-overlay selection than rebuilding a box from the polygon corners.
        box = _normalize_box(raw_box)
        if box is None:
            box = _bbox_from_polygon(polygon)
        if box is None:
            continue

        raw_line_entries.append(
            {
                "source_index": source_index,
                "text": normalized_text,
                "confidence": float(score),
                "polygon": polygon,
                "box": box,
                "rotation": _calculate_rotation(polygon),
            }
        )

    sorted_line_entries = _sort_polygon_entries(raw_line_entries)
    line_id_by_source_index: dict[int, str] = {}
    lines: list[dict[str, Any]] = []

    for order, entry in enumerate(sorted_line_entries, start=1):
        line_id = f"line_{order:03d}"
        line_id_by_source_index[int(entry["source_index"])] = line_id

        left, top, right, bottom = [int(value) for value in entry["box"][:4]]
        lines.append(
            {
                "id": line_id,
                "text": str(entry["text"]),
                "confidence": float(entry["confidence"]),
                "polygon": entry["polygon"],
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": max(0, right - left),
                    "height": max(0, bottom - top),
                },
                "word_ids": [],
                "order": order,
                "rotation": float(entry.get("rotation", 0.0)),
            }
        )

    line_by_id = {str(line["id"]): line for line in lines}
    raw_word_entries: list[dict[str, Any]] = []
    text_words = item.get("text_word", [])
    text_word_regions = item.get("text_word_region", [])

    text_word_boxes = item.get("text_word_boxes", [])

    for source_index, (line_words, line_regions, line_boxes) in enumerate(
        zip(text_words, text_word_regions, text_word_boxes, strict=False),
        start=1,
    ):
        line_id = line_id_by_source_index.get(source_index)
        if not line_id:
            continue

        line_confidence = float(line_by_id.get(line_id, {}).get("confidence", 0.0))
        for raw_word, raw_region, raw_word_box in zip(line_words, line_regions, line_boxes, strict=False):
            normalized_word = normalize_text(str(raw_word))
            if not normalized_word:
                continue

            polygon = _normalize_polygon(raw_region)
            # Prefer Paddle's native word boxes when available.
            box = _normalize_box(raw_word_box)
            if box is None:
                box = _bbox_from_polygon(polygon)
            if box is None:
                continue

            raw_word_entries.append(
                {
                    "text": normalized_word,
                    "confidence": line_confidence,
                    "polygon": polygon,
                    "box": box,
                    "line_id": line_id,
                    "rotation": _calculate_rotation(polygon),
                }
            )

    if not raw_word_entries:
        for source_index, line_words in enumerate(text_words, start=1):
            line_id = line_id_by_source_index.get(source_index)
            if not line_id:
                continue

            line_confidence = float(line_by_id.get(line_id, {}).get("confidence", 0.0))
            line_boxes = text_word_boxes[source_index - 1] if source_index - 1 < len(text_word_boxes) else []
            for raw_word, raw_word_box in zip(line_words, line_boxes, strict=False):
                normalized_word = normalize_text(str(raw_word))
                if not normalized_word:
                    continue

                box = _normalize_box(raw_word_box)
                if box is None:
                    continue

                raw_word_entries.append(
                    {
                        "text": normalized_word,
                        "confidence": line_confidence,
                        "polygon": _build_rectangle_polygon(box[0], box[1], box[2], box[3]),
                        "box": box,
                        "line_id": line_id,
                        "rotation": 0.0,
                    }
                )

    if not raw_word_entries:
        for line in lines:
            bbox = line["bbox"]
            raw_word_entries.extend(
                _segment_line_into_words(
                    {
                        "text": str(line["text"]),
                        "confidence": float(line["confidence"]),
                        "polygon": line["polygon"],
                        "box": [
                            int(bbox["left"]),
                            int(bbox["top"]),
                            int(bbox["right"]),
                            int(bbox["bottom"]),
                        ],
                        "line_id": str(line["id"]),
                        "rotation": float(line.get("rotation", 0.0)),
                    }
                )
            )

    sorted_word_entries = _sort_polygon_entries(raw_word_entries)
    words: list[dict[str, Any]] = []
    words_by_line_id: dict[str, list[str]] = {}

    for order, entry in enumerate(sorted_word_entries, start=1):
        left, top, right, bottom = [int(value) for value in entry["box"][:4]]
        word_id = f"word_{order:04d}"
        line_id = str(entry["line_id"])
        words_by_line_id.setdefault(line_id, []).append(word_id)

        words.append(
            {
                "id": word_id,
                "text": str(entry["text"]),
                "confidence": float(entry["confidence"]),
                "polygon": entry["polygon"],
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": max(0, right - left),
                    "height": max(0, bottom - top),
                },
                "line_id": line_id,
                "order": order,
                "rotation": float(entry.get("rotation", 0.0)),
            }
        )

    for line in lines:
        line["word_ids"] = words_by_line_id.get(str(line["id"]), [])

    width, height = _get_image_size(image_path)
    return {
        "image_width": width,
        "image_height": height,
        "rotation_applied": 0,
        "lines": lines,
        "words": words,
    }


def run_ocr(image_path: Path, *, auto_rotate: bool = True, fast_mode: bool = False) -> str:
    extracted = _extract_ocr_result(image_path, auto_rotate=auto_rotate, fast_mode=fast_mode)
    return _ocr_result_to_text(extracted)


def run_ocr_with_boxes(image_path: Path, *, auto_rotate: bool = True, fast_mode: bool = False) -> dict[str, object]:
    extracted = _extract_ocr_result(image_path, auto_rotate=auto_rotate, fast_mode=fast_mode)
    return _ocr_result_to_overlay(extracted)


def _ocr_result_to_text(extracted: dict[str, Any]) -> str:
    lines = [
        str(line["text"])
        for line in extracted["lines"]
        if str(line.get("text", "")).strip()
    ]
    return normalize_text("\n".join(lines))


def _ocr_result_to_overlay(extracted: dict[str, Any]) -> dict[str, object]:
    return {
        "image_width": extracted["image_width"],
        "image_height": extracted["image_height"],
        "rotation_applied": extracted.get("rotation_applied", 0),
        "words": extracted["words"],
        "lines": extracted["lines"],
    }


def _extract_numeric_stem(image_path: Path) -> int | None:
    stem = image_path.stem.strip()
    if not stem.isdigit():
        return None
    return int(stem)


def _image_sort_key(image_path: Path) -> tuple[int, int | str, str]:
    numeric_stem = _extract_numeric_stem(image_path)
    if numeric_stem is not None:
        return (0, numeric_stem, image_path.suffix.lower())
    return (1, image_path.name.lower(), image_path.suffix.lower())


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]

    pattern = "**/*" if recursive else "*"
    return sorted(
        [
            path.resolve()
            for path in input_path.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=_image_sort_key,
    )


def filter_images_by_index_range(
    images: list[Path],
    *,
    start_index: int | None = None,
    end_index: int | None = None,
) -> list[Path]:
    if start_index is None and end_index is None:
        return images

    filtered_images: list[Path] = []
    for image_path in images:
        numeric_stem = _extract_numeric_stem(image_path)
        if numeric_stem is None:
            continue
        if start_index is not None and numeric_stem < start_index:
            continue
        if end_index is not None and numeric_stem > end_index:
            continue
        filtered_images.append(image_path)

    return filtered_images


def is_duplicate(image_path: Path, existing_records: list[ExistingRecord]) -> ExistingRecord | None:
    normalized_path = normalize_path(image_path)
    file_name = image_path.name.lower()

    for record in existing_records:
        if record.normalized_path == normalized_path or record.file_name == file_name:
            return record

    return None


def process_images_to_database(
    input_path: Path,
    recursive: bool = False,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    generate_layoutlm: bool = False,
) -> dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = collect_images(input_path, recursive)
    images = filter_images_by_index_range(
        images,
        start_index=start_index,
        end_index=end_index,
    )
    if not images:
        print(f"No supported images found at: {input_path}")
        return {"inserted": 0, "skipped_duplicate": 0, "errors": 0}

    with get_connection() as connection:
        cursor = connection.cursor()
        existing_records = load_existing_records(cursor)

        inserted_count = 0
        skipped_count = 0
        error_count = 0

        for image_path in images:
            duplicate_record = is_duplicate(image_path, existing_records)
            if duplicate_record is not None:
                print(
                    f"Skip duplicate: {image_path.name} | existing_id={duplicate_record.id} | existing_path={duplicate_record.image_path}"
                )
                skipped_count += 1
                continue

            try:
                extracted = _extract_ocr_result(image_path)
                raw_ocr_text = _ocr_result_to_text(extracted)
                extracted_payload = {"gt_parse": guess_structured_fields(raw_ocr_text)}
                extracted_json = json.dumps(extracted_payload, ensure_ascii=False)
                record_id = insert_record(cursor, image_path, extracted_json, raw_ocr_text)

                if generate_layoutlm:
                    from app.services.layoutlm_service import build_layoutlm_payload

                    overlay = _ocr_result_to_overlay(extracted)
                    layoutlm_payload = build_layoutlm_payload(
                        image_path.name,
                        overlay["words"],
                        extracted_payload["gt_parse"],
                    )
                    update_layoutlm_json(
                        cursor,
                        record_id=record_id,
                        layoutlm_json=json.dumps(layoutlm_payload, ensure_ascii=False),
                    )

                connection.commit()

                existing_records.append(
                    ExistingRecord(
                        id=-1,
                        image_path=str(image_path.resolve()),
                        normalized_path=normalize_path(image_path),
                        file_name=image_path.name.lower(),
                    )
                )
                inserted_count += 1
                print(f"Inserted OCR result with PaddleOCR: {image_path}")
            except Exception as exc:  # pragma: no cover
                insert_error_record(cursor, image_path, str(exc))
                connection.commit()
                existing_records.append(
                    ExistingRecord(
                        id=-1,
                        image_path=str(image_path.resolve()),
                        normalized_path=normalize_path(image_path),
                        file_name=image_path.name.lower(),
                    )
                )
                error_count += 1
                print(f"Inserted error record: {image_path} | {exc}")

    return {
        "inserted": inserted_count,
        "skipped_duplicate": skipped_count,
        "errors": error_count,
    }
