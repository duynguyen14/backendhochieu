from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2

from app.config import get_passport_portrait_output_dir
from app.services.ocr_service import run_ocr_with_boxes


_FACE_CASCADE: Any | None = None


def detect_passport_portrait(image_path: Path, overlay: dict[str, Any] | None = None) -> dict[str, Any]:
    source_path = image_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Image file not found: {source_path}")

    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"Could not read image with OpenCV: {source_path}")

    image_height, image_width = image.shape[:2]
    if overlay is None:
        try:
            overlay = run_ocr_with_boxes(source_path, auto_rotate=False, fast_mode=True)
        except Exception:
            overlay = None
    face_candidates = _detect_face_candidates(image)
    best_face = _select_best_face(face_candidates, image, overlay)

    if best_face is None:
        return {
            "detected": False,
            "image_path": str(source_path),
            "image_width": image_width,
            "image_height": image_height,
            "face_bbox": _empty_bbox(),
            "portrait_bbox": _empty_bbox(),
            "portrait_image_path": "",
        }

    face_bbox = _face_tuple_to_bbox(best_face)
    portrait_bbox = _expand_face_to_portrait_bbox(face_bbox, image_width, image_height)
    portrait_output_path = _save_portrait_crop(image, portrait_bbox, source_path)

    return {
        "detected": True,
        "image_path": str(source_path),
        "image_width": image_width,
        "image_height": image_height,
        "face_bbox": face_bbox,
        "portrait_bbox": portrait_bbox,
        "portrait_image_path": str(portrait_output_path),
    }


def get_passport_portrait_image_path(source_image_path: Path) -> Path | None:
    result = detect_passport_portrait(source_image_path)
    portrait_image_path = str(result.get("portrait_image_path") or "")
    if not portrait_image_path:
        return None
    return Path(portrait_image_path)


def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")

    _FACE_CASCADE = cascade
    return _FACE_CASCADE


def _detect_face_candidates(image: Any) -> list[tuple[int, int, int, int]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    cascade = _get_face_cascade()

    min_face_width = max(36, int(image.shape[1] * 0.08))
    min_face_height = max(36, int(image.shape[0] * 0.12))

    detected_faces: list[tuple[int, int, int, int]] = []
    for scale_factor, min_neighbors in ((1.08, 5), (1.05, 4), (1.12, 6)):
        faces = cascade.detectMultiScale(
            grayscale,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_width, min_face_height),
        )
        detected_faces.extend(
            (int(x), int(y), int(width), int(height))
            for (x, y, width, height) in faces
        )

    unique_faces: list[tuple[int, int, int, int]] = []
    seen_keys: set[tuple[int, int, int, int]] = set()
    for face in detected_faces:
        key = tuple(int(round(value / 4.0)) for value in face)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_faces.append(face)

    return unique_faces


def _select_best_face(
    face_candidates: list[tuple[int, int, int, int]],
    image: Any,
    overlay: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    if not face_candidates:
        return None

    sorted_by_area = sorted(
        face_candidates,
        key=lambda face: int(face[2]) * int(face[3]),
        reverse=True,
    )
    if len(sorted_by_area) == 1:
        return sorted_by_area[0]

    largest_area = int(sorted_by_area[0][2]) * int(sorted_by_area[0][3])
    second_area = int(sorted_by_area[1][2]) * int(sorted_by_area[1][3])
    if largest_area >= int(second_area * 1.05):
        return sorted_by_area[0]

    image_height, image_width = image.shape[:2]
    best_face: tuple[int, int, int, int] | None = None
    best_score = float("-inf")

    for face in face_candidates:
        score = _score_face_candidate(face, image_width, image_height, image, overlay)
        if score > best_score:
            best_score = score
            best_face = face

    return best_face


def _score_face_candidate(
    face: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    image: Any,
    overlay: dict[str, Any] | None,
) -> float:
    x_value, y_value, width, height = face
    area_ratio = (width * height) / max(1.0, float(image_width * image_height))
    center_x_ratio = (x_value + (width / 2.0)) / max(1.0, float(image_width))
    center_y_ratio = (y_value + (height / 2.0)) / max(1.0, float(image_height))

    size_score = min(area_ratio * 28.0, 6.0)
    side_bonus = 1.4 if center_x_ratio <= 0.42 or center_x_ratio >= 0.58 else -1.5
    vertical_bonus = 1.0 if 0.18 <= center_y_ratio <= 0.72 else -0.8
    aspect_penalty = abs(1.0 - (width / max(1.0, float(height)))) * 0.6

    face_region = image[y_value : y_value + height, x_value : x_value + width]
    if face_region.size > 0:
        grayscale_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        sharpness_score = min(cv2.Laplacian(grayscale_region, cv2.CV_64F).var() / 120.0, 2.5)
        contrast_score = min(float(grayscale_region.std()) / 35.0, 1.5)
    else:
        sharpness_score = 0.0
        contrast_score = 0.0

    face_bbox = _face_tuple_to_bbox(face)
    portrait_bbox = _expand_face_to_portrait_bbox(face_bbox, image_width, image_height)
    text_overlap_penalty = _compute_text_overlap_penalty(portrait_bbox, overlay)

    return (
        size_score
        + side_bonus
        + vertical_bonus
        + sharpness_score
        + contrast_score
        - aspect_penalty
        - text_overlap_penalty
    )


def _face_tuple_to_bbox(face: tuple[int, int, int, int]) -> dict[str, int]:
    x_value, y_value, width, height = face
    return {
        "left": x_value,
        "top": y_value,
        "right": x_value + width,
        "bottom": y_value + height,
        "width": width,
        "height": height,
    }


def _expand_face_to_portrait_bbox(
    face_bbox: dict[str, int],
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    width = int(face_bbox["width"])
    height = int(face_bbox["height"])

    left = max(0, int(round(face_bbox["left"] - (width * 0.55))))
    top = max(0, int(round(face_bbox["top"] - (height * 0.45))))
    right = min(image_width, int(round(face_bbox["right"] + (width * 0.55))))
    bottom = min(image_height, int(round(face_bbox["bottom"] + (height * 0.95))))

    if right <= left:
        right = min(image_width, left + width)
    if bottom <= top:
        bottom = min(image_height, top + height)

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def _save_portrait_crop(image: Any, portrait_bbox: dict[str, int], source_image_path: Path) -> Path:
    crop = image[
        int(portrait_bbox["top"]) : int(portrait_bbox["bottom"]),
        int(portrait_bbox["left"]) : int(portrait_bbox["right"]),
    ]
    if crop.size == 0:
        raise RuntimeError(f"Calculated portrait crop is empty for image: {source_image_path}")

    output_dir = get_passport_portrait_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _build_portrait_output_name(source_image_path)

    suffix = output_path.suffix.lower()
    write_params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        write_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    if not cv2.imwrite(str(output_path), crop, write_params):
        raise RuntimeError(f"Could not write portrait crop: {output_path}")

    return output_path


def _build_portrait_output_name(source_image_path: Path) -> str:
    source_stat = source_image_path.stat()
    source_key = (
        str(source_image_path.resolve()).lower()
        + "|"
        + str(source_stat.st_mtime_ns)
        + "|"
        + str(source_stat.st_size)
    )
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
    suffix = source_image_path.suffix.lower() or ".jpg"
    return f"{source_image_path.stem}_portrait_{digest}{suffix}"


def _empty_bbox() -> dict[str, int]:
    return {
        "left": 0,
        "top": 0,
        "right": 0,
        "bottom": 0,
        "width": 0,
        "height": 0,
    }


def _compute_text_overlap_penalty(
    portrait_bbox: dict[str, int],
    overlay: dict[str, Any] | None,
) -> float:
    if not overlay:
        return 0.0

    words = overlay.get("words", [])
    if not isinstance(words, list) or not words:
        return 0.0

    intersecting_words = 0
    intersecting_word_area = 0.0
    portrait_area = max(1.0, float(portrait_bbox["width"] * portrait_bbox["height"]))

    for word in words:
        word_bbox = word.get("bbox")
        if not isinstance(word_bbox, dict):
            continue

        intersection_area = _intersection_area(portrait_bbox, word_bbox)
        if intersection_area <= 0:
            continue

        intersecting_words += 1
        intersecting_word_area += intersection_area

    area_ratio = intersecting_word_area / portrait_area
    return min((intersecting_words * 0.55) + (area_ratio * 18.0), 7.5)


def _intersection_area(left_bbox: dict[str, int], right_bbox: dict[str, Any]) -> float:
    left = max(int(left_bbox.get("left") or 0), int(right_bbox.get("left") or 0))
    top = max(int(left_bbox.get("top") or 0), int(right_bbox.get("top") or 0))
    right = min(int(left_bbox.get("right") or 0), int(right_bbox.get("right") or 0))
    bottom = min(int(left_bbox.get("bottom") or 0), int(right_bbox.get("bottom") or 0))
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))
