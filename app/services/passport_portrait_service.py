from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2

from app.config import get_passport_portrait_output_dir
from app.services.ocr_service import run_ocr_with_boxes


_FACE_CASCADE: Any | None = None


def detect_passport_portrait(
    image_path: Path,
    overlay: dict[str, Any] | None = None,
    *,
    use_ocr_fallback: bool = True,
) -> dict[str, Any]:
    source_path = image_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Image file not found: {source_path}")

    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"Could not read image with OpenCV: {source_path}")

    image_height, image_width = image.shape[:2]
    if overlay is None and use_ocr_fallback:
        try:
            overlay = run_ocr_with_boxes(source_path, auto_rotate=False, fast_mode=True)
        except Exception:
            overlay = None
    best_result = _detect_best_portrait_candidate(
        image,
        overlay=overlay,
        use_multi_orientation=overlay is None,
    )

    if best_result is None:
        return {
            "detected": False,
            "image_path": str(source_path),
            "image_width": image_width,
            "image_height": image_height,
            "face_bbox": _empty_bbox(),
            "portrait_bbox": _empty_bbox(),
            "portrait_image_path": "",
        }

    face_bbox = best_result["face_bbox"]
    portrait_bbox = best_result["portrait_bbox"]
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


def _detect_best_portrait_candidate(
    image: Any,
    *,
    overlay: dict[str, Any] | None,
    use_multi_orientation: bool,
) -> dict[str, Any] | None:
    original_height, original_width = image.shape[:2]
    if not use_multi_orientation:
        orientation_angles = (0,)
    elif original_width >= original_height:
        orientation_angles = (0,)
    else:
        orientation_angles = (90, 270, 0, 180)

    best_result: dict[str, Any] | None = None
    best_score = float("-inf")

    for angle in orientation_angles:
        rotated_image = _rotate_image_by_angle(image, angle)
        rotated_height, rotated_width = rotated_image.shape[:2]
        rotated_overlay = overlay if angle == 0 else None
        document_bbox = _infer_document_bbox(rotated_width, rotated_height, rotated_overlay)
        face_candidates = _detect_face_candidates(rotated_image, document_bbox)
        best_face, best_face_score = _select_best_face(
            face_candidates,
            rotated_image,
            rotated_overlay,
            document_bbox,
        )
        if best_face is None:
            continue

        rotated_face_bbox = _face_tuple_to_bbox(best_face)
        rotated_portrait_bbox = _expand_face_to_portrait_bbox(
            rotated_face_bbox,
            rotated_width,
            rotated_height,
        )
        face_bbox = _map_bbox_to_original_orientation(
            rotated_face_bbox,
            angle,
            original_width,
            original_height,
        )
        portrait_bbox = _map_bbox_to_original_orientation(
            rotated_portrait_bbox,
            angle,
            original_width,
            original_height,
        )

        orientation_bonus = 0.18 if angle == 0 and overlay is not None else 0.0
        total_score = best_face_score + orientation_bonus
        if total_score <= best_score:
            continue

        best_score = total_score
        best_result = {
            "angle": angle,
            "score": total_score,
            "face_bbox": face_bbox,
            "portrait_bbox": portrait_bbox,
        }

    return best_result


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


def preload_passport_portrait_runtime() -> None:
    _get_face_cascade()


def _detect_face_candidates(
    image: Any,
    document_bbox: dict[str, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    cascade = _get_face_cascade()

    min_face_width = max(36, int(image.shape[1] * 0.08))
    min_face_height = max(36, int(image.shape[0] * 0.12))

    detected_faces: list[tuple[int, int, int, int]] = _detect_faces_in_region(
        cascade,
        grayscale,
        offset_x=0,
        offset_y=0,
        min_face_width=min_face_width,
        min_face_height=min_face_height,
    )

    if document_bbox is not None:
        for portrait_region in _build_passport_portrait_regions(document_bbox):
            region_left = int(portrait_region["left"])
            region_top = int(portrait_region["top"])
            region_right = int(portrait_region["right"])
            region_bottom = int(portrait_region["bottom"])
            if region_right <= region_left or region_bottom <= region_top:
                continue

            region_gray = grayscale[region_top:region_bottom, region_left:region_right]
            if region_gray.size == 0:
                continue

            region_min_face_width = max(30, int((region_right - region_left) * 0.22))
            region_min_face_height = max(36, int((region_bottom - region_top) * 0.22))
            detected_faces.extend(
                _detect_faces_in_region(
                    cascade,
                    region_gray,
                    offset_x=region_left,
                    offset_y=region_top,
                    min_face_width=region_min_face_width,
                    min_face_height=region_min_face_height,
                )
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
    document_bbox: dict[str, int] | None,
) -> tuple[tuple[int, int, int, int] | None, float]:
    if not face_candidates:
        return None, float("-inf")

    filtered_candidates = [
        face for face in face_candidates
        if _is_candidate_inside_passport_region(face, document_bbox)
    ]
    candidate_pool = filtered_candidates or face_candidates

    sorted_by_area = sorted(
        candidate_pool,
        key=lambda face: int(face[2]) * int(face[3]),
        reverse=True,
    )
    if len(sorted_by_area) == 1:
        sole_face = sorted_by_area[0]
        return (
            sole_face,
            _score_face_candidate(
                sole_face,
                image.shape[1],
                image.shape[0],
                image,
                overlay,
                document_bbox,
                _infer_preferred_portrait_side(document_bbox, overlay),
            ),
        )

    largest_area = int(sorted_by_area[0][2]) * int(sorted_by_area[0][3])
    second_area = int(sorted_by_area[1][2]) * int(sorted_by_area[1][3])
    if document_bbox is None:
        competitive_faces = [
            face for face in sorted_by_area
            if (int(face[2]) * int(face[3])) >= int(largest_area * 0.9)
        ]
        best_face = max(
            competitive_faces,
            key=lambda face: _score_face_candidate(
                face,
                image.shape[1],
                image.shape[0],
                image,
                overlay,
                None,
                None,
            ),
        )
        best_score = _score_face_candidate(
            best_face,
            image.shape[1],
            image.shape[0],
            image,
            overlay,
            None,
            None,
        ) + 0.4
        return best_face, best_score

    image_height, image_width = image.shape[:2]
    best_face: tuple[int, int, int, int] | None = None
    best_score = float("-inf")

    preferred_side = _infer_preferred_portrait_side(document_bbox, overlay)
    for face in candidate_pool:
        score = _score_face_candidate(
            face,
            image_width,
            image_height,
            image,
            overlay,
            document_bbox,
            preferred_side,
        )
        if score > best_score:
            best_score = score
            best_face = face

    if largest_area >= int(second_area * 1.12) and sorted_by_area[0] == best_face:
        best_score += 0.12

    return best_face, best_score


def _score_face_candidate(
    face: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    image: Any,
    overlay: dict[str, Any] | None,
    document_bbox: dict[str, int] | None,
    preferred_side: str | None,
) -> float:
    x_value, y_value, width, height = face
    area_ratio = (width * height) / max(1.0, float(image_width * image_height))
    center_x_ratio = (x_value + (width / 2.0)) / max(1.0, float(image_width))
    center_y_ratio = (y_value + (height / 2.0)) / max(1.0, float(image_height))

    size_score = min(area_ratio * 28.0, 6.0)
    side_bonus = 1.4 if center_x_ratio <= 0.42 or center_x_ratio >= 0.58 else -1.5
    vertical_bonus = 1.0 if 0.18 <= center_y_ratio <= 0.72 else -0.8
    aspect_penalty = abs(1.0 - (width / max(1.0, float(height)))) * 0.6

    document_layout_bonus = 0.0
    document_bounds_penalty = 0.0
    preferred_side_bonus = 0.0
    if document_bbox is not None:
        document_layout_bonus, document_bounds_penalty = _score_passport_document_layout(
            face,
            document_bbox,
        )
        if preferred_side:
            document_center_x_ratio = (
                (x_value + (width / 2.0)) - document_bbox["left"]
            ) / max(1.0, float(document_bbox["width"]))
            candidate_side = "left" if document_center_x_ratio <= 0.5 else "right"
            preferred_side_bonus = 2.6 if candidate_side == preferred_side else -0.9

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
        + document_layout_bonus
        + preferred_side_bonus
        + sharpness_score
        + contrast_score
        - aspect_penalty
        - document_bounds_penalty
        - text_overlap_penalty
    )


def _detect_faces_in_region(
    cascade: Any,
    grayscale_region: Any,
    offset_x: int,
    offset_y: int,
    min_face_width: int,
    min_face_height: int,
) -> list[tuple[int, int, int, int]]:
    detected_faces: list[tuple[int, int, int, int]] = []
    for scale_factor, min_neighbors in ((1.08, 5), (1.05, 4), (1.12, 6)):
        faces = cascade.detectMultiScale(
            grayscale_region,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_width, min_face_height),
        )
        detected_faces.extend(
            (
                int(x) + offset_x,
                int(y) + offset_y,
                int(width),
                int(height),
            )
            for (x, y, width, height) in faces
        )
    return detected_faces


def _infer_document_bbox(
    image_width: int,
    image_height: int,
    overlay: dict[str, Any] | None,
) -> dict[str, int] | None:
    if not overlay:
        return None

    words = overlay.get("words", [])
    if not isinstance(words, list) or not words:
        return None

    left_values: list[float] = []
    top_values: list[float] = []
    right_values: list[float] = []
    bottom_values: list[float] = []
    for word in words:
        bbox = word.get("bbox")
        if not isinstance(bbox, dict):
            continue
        left = float(bbox.get("left") or 0)
        top = float(bbox.get("top") or 0)
        width = float(bbox.get("width") or 0)
        height = float(bbox.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        left_values.append(left)
        top_values.append(top)
        right_values.append(left + width)
        bottom_values.append(top + height)

    if not left_values:
        return None

    word_left = min(left_values)
    word_top = min(top_values)
    word_right = max(right_values)
    word_bottom = max(bottom_values)
    word_width = max(1.0, word_right - word_left)
    word_height = max(1.0, word_bottom - word_top)

    left = max(0, int(round(word_left - (word_width * 0.12))))
    top = max(0, int(round(word_top - (word_height * 0.1))))
    right = min(image_width, int(round(word_right + (word_width * 0.08))))
    bottom = min(image_height, int(round(word_bottom + (word_height * 0.12))))

    if right <= left or bottom <= top:
        return None

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _build_passport_portrait_regions(document_bbox: dict[str, int]) -> list[dict[str, int]]:
    doc_left = int(document_bbox["left"])
    doc_top = int(document_bbox["top"])
    doc_width = int(document_bbox["width"])
    doc_height = int(document_bbox["height"])
    doc_right = int(document_bbox["right"])
    doc_bottom = int(document_bbox["bottom"])

    region_top = doc_top + int(round(doc_height * 0.2))
    region_bottom = doc_top + int(round(doc_height * 0.82))
    left_region_right = doc_left + int(round(doc_width * 0.36))
    right_region_left = doc_right - int(round(doc_width * 0.36))

    return [
        _build_bbox(
            doc_left,
            region_top,
            left_region_right,
            min(doc_bottom, region_bottom),
        ),
        _build_bbox(
            max(doc_left, right_region_left),
            region_top,
            doc_right,
            min(doc_bottom, region_bottom),
        ),
    ]


def _is_candidate_inside_passport_region(
    face: tuple[int, int, int, int],
    document_bbox: dict[str, int] | None,
) -> bool:
    if document_bbox is None:
        return True

    x_value, y_value, width, height = face
    center_x = x_value + (width / 2.0)
    center_y = y_value + (height / 2.0)

    if not (
        document_bbox["left"] <= center_x <= document_bbox["right"]
        and document_bbox["top"] <= center_y <= document_bbox["bottom"]
    ):
        return False

    relative_x = (center_x - document_bbox["left"]) / max(1.0, float(document_bbox["width"]))
    relative_y = (center_y - document_bbox["top"]) / max(1.0, float(document_bbox["height"]))
    return (
        0.04 <= relative_x <= 0.96
        and 0.22 <= relative_y <= 0.9
        and (relative_x <= 0.42 or relative_x >= 0.58)
    )


def _score_passport_document_layout(
    face: tuple[int, int, int, int],
    document_bbox: dict[str, int],
) -> tuple[float, float]:
    x_value, y_value, width, height = face
    center_x = x_value + (width / 2.0)
    center_y = y_value + (height / 2.0)
    relative_x = (center_x - document_bbox["left"]) / max(1.0, float(document_bbox["width"]))
    relative_y = (center_y - document_bbox["top"]) / max(1.0, float(document_bbox["height"]))
    relative_area = (width * height) / max(1.0, float(document_bbox["width"] * document_bbox["height"]))

    bonus = 0.0
    penalty = 0.0

    if 0.26 <= relative_y <= 0.78:
        bonus += 3.4
    else:
        penalty += min(abs(relative_y - 0.52) * 10.0, 5.2)

    if relative_x <= 0.34 or relative_x >= 0.66:
        bonus += 2.4
    else:
        penalty += min(abs(relative_x - 0.5) * 6.0, 2.6)

    if 0.012 <= relative_area <= 0.085:
        bonus += 1.6
    else:
        penalty += min(abs(relative_area - 0.038) * 36.0, 3.4)

    return bonus, penalty


def _infer_preferred_portrait_side(
    document_bbox: dict[str, int] | None,
    overlay: dict[str, Any] | None,
) -> str | None:
    if document_bbox is None or not overlay:
        return None

    portrait_regions = _build_passport_portrait_regions(document_bbox)
    if len(portrait_regions) != 2:
        return None

    left_density = _compute_text_overlap_penalty(portrait_regions[0], overlay)
    right_density = _compute_text_overlap_penalty(portrait_regions[1], overlay)
    if abs(left_density - right_density) < 0.6:
        return None

    return "left" if left_density < right_density else "right"


def _face_tuple_to_bbox(face: tuple[int, int, int, int]) -> dict[str, int]:
    x_value, y_value, width, height = face
    return _build_bbox(
        x_value,
        y_value,
        x_value + width,
        y_value + height,
    )


def _expand_face_to_portrait_bbox(
    face_bbox: dict[str, int],
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    width = int(face_bbox["width"])
    height = int(face_bbox["height"])

    horizontal_padding = width * 0.18
    top_padding = height * 0.40
    bottom_padding = height * 0.42

    left = max(0, int(round(face_bbox["left"] - horizontal_padding)))
    top = max(0, int(round(face_bbox["top"] - top_padding)))
    right = min(image_width, int(round(face_bbox["right"] + horizontal_padding)))
    bottom = min(image_height, int(round(face_bbox["bottom"] + bottom_padding)))

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


def _build_bbox(left: int, top: int, right: int, bottom: int) -> dict[str, int]:
    normalized_left = int(left)
    normalized_top = int(top)
    normalized_right = int(right)
    normalized_bottom = int(bottom)
    return {
        "left": normalized_left,
        "top": normalized_top,
        "right": normalized_right,
        "bottom": normalized_bottom,
        "width": max(0, normalized_right - normalized_left),
        "height": max(0, normalized_bottom - normalized_top),
    }


def _rotate_image_by_angle(image: Any, angle: int) -> Any:
    normalized_angle = angle % 360
    if normalized_angle == 0:
        return image
    if normalized_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized_angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def _map_bbox_to_original_orientation(
    rotated_bbox: dict[str, int],
    angle: int,
    original_width: int,
    original_height: int,
) -> dict[str, int]:
    normalized_angle = angle % 360
    if normalized_angle == 0:
        return dict(rotated_bbox)

    corners = [
        (float(rotated_bbox["left"]), float(rotated_bbox["top"])),
        (float(rotated_bbox["right"]), float(rotated_bbox["top"])),
        (float(rotated_bbox["right"]), float(rotated_bbox["bottom"])),
        (float(rotated_bbox["left"]), float(rotated_bbox["bottom"])),
    ]
    original_corners = [
        _map_rotated_point_to_original(point_x, point_y, normalized_angle, original_width, original_height)
        for point_x, point_y in corners
    ]

    x_values = [point[0] for point in original_corners]
    y_values = [point[1] for point in original_corners]
    return _build_bbox(
        max(0, int(round(min(x_values)))),
        max(0, int(round(min(y_values)))),
        min(original_width, int(round(max(x_values)))),
        min(original_height, int(round(max(y_values)))),
    )


def _map_rotated_point_to_original(
    x_value: float,
    y_value: float,
    angle: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float]:
    if angle == 90:
        return y_value, float(original_height) - x_value
    if angle == 180:
        return float(original_width) - x_value, float(original_height) - y_value
    if angle == 270:
        return float(original_width) - y_value, x_value
    return x_value, y_value


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
