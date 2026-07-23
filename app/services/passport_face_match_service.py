from __future__ import annotations

import base64
import hashlib
import tempfile
import threading
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import (
    get_face_match_detector_model_path,
    get_face_match_input_height,
    get_face_match_input_width,
    get_face_match_match_threshold,
    get_face_match_model_dir,
    get_face_match_prefer_cuda,
    get_face_match_recognizer_model_path,
    get_face_match_review_threshold,
)
from app.services.passport_inference_service import decode_base64_image_payload


YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)

_FACE_MATCH_RUNTIME: dict[str, Any] | None = None
_FACE_MATCH_RUNTIME_LOCK = threading.Lock()


def _load_cv_runtime() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Face match dependencies are missing. Install OpenCV and NumPy in the backend environment."
        ) from exc

    return cv2, np


def _download_model(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(output_path.parent), suffix=".tmp") as temp_file:
            temp_file.write(response.read())
            temp_path = Path(temp_file.name)

    temp_path.replace(output_path)


def _ensure_model_file(model_path: Path, download_url: str) -> Path:
    if model_path.exists():
        return model_path

    model_dir = get_face_match_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    _download_model(download_url, model_path)
    return model_path


def _resolve_backend_target(cv2: Any, *, prefer_cuda: bool) -> tuple[int, int, str, str]:
    if prefer_cuda and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        return (
            cv2.dnn.DNN_BACKEND_CUDA,
            cv2.dnn.DNN_TARGET_CUDA_FP16,
            "cuda",
            "cuda_fp16",
        )

    return (
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
        "cpu",
        "cpu",
    )


def _create_runtime(*, prefer_cuda: bool) -> dict[str, Any]:
    cv2, np = _load_cv_runtime()
    detector_model_path = _ensure_model_file(get_face_match_detector_model_path(), YUNET_MODEL_URL)
    recognizer_model_path = _ensure_model_file(get_face_match_recognizer_model_path(), SFACE_MODEL_URL)
    backend_id, target_id, device, target = _resolve_backend_target(cv2, prefer_cuda=prefer_cuda)

    detector = cv2.FaceDetectorYN_create(
        str(detector_model_path),
        "",
        (get_face_match_input_width(), get_face_match_input_height()),
        0.7,
        0.3,
        5000,
        backend_id,
        target_id,
    )
    recognizer = cv2.FaceRecognizerSF_create(
        str(recognizer_model_path),
        "",
        backend_id,
        target_id,
    )

    return {
        "cv2": cv2,
        "np": np,
        "detector": detector,
        "recognizer": recognizer,
        "device": device,
        "target": target,
        "backend_id": backend_id,
        "target_id": target_id,
        "detector_model_path": str(detector_model_path),
        "recognizer_model_path": str(recognizer_model_path),
        "input_width": get_face_match_input_width(),
        "input_height": get_face_match_input_height(),
    }


def _get_runtime() -> dict[str, Any]:
    global _FACE_MATCH_RUNTIME

    if _FACE_MATCH_RUNTIME is not None:
        return _FACE_MATCH_RUNTIME

    with _FACE_MATCH_RUNTIME_LOCK:
        if _FACE_MATCH_RUNTIME is not None:
            return _FACE_MATCH_RUNTIME

        prefer_cuda = get_face_match_prefer_cuda()
        try:
            runtime = _create_runtime(prefer_cuda=prefer_cuda)
        except Exception:
            if not prefer_cuda:
                raise
            runtime = _create_runtime(prefer_cuda=False)

        _FACE_MATCH_RUNTIME = runtime

    return _FACE_MATCH_RUNTIME


def preload_passport_face_match_runtime() -> dict[str, Any]:
    return _get_runtime()


def _decode_image(file_bytes: bytes) -> Any:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    np = runtime["np"]
    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode face image.")
    return image


def _to_face_row(face: Any) -> Any:
    return face.reshape(1, -1)


def _select_primary_face(faces: Any) -> Any | None:
    if faces is None or len(faces) == 0:
        return None

    def sort_key(face_row: Any) -> float:
        width = max(0.0, float(face_row[2]))
        height = max(0.0, float(face_row[3]))
        score = float(face_row[14]) if len(face_row) > 14 else 0.0
        return score * max(1.0, width * height)

    return max(faces, key=sort_key)


def _detect_primary_face(image: Any) -> tuple[Any, dict[str, Any]]:
    runtime = _get_runtime()
    detector = runtime["detector"]

    image_height, image_width = image.shape[:2]
    detector.setInputSize((int(image_width), int(image_height)))

    started = perf_counter()
    _, faces = detector.detect(image)
    duration_ms = round((perf_counter() - started) * 1000, 2)

    primary_face = _select_primary_face(faces)
    return primary_face, {
        "count": int(len(faces) if faces is not None else 0),
        "duration_ms": duration_ms,
    }


def _face_bbox(face_row: Any) -> dict[str, float]:
    return {
        "left": round(float(face_row[0]), 2),
        "top": round(float(face_row[1]), 2),
        "width": round(float(face_row[2]), 2),
        "height": round(float(face_row[3]), 2),
    }


def _face_confidence(face_row: Any) -> float:
    if len(face_row) <= 14:
        return 0.0
    return round(float(face_row[14]), 6)


def _encode_image_to_base64(image: Any) -> tuple[str, str]:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        return "", ""
    return "image/jpeg", base64.b64encode(encoded_image.tobytes()).decode("ascii")


def _extract_face_embedding(image: Any, face_row: Any) -> tuple[Any, dict[str, Any]]:
    runtime = _get_runtime()
    recognizer = runtime["recognizer"]

    started = perf_counter()
    aligned_face = recognizer.alignCrop(image, _to_face_row(face_row))
    feature = recognizer.feature(aligned_face)
    duration_ms = round((perf_counter() - started) * 1000, 2)
    content_type, aligned_base64 = _encode_image_to_base64(aligned_face)

    return feature, {
        "aligned_face_content_type": content_type,
        "aligned_face_base64": aligned_base64,
        "align_and_embed_duration_ms": duration_ms,
        "embedding_norm": round(float((feature ** 2).sum() ** 0.5), 6),
    }


def _match_features(left_feature: Any, right_feature: Any) -> float:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    recognizer = runtime["recognizer"]
    return float(recognizer.match(left_feature, right_feature, cv2.FaceRecognizerSF_FR_COSINE))


def _build_image_summary(*, file_name: str, content_type: str, base64_value: str) -> dict[str, str]:
    return {
        "file_name": file_name,
        "content_type": content_type,
        "base64": base64_value,
    }


def _build_decision(score: float) -> tuple[str, bool, bool]:
    match_threshold = get_face_match_match_threshold()
    review_threshold = get_face_match_review_threshold()

    if score >= match_threshold:
        return "match", True, False
    if score >= review_threshold:
        return "review", False, True
    return "mismatch", False, False


def verify_passport_face_match(
    *,
    passport_face_base64: str,
    passport_face_file_name: str,
    uploaded_face_base64: str,
    uploaded_face_file_name: str,
) -> dict[str, Any]:
    runtime = _get_runtime()

    total_started = perf_counter()
    passport_bytes, resolved_passport_file_name = decode_base64_image_payload(
        passport_face_base64,
        passport_face_file_name,
    )
    uploaded_bytes, resolved_uploaded_file_name = decode_base64_image_payload(
        uploaded_face_base64,
        uploaded_face_file_name,
    )

    passport_image = _decode_image(passport_bytes)
    uploaded_image = _decode_image(uploaded_bytes)

    passport_face_row, passport_detect_meta = _detect_primary_face(passport_image)
    uploaded_face_row, uploaded_detect_meta = _detect_primary_face(uploaded_image)

    passport_content_type, passport_input_base64 = _encode_image_to_base64(passport_image)
    uploaded_content_type, uploaded_input_base64 = _encode_image_to_base64(uploaded_image)

    response: dict[str, Any] = {
        "matched": False,
        "review_required": False,
        "decision": "review",
        "score": 0.0,
        "message": "",
        "thresholds": {
            "match": get_face_match_match_threshold(),
            "review": get_face_match_review_threshold(),
            "metric": "cosine_similarity",
        },
        "engine": {
            "detector": "YuNet",
            "recognizer": "SFace",
            "device": runtime["device"],
            "target": runtime["target"],
            "detector_model_path": runtime["detector_model_path"],
            "recognizer_model_path": runtime["recognizer_model_path"],
            "input_width": runtime["input_width"],
            "input_height": runtime["input_height"],
        },
        "passport_face": {
            **_build_image_summary(
                file_name=resolved_passport_file_name,
                content_type=passport_content_type,
                base64_value=passport_input_base64,
            ),
            "detected": passport_face_row is not None,
            "face_count": passport_detect_meta["count"],
            "face_bbox": _face_bbox(passport_face_row) if passport_face_row is not None else None,
            "face_confidence": _face_confidence(passport_face_row) if passport_face_row is not None else 0.0,
            "aligned_face_content_type": "",
            "aligned_face_base64": "",
        },
        "uploaded_face": {
            **_build_image_summary(
                file_name=resolved_uploaded_file_name,
                content_type=uploaded_content_type,
                base64_value=uploaded_input_base64,
            ),
            "detected": uploaded_face_row is not None,
            "face_count": uploaded_detect_meta["count"],
            "face_bbox": _face_bbox(uploaded_face_row) if uploaded_face_row is not None else None,
            "face_confidence": _face_confidence(uploaded_face_row) if uploaded_face_row is not None else 0.0,
            "aligned_face_content_type": "",
            "aligned_face_base64": "",
        },
        "performance": {
            "passport_detect_duration_ms": passport_detect_meta["duration_ms"],
            "uploaded_detect_duration_ms": uploaded_detect_meta["duration_ms"],
            "passport_align_embed_duration_ms": 0.0,
            "uploaded_align_embed_duration_ms": 0.0,
            "match_duration_ms": 0.0,
            "total_duration_ms": 0.0,
        },
        "request_hash": hashlib.sha256(passport_bytes + b"::" + uploaded_bytes).hexdigest(),
    }

    if passport_face_row is None and uploaded_face_row is None:
        response["message"] = "Khong detect duoc mat tren ca 2 anh."
    elif passport_face_row is None:
        response["message"] = "Khong detect duoc mat tren anh chan dung cat tu passport."
    elif uploaded_face_row is None:
        response["message"] = "Khong detect duoc mat tren anh mat vua upload."
    else:
        passport_feature, passport_feature_meta = _extract_face_embedding(passport_image, passport_face_row)
        uploaded_feature, uploaded_feature_meta = _extract_face_embedding(uploaded_image, uploaded_face_row)
        response["passport_face"]["aligned_face_content_type"] = passport_feature_meta["aligned_face_content_type"]
        response["passport_face"]["aligned_face_base64"] = passport_feature_meta["aligned_face_base64"]
        response["uploaded_face"]["aligned_face_content_type"] = uploaded_feature_meta["aligned_face_content_type"]
        response["uploaded_face"]["aligned_face_base64"] = uploaded_feature_meta["aligned_face_base64"]
        response["performance"]["passport_align_embed_duration_ms"] = passport_feature_meta["align_and_embed_duration_ms"]
        response["performance"]["uploaded_align_embed_duration_ms"] = uploaded_feature_meta["align_and_embed_duration_ms"]

        match_started = perf_counter()
        score = _match_features(passport_feature, uploaded_feature)
        response["performance"]["match_duration_ms"] = round((perf_counter() - match_started) * 1000, 2)
        response["score"] = round(score, 6)
        decision, matched, review_required = _build_decision(score)
        response["decision"] = decision
        response["matched"] = matched
        response["review_required"] = review_required
        response["message"] = {
            "match": "Anh mat khop nhau.",
            "review": "Anh mat gan giong, nen review them truoc khi chap nhan.",
            "mismatch": "Anh mat khong khop nhau.",
        }[decision]

    response["performance"]["total_duration_ms"] = round((perf_counter() - total_started) * 1000, 2)
    return response
