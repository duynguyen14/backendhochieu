from __future__ import annotations

import base64
import hashlib
import math
import tempfile
import threading
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import (
    get_face_match_detector_model_path,
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
_YUNET_STRIDES = (8, 16, 32)
_YUNET_SCORE_THRESHOLD = 0.7
_YUNET_NMS_THRESHOLD = 0.3
_YUNET_TOP_K = 5000
_SFACE_LANDMARK_TEMPLATE = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Face match dependencies are missing. Install OpenCV, NumPy, and onnxruntime-gpu."
        ) from exc

    return cv2, np, ort


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


def _build_session_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _select_ort_providers(ort: Any, *, prefer_cuda: bool) -> tuple[list[Any], str, str]:
    available_providers = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda", "CUDAExecutionProvider"

    return ["CPUExecutionProvider"], "cpu", "CPUExecutionProvider"


def _resolve_model_input_size(session: Any, *, fallback: tuple[int, int]) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    try:
        height = int(shape[2])
        width = int(shape[3])
    except (TypeError, ValueError, IndexError):
        width, height = fallback

    return width, height


def _create_runtime(*, prefer_cuda: bool) -> dict[str, Any]:
    cv2, np, ort = _load_runtime_dependencies()
    detector_model_path = _ensure_model_file(get_face_match_detector_model_path(), YUNET_MODEL_URL)
    recognizer_model_path = _ensure_model_file(get_face_match_recognizer_model_path(), SFACE_MODEL_URL)
    providers, device, target = _select_ort_providers(ort, prefer_cuda=prefer_cuda)
    session_options = _build_session_options(ort)

    detector_session = ort.InferenceSession(
        str(detector_model_path),
        sess_options=session_options,
        providers=providers,
    )
    recognizer_session = ort.InferenceSession(
        str(recognizer_model_path),
        sess_options=session_options,
        providers=providers,
    )

    detector_input_name = detector_session.get_inputs()[0].name
    recognizer_input_name = recognizer_session.get_inputs()[0].name
    detector_output_names = [output.name for output in detector_session.get_outputs()]
    recognizer_output_name = recognizer_session.get_outputs()[0].name
    detector_input_width, detector_input_height = _resolve_model_input_size(
        detector_session,
        fallback=(640, 640),
    )

    return {
        "cv2": cv2,
        "np": np,
        "ort": ort,
        "detector_session": detector_session,
        "recognizer_session": recognizer_session,
        "detector_input_name": detector_input_name,
        "recognizer_input_name": recognizer_input_name,
        "detector_output_names": detector_output_names,
        "recognizer_output_name": recognizer_output_name,
        "device": device,
        "target": target,
        "available_providers": ort.get_available_providers(),
        "active_detector_providers": detector_session.get_providers(),
        "active_recognizer_providers": recognizer_session.get_providers(),
        "detector_model_path": str(detector_model_path),
        "recognizer_model_path": str(recognizer_model_path),
        "input_width": detector_input_width,
        "input_height": detector_input_height,
    }


def _get_runtime() -> dict[str, Any]:
    global _FACE_MATCH_RUNTIME

    if _FACE_MATCH_RUNTIME is not None:
        return _FACE_MATCH_RUNTIME

    with _FACE_MATCH_RUNTIME_LOCK:
        if _FACE_MATCH_RUNTIME is not None:
            return _FACE_MATCH_RUNTIME

        prefer_cuda = get_face_match_prefer_cuda()
        _FACE_MATCH_RUNTIME = _create_runtime(prefer_cuda=prefer_cuda)

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


def _prepare_yunet_input(image: Any) -> tuple[Any, float, float]:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    np = runtime["np"]
    input_width = int(runtime["input_width"])
    input_height = int(runtime["input_height"])
    image_height, image_width = image.shape[:2]
    resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    blob = resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    scale_x = image_width / float(input_width)
    scale_y = image_height / float(input_height)
    return blob, scale_x, scale_y


def _run_yunet(image: Any) -> list[Any]:
    runtime = _get_runtime()
    blob, scale_x, scale_y = _prepare_yunet_input(image)
    outputs = runtime["detector_session"].run(
        runtime["detector_output_names"],
        {runtime["detector_input_name"]: blob},
    )
    return _decode_yunet_outputs(outputs, scale_x=scale_x, scale_y=scale_y)


def _decode_yunet_outputs(outputs: list[Any], *, scale_x: float, scale_y: float) -> list[Any]:
    runtime = _get_runtime()
    np = runtime["np"]
    input_width = int(runtime["input_width"])
    input_height = int(runtime["input_height"])
    faces: list[Any] = []

    for stride_index, stride in enumerate(_YUNET_STRIDES):
        cls = outputs[stride_index].reshape(-1)
        obj = outputs[stride_index + len(_YUNET_STRIDES)].reshape(-1)
        bbox = outputs[stride_index + len(_YUNET_STRIDES) * 2].reshape(-1, 4)
        kps = outputs[stride_index + len(_YUNET_STRIDES) * 3].reshape(-1, 10)
        cols = int(input_width / stride)
        rows = int(input_height / stride)

        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                cls_score = min(1.0, max(0.0, float(cls[index])))
                obj_score = min(1.0, max(0.0, float(obj[index])))
                score = math.sqrt(cls_score * obj_score)
                if score < _YUNET_SCORE_THRESHOLD:
                    continue

                center_x = (col + float(bbox[index, 0])) * stride
                center_y = (row + float(bbox[index, 1])) * stride
                width = math.exp(float(bbox[index, 2])) * stride
                height = math.exp(float(bbox[index, 3])) * stride
                left = (center_x - width / 2.0) * scale_x
                top = (center_y - height / 2.0) * scale_y
                width *= scale_x
                height *= scale_y

                face = [
                    left,
                    top,
                    width,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    score,
                ]
                for landmark_index in range(5):
                    face[4 + 2 * landmark_index] = (
                        float(kps[index, 2 * landmark_index]) + col
                    ) * stride * scale_x
                    face[4 + 2 * landmark_index + 1] = (
                        float(kps[index, 2 * landmark_index + 1]) + row
                    ) * stride * scale_y
                faces.append(face)

    if not faces:
        return []

    face_array = np.asarray(faces, dtype=np.float32)
    keep_indices = _nms_faces(face_array)
    return [face_array[index] for index in keep_indices]


def _nms_faces(faces: Any) -> list[int]:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    boxes = [
        [int(face[0]), int(face[1]), max(1, int(face[2])), max(1, int(face[3]))]
        for face in faces
    ]
    scores = [float(face[14]) for face in faces]
    keep = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        _YUNET_SCORE_THRESHOLD,
        _YUNET_NMS_THRESHOLD,
        eta=1.0,
        top_k=_YUNET_TOP_K,
    )
    if keep is None or len(keep) == 0:
        return []

    return [int(index) for index in keep.flatten()]


def _select_primary_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None

    def sort_key(face_row: Any) -> float:
        width = max(0.0, float(face_row[2]))
        height = max(0.0, float(face_row[3]))
        score = float(face_row[14]) if len(face_row) > 14 else 0.0
        return score * max(1.0, width * height)

    return max(faces, key=sort_key)


def _detect_primary_face(image: Any) -> tuple[Any, dict[str, Any]]:
    started = perf_counter()
    faces = _run_yunet(image)
    duration_ms = round((perf_counter() - started) * 1000, 2)
    primary_face = _select_primary_face(faces)
    return primary_face, {
        "count": int(len(faces)),
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


def _align_face(image: Any, face_row: Any) -> Any:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    np = runtime["np"]
    source_landmarks = np.asarray(
        [[float(face_row[4 + 2 * index]), float(face_row[5 + 2 * index])] for index in range(5)],
        dtype=np.float32,
    )
    target_landmarks = np.asarray(_SFACE_LANDMARK_TEMPLATE, dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks, method=cv2.LMEDS)
    if transform is None:
        transform = cv2.getAffineTransform(source_landmarks[:3], target_landmarks[:3])
    return cv2.warpAffine(image, transform, (112, 112), flags=cv2.INTER_LINEAR)


def _prepare_sface_input(aligned_face: Any) -> Any:
    runtime = _get_runtime()
    cv2 = runtime["cv2"]
    np = runtime["np"]
    resized = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)


def _extract_face_embedding(image: Any, face_row: Any) -> tuple[Any, dict[str, Any]]:
    runtime = _get_runtime()

    started = perf_counter()
    aligned_face = _align_face(image, face_row)
    blob = _prepare_sface_input(aligned_face)
    feature = runtime["recognizer_session"].run(
        [runtime["recognizer_output_name"]],
        {runtime["recognizer_input_name"]: blob},
    )[0]
    duration_ms = round((perf_counter() - started) * 1000, 2)
    content_type, aligned_base64 = _encode_image_to_base64(aligned_face)

    return feature, {
        "aligned_face_content_type": content_type,
        "aligned_face_base64": aligned_base64,
        "align_and_embed_duration_ms": duration_ms,
        "embedding_norm": round(_feature_norm(feature), 6),
    }


def _feature_norm(feature: Any) -> float:
    runtime = _get_runtime()
    np = runtime["np"]
    return float(np.linalg.norm(feature.reshape(-1)))


def _match_features(left_feature: Any, right_feature: Any) -> float:
    runtime = _get_runtime()
    np = runtime["np"]
    left = left_feature.reshape(-1).astype(np.float32)
    right = right_feature.reshape(-1).astype(np.float32)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left / left_norm, right / right_norm))


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
            "runtime": "onnxruntime",
            "device": runtime["device"],
            "target": runtime["target"],
            "available_providers": runtime["available_providers"],
            "active_detector_providers": runtime["active_detector_providers"],
            "active_recognizer_providers": runtime["active_recognizer_providers"],
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
