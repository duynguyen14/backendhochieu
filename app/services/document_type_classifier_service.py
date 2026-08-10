from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.config import (
    get_document_type_cpu_threads,
    get_document_type_metadata_path,
    get_document_type_min_confidence,
    get_document_type_model_path,
    get_document_type_warmup_enabled,
)
from app.services.passport_inference_service import decode_base64_image_payload


DEFAULT_CLASS_NAMES = ["passport", "face", "EVISA_RESULT", "VOA_RESULT"]
_CLASSIFIER_RUNTIME: DocumentTypeClassifierRuntime | None = None
_CLASSIFIER_RUNTIME_LOCK = Lock()


class DocumentTypeClassifierRuntime:
    def __init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for document type detection.") from exc

        self.model_path = _resolve_model_path(get_document_type_model_path())
        self.metadata_path = _resolve_metadata_path(self.model_path, get_document_type_metadata_path())
        self.metadata = _read_metadata(self.metadata_path)
        self.class_names = _read_class_names(self.metadata)
        self.image_size = int(self.metadata.get("img_size") or 224)
        self.min_confidence = get_document_type_min_confidence()
        self.cpu_threads = get_document_type_cpu_threads()
        self.mean = np.asarray(
            self.metadata.get("normalize_mean") or [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        self.std = np.asarray(
            self.metadata.get("normalize_std") or [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.cpu_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        started = perf_counter()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.model_load_ms = round((perf_counter() - started) * 1000, 2)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._predict_lock = Lock()

    def warmup(self) -> None:
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.detect(image)

    def detect(self, image: np.ndarray) -> dict[str, object]:
        if image is None or image.size == 0:
            raise ValueError("Image is empty.")

        batch = _preprocess_image(
            image=image,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        with self._predict_lock:
            outputs = self.session.run([self.output_name], {self.input_name: batch})

        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[0]
        probabilities = _softmax(logits.reshape(-1))
        label_id = int(np.argmax(probabilities))
        confidence = round(float(probabilities[label_id]), 6)
        label = self.class_names[label_id] if label_id < len(self.class_names) else str(label_id)

        return {
            "labelId": label_id,
            "label": label,
            "confidence": confidence,
        }


def preload_document_type_classifier_runtime() -> None:
    runtime = get_document_type_classifier_runtime()
    if get_document_type_warmup_enabled():
        runtime.warmup()


def get_document_type_classifier_runtime() -> DocumentTypeClassifierRuntime:
    global _CLASSIFIER_RUNTIME
    if _CLASSIFIER_RUNTIME is not None:
        return _CLASSIFIER_RUNTIME
    with _CLASSIFIER_RUNTIME_LOCK:
        if _CLASSIFIER_RUNTIME is None:
            _CLASSIFIER_RUNTIME = DocumentTypeClassifierRuntime()
        return _CLASSIFIER_RUNTIME


def detect_document_type_from_base64(
    base64_payload: str,
    file_name: str | None = None,
) -> dict[str, object]:
    file_bytes, _ = decode_base64_image_payload(base64_payload, file_name)
    image = _decode_image_bytes(file_bytes)
    return get_document_type_classifier_runtime().detect(image)


def _resolve_model_path(configured_path: Path) -> Path:
    path = configured_path.expanduser().resolve()
    if path.is_dir():
        path = path / "model.onnx"
    if not path.is_file():
        raise FileNotFoundError(f"DOCUMENT_TYPE_MODEL_PATH does not exist: {path}")
    return path


def _resolve_metadata_path(model_path: Path, configured_path: Path | None) -> Path | None:
    if configured_path is not None:
        path = configured_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DOCUMENT_TYPE_METADATA_PATH does not exist: {path}")
        return path
    candidate = model_path.parent / "metadata.json"
    return candidate if candidate.is_file() else None


def _read_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_class_names(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("class_names")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    labels = metadata.get("labels")
    if isinstance(labels, dict) and labels:
        return [str(name) for name, _ in sorted(labels.items(), key=lambda item: int(item[1]))]
    return DEFAULT_CLASS_NAMES


def _decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image payload.")
    return image


def _preprocess_image(
    *,
    image: np.ndarray,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
