from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageOps

from app.config import (
    get_donut_cache_size,
    get_donut_cpu_threads,
    get_donut_device,
    get_donut_inference_image_height,
    get_donut_inference_image_width,
    get_donut_max_new_tokens,
    get_donut_model_dir,
    get_donut_processor_dir,
    get_donut_task_prompt,
    get_donut_use_dynamic_quantization,
    get_inference_skip_ocr_auto_rotate,
    get_inference_upload_dir,
)
from app.services.ocr_service import build_empty_passport_json, normalize_date, run_ocr_with_boxes
from app.services.passport_review_service import PASSPORT_FIELD_KEYS


SUPPORTED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_DONUT_RUNTIME: dict[str, Any] | None = None
_DONUT_RUNTIME_LOCK = threading.Lock()
_INFERENCE_RESULT_CACHE: dict[str, dict[str, Any]] = {}
_INFERENCE_CACHE_ORDER: list[str] = []
_INFERENCE_CACHE_LOCK = threading.Lock()


def _load_transformers_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Donut inference dependencies are missing. Install `torch`, `transformers`, "
            "`sentencepiece`, and `python-multipart` in the backend virtual environment."
        ) from exc

    return torch, (DonutProcessor, VisionEncoderDecoderModel)


def _resolve_donut_device(torch: Any) -> str:
    configured_device = get_donut_device()
    if configured_device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if configured_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "DONUT_DEVICE is set to CUDA but PyTorch cannot access a CUDA GPU. "
            "Install a CUDA-enabled torch build or set DONUT_DEVICE=cpu."
        )

    return configured_device


def _get_processor_dir(model_dir: Path) -> Path:
    configured_processor_dir = get_donut_processor_dir()
    if configured_processor_dir.exists():
        return configured_processor_dir

    required_processor_files = {
        "preprocessor_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "spiece.model",
    }
    if any((model_dir / file_name).exists() for file_name in required_processor_files):
        return model_dir

    parent_dir = model_dir.parent
    if any((parent_dir / file_name).exists() for file_name in required_processor_files):
        return parent_dir

    return model_dir


def _get_donut_runtime() -> dict[str, Any]:
    global _DONUT_RUNTIME

    if _DONUT_RUNTIME is not None:
        return _DONUT_RUNTIME

    with _DONUT_RUNTIME_LOCK:
        if _DONUT_RUNTIME is not None:
            return _DONUT_RUNTIME

        torch, (DonutProcessor, VisionEncoderDecoderModel) = _load_transformers_runtime()
        model_dir = get_donut_model_dir()
        if not model_dir.exists():
            raise RuntimeError(f"Donut model directory not found: {model_dir}")

        cpu_threads = get_donut_cpu_threads()
        torch.set_num_threads(cpu_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

        device = _resolve_donut_device(torch)
        processor_dir = _get_processor_dir(model_dir)
        processor = DonutProcessor.from_pretrained(str(processor_dir), local_files_only=True)
        try:
            model = VisionEncoderDecoderModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except (TypeError, ImportError, ValueError):
            model = VisionEncoderDecoderModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )

        if device == "cpu" and get_donut_use_dynamic_quantization():
            quantize_dynamic = getattr(torch, "quantization", None)
            if quantize_dynamic is not None and hasattr(quantize_dynamic, "quantize_dynamic"):
                try:
                    model = quantize_dynamic.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                except Exception:
                    pass
        model.to(device)
        model.eval()

        _DONUT_RUNTIME = {
            "torch": torch,
            "processor": processor,
            "model": model,
            "device": device,
            "model_dir": str(model_dir),
            "processor_dir": str(processor_dir),
            "cpu_threads": cpu_threads,
        }

    return _DONUT_RUNTIME


def _normalize_file_extension(file_name: str) -> str:
    suffix = Path(file_name or "").suffix.lower()
    if suffix in SUPPORTED_UPLOAD_EXTENSIONS:
        return suffix
    return ".jpg"


def _store_uploaded_image(file_bytes: bytes, file_name: str) -> tuple[str, Path]:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    upload_dir = get_inference_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(file_bytes).hexdigest()
    extension = _normalize_file_extension(file_name)
    image_id = f"{digest}{extension}"
    image_path = upload_dir / image_id

    if not image_path.exists():
        image_path.write_bytes(file_bytes)

    return image_id, image_path


def _extract_json_substring(raw_text: str) -> dict[str, Any] | None:
    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index < 0 or end_index <= start_index:
        return None

    try:
        parsed = json.loads(raw_text[start_index : end_index + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_date_value(value: str) -> str:
    stripped = str(value or "").strip()
    if not stripped:
        return ""

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return stripped

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", stripped):
        return normalize_date(stripped)

    return stripped


def _normalize_donut_gt_parse(raw_payload: dict[str, Any]) -> dict[str, str]:
    gt_parse = raw_payload.get("gt_parse", raw_payload)
    if not isinstance(gt_parse, dict):
        gt_parse = {}

    normalized = build_empty_passport_json()
    for field_key in PASSPORT_FIELD_KEYS:
        value = gt_parse.get(field_key, "")
        normalized_value = str(value) if value is not None else ""
        if field_key in {"passport_type", "issuing_country", "sex", "nationality_current", "nationality_at_birth"}:
            normalized_value = normalized_value.upper()
        if field_key in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
            normalized_value = _normalize_date_value(normalized_value)
        normalized[field_key] = normalized_value

    return normalized


def _decode_donut_sequence(processor: Any, sequence: str) -> tuple[dict[str, str], dict[str, Any]]:
    cleaned_sequence = (
        sequence.replace(processor.tokenizer.eos_token or "", "")
        .replace(processor.tokenizer.pad_token or "", "")
        .strip()
    )
    cleaned_sequence = re.sub(r"^<[^>]+>", "", cleaned_sequence, count=1).strip()

    parsed_payload: dict[str, Any] | None = None
    token_to_json = getattr(processor, "token2json", None)
    if callable(token_to_json):
        try:
            tokenized_payload = token_to_json(cleaned_sequence)
            if isinstance(tokenized_payload, dict):
                parsed_payload = tokenized_payload
        except Exception:
            parsed_payload = None

    if parsed_payload is None:
        parsed_payload = _extract_json_substring(cleaned_sequence)

    if parsed_payload is None:
        parsed_payload = {}

    return _normalize_donut_gt_parse(parsed_payload), parsed_payload


def _run_donut_inference(image_path: Path) -> dict[str, Any]:
    runtime = _get_donut_runtime()
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]

    with Image.open(image_path) as image:
        prepared_image = ImageOps.exif_transpose(image).convert("RGB")

    task_prompt = get_donut_task_prompt()
    pixel_values = processor(
        prepared_image,
        return_tensors="pt",
        size={
            "height": get_donut_inference_image_height(),
            "width": get_donut_inference_image_width(),
        },
    ).pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    generation_kwargs: dict[str, Any] = {
        "decoder_input_ids": decoder_input_ids,
        "max_new_tokens": get_donut_max_new_tokens(),
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "num_beams": 1,
        "do_sample": False,
    }
    if processor.tokenizer.unk_token_id is not None:
        generation_kwargs["bad_words_ids"] = [[processor.tokenizer.unk_token_id]]

    with torch.inference_mode():
        generated_sequences = model.generate(
            pixel_values,
            **generation_kwargs,
        )

    decoded_sequence = processor.batch_decode(generated_sequences, skip_special_tokens=False)[0]
    editable_fields, parsed_payload = _decode_donut_sequence(processor, decoded_sequence)
    return {
        "raw_sequence": decoded_sequence,
        "editable_fields": editable_fields,
        "parsed_payload": parsed_payload,
        "task_prompt": task_prompt,
    }


def _read_inference_cache(cache_key: str) -> dict[str, Any] | None:
    with _INFERENCE_CACHE_LOCK:
        cached_result = _INFERENCE_RESULT_CACHE.get(cache_key)
        if cached_result is None:
            return None
        performance = dict(cached_result.get("performance", {}))
        performance["cache_hit"] = True
        return {
            **cached_result,
            "performance": performance,
        }


def _write_inference_cache(cache_key: str, payload: dict[str, Any]) -> None:
    with _INFERENCE_CACHE_LOCK:
        _INFERENCE_RESULT_CACHE[cache_key] = payload
        if cache_key in _INFERENCE_CACHE_ORDER:
            _INFERENCE_CACHE_ORDER.remove(cache_key)
        _INFERENCE_CACHE_ORDER.append(cache_key)

        cache_size = get_donut_cache_size()
        while len(_INFERENCE_CACHE_ORDER) > cache_size:
            oldest_key = _INFERENCE_CACHE_ORDER.pop(0)
            _INFERENCE_RESULT_CACHE.pop(oldest_key, None)


def get_inference_image_path(image_id: str) -> Path:
    safe_name = Path(image_id).name
    return get_inference_upload_dir() / safe_name


def run_passport_inference(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    image_id, image_path = _store_uploaded_image(file_bytes, file_name)
    cached_result = _read_inference_cache(image_id)
    if cached_result is not None and image_path.exists():
        return cached_result

    total_started = perf_counter()
    ocr_started = perf_counter()
    overlay = run_ocr_with_boxes(
        image_path,
        auto_rotate=not get_inference_skip_ocr_auto_rotate(),
        fast_mode=True,
    )
    ocr_duration_ms = round((perf_counter() - ocr_started) * 1000, 2)

    donut_started = perf_counter()
    donut_result = _run_donut_inference(image_path)
    donut_duration_ms = round((perf_counter() - donut_started) * 1000, 2)
    total_duration_ms = round((perf_counter() - total_started) * 1000, 2)

    runtime = _get_donut_runtime()
    result = {
        "image_id": image_id,
        "image_name": Path(file_name or image_path.name).name,
        "image_path": str(image_path),
        "overlay": overlay,
        "editable_fields": donut_result["editable_fields"],
        "donut_raw_text": donut_result["raw_sequence"],
        "donut_json": donut_result["parsed_payload"],
        "task_prompt": donut_result["task_prompt"],
        "performance": {
            "cache_hit": False,
            "ocr_duration_ms": ocr_duration_ms,
            "donut_duration_ms": donut_duration_ms,
            "total_duration_ms": total_duration_ms,
            "donut_cpu_threads": runtime["cpu_threads"],
            "donut_device": runtime["device"],
            "donut_model_dir": runtime["model_dir"],
            "donut_processor_dir": runtime["processor_dir"],
        },
    }
    _write_inference_cache(image_id, result)
    return result
