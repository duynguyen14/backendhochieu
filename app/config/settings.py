from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = BASE_DIR / ".env"
DEFAULT_DB_PATH = BASE_DIR / "data" / "passport_ocr.db"
DEFAULT_IMAGE_INPUT_DIR = BASE_DIR / "images"
DEFAULT_RENAME_IMAGE_DIR = DEFAULT_IMAGE_INPUT_DIR
DEFAULT_IMPORT_SOURCE_IMAGE_DIR = BASE_DIR / "imagesGoc"
DEFAULT_IMPORT_TARGET_IMAGE_DIR = DEFAULT_IMAGE_INPUT_DIR
DEFAULT_MLZ_MASK_INPUT_DIR = Path(r"C:\Users\Admin\Downloads\tonghop\tonghop\tonghop")
DEFAULT_MLZ_MASK_OUTPUT_DIR = BASE_DIR.parent / "tonghop_mask"
DEFAULT_MLZ_CROP_INPUT_DIR = Path(r"C:\Users\Admin\Downloads\tonghop\tonghop\tonghop")
DEFAULT_MLZ_CROP_OUTPUT_DIR = BASE_DIR.parent / "tonghop_crop"
DEFAULT_MLZ_METADATA_INPUT_PATH = DEFAULT_MLZ_MASK_INPUT_DIR / "metadata.jsonl"
DEFAULT_MASK_REVIEW_IMAGE_DIR = DEFAULT_MLZ_MASK_OUTPUT_DIR
DEFAULT_MASK_REVIEW_ERROR_DIR = BASE_DIR.parent / "tong_hop_mask_loi"
DEFAULT_MASK_REVIEW_STATE_PATH = DEFAULT_MASK_REVIEW_IMAGE_DIR / "review_state.json"
DEFAULT_LOG_DIR = BASE_DIR / "logs"
DEFAULT_DONUT_MODEL_DIR = BASE_DIR / "models" / "donut" / "checkpoint-16180"
DEFAULT_DONUT_PROCESSOR_DIR = BASE_DIR / "models" / "donut" / "processor"
DEFAULT_INFERENCE_UPLOAD_DIR = BASE_DIR / "uploads" / "passport_inference"


def load_env_file(env_file_path: Path = ENV_FILE_PATH) -> None:
    if not env_file_path.exists():
        return

    for raw_line in env_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_env_value(name: str, default: str = "") -> str:
    load_env_file()
    return os.getenv(name, default).strip()


def get_bool_env(name: str, default: bool) -> bool:
    raw_value = get_env_value(name, str(default)).lower()
    return raw_value in {"1", "true", "yes", "y", "on"}


def get_sql_server_connection_string() -> str:
    driver = get_env_value("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server")
    server = get_env_value("SQLSERVER_SERVER", r".\SQLEXPRESS")
    database = get_env_value("SQLSERVER_DATABASE", "HOCHIEU")
    username = get_env_value("SQLSERVER_USERNAME")
    password = get_env_value("SQLSERVER_PASSWORD")
    trusted_connection = get_bool_env("SQLSERVER_TRUSTED_CONNECTION", True)
    trust_server_certificate = get_bool_env("SQLSERVER_TRUST_SERVER_CERTIFICATE", True)

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
    ]

    if trusted_connection:
        parts.append("Trusted_Connection=yes")
    else:
        parts.append(f"UID={username}")
        parts.append(f"PWD={password}")

    if trust_server_certificate:
        parts.append("TrustServerCertificate=yes")

    return ";".join(parts) + ";"


def get_image_input_dir() -> Path:
    configured_path = get_env_value("OCR_IMAGE_INPUT_DIR", str(DEFAULT_IMAGE_INPUT_DIR))
    return Path(configured_path).expanduser().resolve()


def get_rename_image_dir() -> Path:
    configured_path = get_env_value("RENAME_IMAGE_INPUT_DIR", str(DEFAULT_RENAME_IMAGE_DIR))
    return Path(configured_path).expanduser().resolve()


def get_import_source_image_dir() -> Path:
    configured_path = get_env_value(
        "IMPORT_SOURCE_IMAGE_INPUT_DIR",
        str(DEFAULT_IMPORT_SOURCE_IMAGE_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_import_target_image_dir() -> Path:
    configured_path = get_env_value(
        "IMPORT_TARGET_IMAGE_OUTPUT_DIR",
        str(DEFAULT_IMPORT_TARGET_IMAGE_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mlz_mask_input_dir() -> Path:
    configured_path = get_env_value(
        "MLZ_MASK_INPUT_DIR",
        str(DEFAULT_MLZ_MASK_INPUT_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mlz_mask_output_dir() -> Path:
    configured_path = get_env_value(
        "MLZ_MASK_OUTPUT_DIR",
        str(DEFAULT_MLZ_MASK_OUTPUT_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mlz_crop_input_dir() -> Path:
    configured_path = get_env_value(
        "MLZ_CROP_INPUT_DIR",
        str(DEFAULT_MLZ_CROP_INPUT_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mlz_crop_output_dir() -> Path:
    configured_path = get_env_value(
        "MLZ_CROP_OUTPUT_DIR",
        str(DEFAULT_MLZ_CROP_OUTPUT_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mlz_metadata_input_path() -> Path:
    configured_path = get_env_value(
        "MLZ_METADATA_INPUT_PATH",
        str(DEFAULT_MLZ_METADATA_INPUT_PATH),
    )
    return Path(configured_path).expanduser().resolve()


def get_mask_review_image_dir() -> Path:
    configured_path = get_env_value(
        "MASK_REVIEW_IMAGE_DIR",
        str(DEFAULT_MASK_REVIEW_IMAGE_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mask_review_error_dir() -> Path:
    configured_path = get_env_value(
        "MASK_REVIEW_ERROR_DIR",
        str(DEFAULT_MASK_REVIEW_ERROR_DIR),
    )
    return Path(configured_path).expanduser().resolve()


def get_mask_review_state_path() -> Path:
    configured_path = get_env_value(
        "MASK_REVIEW_STATE_PATH",
        str(DEFAULT_MASK_REVIEW_STATE_PATH),
    )
    return Path(configured_path).expanduser().resolve()


def get_log_dir() -> Path:
    configured_path = get_env_value("APP_LOG_DIR", str(DEFAULT_LOG_DIR))
    return Path(configured_path).expanduser().resolve()


def get_donut_model_dir() -> Path:
    configured_path = get_env_value("DONUT_MODEL_DIR", str(DEFAULT_DONUT_MODEL_DIR))
    return Path(configured_path).expanduser().resolve()


def get_donut_processor_dir() -> Path:
    configured_path = get_env_value("DONUT_PROCESSOR_DIR", str(DEFAULT_DONUT_PROCESSOR_DIR))
    return Path(configured_path).expanduser().resolve()


def get_donut_task_prompt() -> str:
    return get_env_value("DONUT_TASK_PROMPT", "<s_passport>")


def get_donut_max_new_tokens() -> int:
    return int(get_env_value("DONUT_MAX_NEW_TOKENS", "256"))


def get_donut_cpu_threads() -> int:
    return max(1, int(get_env_value("DONUT_CPU_THREADS", "4")))


def get_donut_device() -> str:
    return get_env_value("DONUT_DEVICE", "auto").lower()


def get_donut_cache_size() -> int:
    return max(1, int(get_env_value("DONUT_CACHE_SIZE", "32")))


def get_inference_upload_dir() -> Path:
    configured_path = get_env_value("INFERENCE_UPLOAD_DIR", str(DEFAULT_INFERENCE_UPLOAD_DIR))
    return Path(configured_path).expanduser().resolve()


def get_inference_skip_ocr_auto_rotate() -> bool:
    return get_bool_env("INFERENCE_SKIP_OCR_AUTO_ROTATE", False)


def get_donut_inference_image_width() -> int:
    return max(256, int(get_env_value("DONUT_INFERENCE_IMAGE_WIDTH", "2560")))


def get_donut_inference_image_height() -> int:
    return max(256, int(get_env_value("DONUT_INFERENCE_IMAGE_HEIGHT", "1920")))


def get_donut_use_dynamic_quantization() -> bool:
    return get_bool_env("DONUT_USE_DYNAMIC_QUANTIZATION", True)


def get_ocr_language() -> str:
    return get_env_value("OCR_LANGUAGE", "en")


def get_paddle_ocr_version() -> str:
    return get_env_value("PADDLE_OCR_VERSION", "PP-OCRv5")


def get_paddle_ocr_device() -> str:
    return get_env_value("PADDLE_OCR_DEVICE", "cpu")


def get_paddle_model_source() -> str:
    return get_env_value("PADDLE_PDX_MODEL_SOURCE", "BOS")


def get_paddle_text_detection_model_dir() -> Path | None:
    configured_path = get_env_value("PADDLE_TEXT_DETECTION_MODEL_DIR")
    if not configured_path:
        return None
    return Path(configured_path).expanduser().resolve()


def get_paddle_text_recognition_model_dir() -> Path | None:
    configured_path = get_env_value("PADDLE_TEXT_RECOGNITION_MODEL_DIR")
    if not configured_path:
        return None
    return Path(configured_path).expanduser().resolve()


def get_paddle_doc_orientation_model_dir() -> Path | None:
    configured_path = get_env_value("PADDLE_DOC_ORIENTATION_MODEL_DIR")
    if not configured_path:
        return None
    return Path(configured_path).expanduser().resolve()


def get_paddle_textline_orientation_model_dir() -> Path | None:
    configured_path = get_env_value("PADDLE_TEXTLINE_ORIENTATION_MODEL_DIR")
    if not configured_path:
        return None
    return Path(configured_path).expanduser().resolve()


def get_paddle_use_doc_orientation_classify() -> bool:
    return get_bool_env("PADDLE_USE_DOC_ORIENTATION_CLASSIFY", True)


def get_paddle_use_textline_orientation() -> bool:
    return get_bool_env("PADDLE_USE_TEXTLINE_ORIENTATION", True)


def get_ocr_auto_rotate_and_overwrite() -> bool:
    return get_bool_env("OCR_AUTO_ROTATE_AND_OVERWRITE", True)


def get_api_host() -> str:
    return get_env_value("API_HOST", "0.0.0.0")


def get_api_port() -> int:
    return int(get_env_value("API_PORT", "8000"))


def get_frontend_allowed_origins() -> list[str]:
    raw_value = get_env_value("FRONTEND_ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]
