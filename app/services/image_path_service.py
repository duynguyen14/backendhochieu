from __future__ import annotations

from pathlib import Path

from app.config import get_image_input_dir


def resolve_record_image_path(image_path_value: str | Path) -> Path:
    image_path = Path(str(image_path_value))
    if image_path.exists():
        return image_path

    file_name = image_path.name.strip()
    if file_name:
        fallback_path = get_image_input_dir() / file_name
        if fallback_path.exists():
            return fallback_path.resolve()

    return image_path
