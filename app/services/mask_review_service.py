from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Request

from app.config import (
    get_mask_review_error_dir,
    get_mask_review_image_dir,
    get_mask_review_state_path,
)
from app.services.ocr_service import SUPPORTED_EXTENSIONS


ReviewDecision = Literal["approved", "rejected"]
METADATA_FILE_NAME = "metadata.jsonl"


def _safe_file_name(file_name: str) -> str:
    return Path(str(file_name or "")).name.strip()


def _file_name_sort_key(file_name: str) -> tuple[int, int | str, str]:
    path = Path(file_name)
    stem = path.stem.replace("_mask", "").strip()
    if stem.isdigit():
        return (0, int(stem), path.suffix.lower())
    return (1, path.name.lower(), path.suffix.lower())


def _image_sort_key(image_path: Path) -> tuple[int, int | str, str]:
    return _file_name_sort_key(image_path.name)


def _get_review_image_dir() -> Path:
    return get_mask_review_image_dir().resolve()


def _get_review_error_dir() -> Path:
    return get_mask_review_error_dir().resolve()


def _get_review_state_path() -> Path:
    return get_mask_review_state_path().resolve()


def _get_metadata_path() -> Path:
    return _get_review_image_dir() / METADATA_FILE_NAME


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return loaded if isinstance(loaded, dict) else {}


def _load_review_state() -> dict[str, Any]:
    loaded = _load_json_file(_get_review_state_path())
    decisions = loaded.get("decisions", {})
    if not isinstance(decisions, dict):
        decisions = {}

    normalized_decisions: dict[str, dict[str, Any]] = {}
    for raw_file_name, raw_entry in decisions.items():
        file_name = _safe_file_name(str(raw_file_name))
        if not file_name or not isinstance(raw_entry, dict):
            continue

        status = str(raw_entry.get("status", "")).strip().lower()
        if status not in {"approved", "rejected"}:
            continue

        normalized_decisions[file_name] = {
            "status": status,
            "reviewed_at_utc": str(raw_entry.get("reviewed_at_utc", "")).strip(),
            "moved_to_error_path": str(raw_entry.get("moved_to_error_path", "")).strip(),
        }

    return {
        "decisions": normalized_decisions,
        "last_reviewed_file_name": _safe_file_name(str(loaded.get("last_reviewed_file_name", ""))),
        "updated_at_utc": str(loaded.get("updated_at_utc", "")).strip(),
    }


def _save_review_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path = _get_review_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _list_review_images() -> list[Path]:
    image_dir = _get_review_image_dir()
    if not image_dir.exists():
        return []

    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=_image_sort_key,
    )


def _list_error_images() -> list[Path]:
    error_dir = _get_review_error_dir()
    if not error_dir.exists():
        return []

    return sorted(
        [
            path
            for path in error_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=_image_sort_key,
    )


def _get_existing_review_image_path(file_name: str) -> Path | None:
    safe_name = _safe_file_name(file_name)
    if not safe_name:
        return None

    image_path = _get_review_image_dir() / safe_name
    if image_path.exists():
        return image_path

    error_path = _get_review_error_dir() / safe_name
    if error_path.exists():
        return error_path

    return None


def _load_metadata_entries() -> list[dict[str, Any]]:
    metadata_path = _get_metadata_path()
    if not metadata_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            entries.append(parsed)

    return entries


def _write_metadata_entries(entries: list[dict[str, Any]]) -> None:
    metadata_path = _get_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    metadata_path.write_text("\n".join(lines), encoding="utf-8")


def _remove_metadata_entry(file_name: str) -> None:
    safe_name = _safe_file_name(file_name)
    if not safe_name:
        return

    filtered_entries = [
        entry
        for entry in _load_metadata_entries()
        if _safe_file_name(str(entry.get("file_name", ""))) != safe_name
    ]
    _write_metadata_entries(filtered_entries)


def _move_image_to_error_folder(image_path: Path) -> Path:
    error_dir = _get_review_error_dir()
    error_dir.mkdir(parents=True, exist_ok=True)
    destination_path = error_dir / image_path.name
    if destination_path.exists():
        destination_path.unlink()
    return image_path.replace(destination_path)


def _build_recent_decisions(
    decisions: dict[str, dict[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    items = sorted(
        decisions.items(),
        key=lambda item: item[1].get("reviewed_at_utc", ""),
        reverse=True,
    )

    return [
        {
            "file_name": file_name,
            "status": entry.get("status", ""),
            "reviewed_at_utc": entry.get("reviewed_at_utc", ""),
            "moved_to_error_path": entry.get("moved_to_error_path", ""),
        }
        for file_name, entry in items[:limit]
    ]


def _build_current_item_payload(
    request: Request,
    *,
    file_name: str,
    ordered_file_names: list[str],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    index = ordered_file_names.index(file_name)
    status = decisions.get(file_name, {}).get("status", "pending")
    return {
        "file_name": file_name,
        "source_file_name": file_name.replace("_mask", ""),
        "image_url": str(request.url_for("get_mask_review_image", file_name=file_name)),
        "position": index + 1,
        "status": status,
        "is_reviewed": status in {"approved", "rejected"},
        "previous_file_name": ordered_file_names[index - 1] if index > 0 else "",
        "next_file_name": ordered_file_names[index + 1] if index < len(ordered_file_names) - 1 else "",
    }


def _find_resume_file_name(
    *,
    pending_file_names: list[str],
    ordered_file_names: list[str],
    last_reviewed_file_name: str,
) -> str:
    if not pending_file_names:
        return ""

    safe_last_reviewed = _safe_file_name(last_reviewed_file_name)
    if safe_last_reviewed not in ordered_file_names:
        return pending_file_names[0]

    ordered_index_by_name = {
        file_name: index
        for index, file_name in enumerate(ordered_file_names)
    }
    last_index = ordered_index_by_name[safe_last_reviewed]
    for file_name in pending_file_names:
        if ordered_index_by_name.get(file_name, -1) > last_index:
            return file_name

    return pending_file_names[0]


def _build_review_session_payload(
    request: Request,
    *,
    selected_file_name: str = "",
) -> dict[str, Any]:
    image_dir = _get_review_image_dir()
    state = _load_review_state()
    review_images = _list_review_images()
    error_images = _list_error_images()
    current_file_names = {image_path.name for image_path in review_images}
    error_file_names = {image_path.name for image_path in error_images}
    decisions = state["decisions"]

    approved_count = sum(
        1
        for file_name, entry in decisions.items()
        if entry.get("status") == "approved" and file_name in current_file_names
    )
    rejected_count = sum(
        1 for entry in decisions.values() if entry.get("status") == "rejected"
    )

    pending_images = [
        image_path
        for image_path in review_images
        if image_path.name not in decisions
    ]
    pending_file_names = [image_path.name for image_path in pending_images]

    reviewed_items = approved_count + rejected_count
    total_items = len(review_images) + rejected_count
    pending_items = len(pending_images)

    ordered_file_names = sorted(
        current_file_names | error_file_names,
        key=_file_name_sort_key,
    )
    selected_name = _safe_file_name(selected_file_name)
    if selected_name not in ordered_file_names:
        selected_name = _find_resume_file_name(
            pending_file_names=pending_file_names,
            ordered_file_names=ordered_file_names,
            last_reviewed_file_name=state.get("last_reviewed_file_name", ""),
        )

    current_item_payload: dict[str, Any] | None = None
    if selected_name and selected_name in ordered_file_names:
        current_item_payload = _build_current_item_payload(
            request,
            file_name=selected_name,
            ordered_file_names=ordered_file_names,
            decisions=decisions,
        )

    return {
        "status": "success",
        "data": {
            "image_dir": str(image_dir),
            "error_dir": str(_get_review_error_dir()),
            "metadata_path": str(_get_metadata_path()),
            "review_state_path": str(_get_review_state_path()),
            "current_item": current_item_payload,
            "recent_decisions": _build_recent_decisions(decisions),
            "stats": {
                "total_items": total_items,
                "reviewed_items": reviewed_items,
                "approved_items": approved_count,
                "rejected_items": rejected_count,
                "pending_items": pending_items,
            },
            "last_reviewed_file_name": state.get("last_reviewed_file_name", ""),
        },
    }


def get_mask_review_session(request: Request) -> dict[str, Any]:
    return _build_review_session_payload(request)


def get_mask_review_session_for_file(request: Request, file_name: str) -> dict[str, Any]:
    return _build_review_session_payload(request, selected_file_name=file_name)


def get_mask_review_image_path(file_name: str) -> Path:
    safe_name = _safe_file_name(file_name)
    if not safe_name:
        raise FileNotFoundError("Review image file name is empty.")

    image_path = _get_existing_review_image_path(safe_name)
    if image_path is None:
        raise FileNotFoundError(f"Review image not found: {safe_name}")

    return image_path


def save_mask_review_decision(
    *,
    file_name: str,
    decision: ReviewDecision,
    request: Request,
) -> dict[str, Any]:
    safe_name = _safe_file_name(file_name)
    if not safe_name:
        raise FileNotFoundError("Review image file name is empty.")

    image_path = _get_review_image_dir() / safe_name
    if not image_path.exists():
        existing_image_path = _get_existing_review_image_path(safe_name)
        if existing_image_path is not None and decision == "approved":
            return _build_review_session_payload(request, selected_file_name=safe_name)
        raise FileNotFoundError(f"Review image not found: {safe_name}")

    state = _load_review_state()
    ordered_file_names = sorted(
        {image.name for image in _list_review_images()} | {image.name for image in _list_error_images()},
        key=_file_name_sort_key,
    )
    next_file_name = ""
    if safe_name in ordered_file_names:
        current_index = ordered_file_names.index(safe_name)
        if current_index < len(ordered_file_names) - 1:
            next_file_name = ordered_file_names[current_index + 1]

    decision_entry = {
        "status": decision,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "moved_to_error_path": "",
    }

    if decision == "rejected":
        moved_path = _move_image_to_error_folder(image_path)
        decision_entry["moved_to_error_path"] = str(moved_path)
        _remove_metadata_entry(safe_name)

    state["decisions"][safe_name] = decision_entry
    state["last_reviewed_file_name"] = safe_name
    _save_review_state(state)

    return _build_review_session_payload(request, selected_file_name=next_file_name)
