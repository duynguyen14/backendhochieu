from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageOps

from app.services.ocr_service import SUPPORTED_EXTENSIONS, run_ocr_with_boxes


MRZ_MODE = Literal["mask", "crop"]
MRZ_MIN_TOP_RATIO = 0.45
MRZ_STRONG_TOP_RATIO = 0.6
MRZ_MIN_WIDTH_RATIO = 0.3
MRZ_STRONG_WIDTH_RATIO = 0.45
MRZ_MIN_TEXT_LENGTH = 20
MRZ_VERTICAL_PADDING_RATIO = 0.012
MRZ_MASK_FILL_COLOR = (255, 255, 255)
METADATA_FILE_NAME = "metadata.jsonl"
PROGRESS_FILE_NAME = "progress.json"
CHECKPOINT_BATCH_SIZE = 50
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MRZ_VALID_CHAR_PATTERN = re.compile(r"^[A-Z0-9<]+$")


def collect_supported_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path.resolve()]

    pattern = "**/*" if recursive else "*"
    return sorted(
        path.resolve()
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _clean_mrz_text(raw_text: str) -> str:
    return _WHITESPACE_PATTERN.sub("", str(raw_text or "").upper())


def _compute_mrz_score(
    *,
    cleaned_text: str,
    bbox: dict[str, Any],
    image_width: int,
    image_height: int,
) -> int:
    width = max(0, int(bbox["width"]))
    top = max(0, int(bbox["top"]))
    bottom = max(top, int(bbox["bottom"]))

    width_ratio = width / max(1, image_width)
    top_ratio = top / max(1, image_height)
    bottom_ratio = bottom / max(1, image_height)
    valid_char_ratio = 0.0
    if cleaned_text:
        valid_char_count = sum(1 for char in cleaned_text if char.isalnum() or char == "<")
        valid_char_ratio = valid_char_count / len(cleaned_text)

    score = 0
    if "<" in cleaned_text:
        score += 4
    if cleaned_text.count("<") >= 2:
        score += 2
    if len(cleaned_text) >= 30:
        score += 2
    elif len(cleaned_text) >= MRZ_MIN_TEXT_LENGTH:
        score += 1
    if width_ratio >= MRZ_STRONG_WIDTH_RATIO:
        score += 3
    elif width_ratio >= MRZ_MIN_WIDTH_RATIO:
        score += 1
    if top_ratio >= MRZ_STRONG_TOP_RATIO:
        score += 3
    elif top_ratio >= MRZ_MIN_TOP_RATIO:
        score += 1
    if bottom_ratio >= 0.8:
        score += 2
    if valid_char_ratio >= 0.95:
        score += 2
    elif valid_char_ratio >= 0.85:
        score += 1
    if cleaned_text.startswith("P<"):
        score += 2
    if _MRZ_VALID_CHAR_PATTERN.fullmatch(cleaned_text):
        score += 1

    return score


def detect_mrz_region(image_path: Path) -> dict[str, int] | None:
    overlay = run_ocr_with_boxes(image_path, auto_rotate=False, fast_mode=True)
    image_width = int(overlay.get("image_width", 0))
    image_height = int(overlay.get("image_height", 0))
    if image_width <= 0 or image_height <= 0:
        return None

    scored_lines: list[dict[str, Any]] = []
    for line in overlay.get("lines", []):
        bbox = line.get("bbox")
        if not isinstance(bbox, dict):
            continue

        cleaned_text = _clean_mrz_text(str(line.get("text", "")))
        if len(cleaned_text) < MRZ_MIN_TEXT_LENGTH:
            continue

        score = _compute_mrz_score(
            cleaned_text=cleaned_text,
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )
        if score < 5:
            continue

        scored_lines.append(
            {
                "score": score,
                "bbox": bbox,
                "cleaned_text": cleaned_text,
            }
        )

    if not scored_lines:
        return None

    strong_candidates = [
        line
        for line in scored_lines
        if line["score"] >= 8 and int(line["bbox"]["top"]) >= int(image_height * MRZ_MIN_TOP_RATIO)
    ]
    candidates = strong_candidates or sorted(
        scored_lines,
        key=lambda line: (int(line["bbox"]["bottom"]), line["score"]),
        reverse=True,
    )[:2]
    if not candidates:
        return None

    top = min(int(candidate["bbox"]["top"]) for candidate in candidates)
    bottom = image_height

    pad_y = max(4, int(image_height * MRZ_VERTICAL_PADDING_RATIO))

    return {
        "left": 0,
        "top": max(0, top - pad_y),
        "right": image_width,
        "bottom": bottom,
    }


def _ensure_output_path(
    *,
    image_path: Path,
    input_path: Path,
    output_dir: Path,
) -> Path:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()

    output_name = image_path.name
    if input_path.is_file():
        destination_path = output_dir / output_name
    else:
        relative_path = image_path.relative_to(input_path)
        destination_path = output_dir / relative_path.parent / output_name

    if destination_path.resolve() == image_path.resolve():
        raise ValueError("Output path must be different from input path.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    return destination_path


def _save_image(image: Image.Image, destination_path: Path) -> None:
    suffix = destination_path.suffix.lower()
    save_kwargs: dict[str, Any] = {}

    if suffix in {".jpg", ".jpeg"}:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        save_kwargs["quality"] = 95
    elif suffix == ".png":
        save_kwargs["compress_level"] = 3

    image.save(destination_path, **save_kwargs)


def _build_output_file_name(image_path: Path, suffix: str) -> str:
    return f"{image_path.stem}_{suffix}{image_path.suffix.lower()}"


def _safe_load_json(raw_text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(loaded, dict):
        return {}

    return loaded


def _load_metadata_by_file_name(metadata_path: Path) -> dict[str, dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_by_file_name: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {metadata_path}:{line_number}") from exc

        if not isinstance(payload, dict):
            continue

        file_name = str(payload.get("file_name", "")).strip()
        if not file_name:
            continue

        metadata_by_file_name[file_name.lower()] = payload

    return metadata_by_file_name


def _build_output_metadata_entry(
    source_entry: dict[str, Any],
    *,
    output_file_name: str,
) -> dict[str, Any]:
    cloned_entry = dict(source_entry)
    cloned_entry["file_name"] = output_file_name

    ground_truth_payload = _safe_load_json(str(cloned_entry.get("ground_truth", "")))
    gt_parse = ground_truth_payload.get("gt_parse")
    if isinstance(gt_parse, dict):
        normalized_gt_parse = dict(gt_parse)
        normalized_gt_parse["personal_number"] = ""
        ground_truth_payload["gt_parse"] = normalized_gt_parse
        cloned_entry["ground_truth"] = json.dumps(ground_truth_payload, ensure_ascii=False)

    return cloned_entry


def _write_metadata_file(output_dir: Path, metadata_entries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE_NAME
    lines = [json.dumps(entry, ensure_ascii=False) for entry in metadata_entries]
    metadata_path.write_text("\n".join(lines), encoding="utf-8")


def _output_metadata_path(output_dir: Path) -> Path:
    return output_dir / METADATA_FILE_NAME


def _progress_path(output_dir: Path) -> Path:
    return output_dir / PROGRESS_FILE_NAME


def _load_existing_output_metadata(output_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = _output_metadata_path(output_dir)
    if not metadata_path.exists():
        return {}
    return _load_metadata_by_file_name(metadata_path)


def _safe_load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_progress_state(output_dir: Path) -> dict[str, Any]:
    loaded = _safe_load_json_file(_progress_path(output_dir))
    processed_sources = loaded.get("processed_sources", [])
    skipped_no_mrz_sources = loaded.get("skipped_no_mrz_sources", [])
    error_sources = loaded.get("error_sources", [])

    return {
        "processed_sources": {
            str(file_name).strip().lower()
            for file_name in processed_sources
            if str(file_name).strip()
        },
        "skipped_no_mrz_sources": {
            str(file_name).strip().lower()
            for file_name in skipped_no_mrz_sources
            if str(file_name).strip()
        },
        "error_sources": {
            str(file_name).strip().lower()
            for file_name in error_sources
            if str(file_name).strip()
        },
        "last_completed_source": str(loaded.get("last_completed_source", "")).strip(),
    }


def _persist_progress_state(
    *,
    output_dir: Path,
    mode: MRZ_MODE,
    input_path: Path,
    metadata_path: Path | None,
    progress_state: dict[str, Any],
) -> None:
    payload = {
        "mode": mode,
        "checkpoint_batch_size": CHECKPOINT_BATCH_SIZE,
        "input_path": str(input_path),
        "metadata_path": str(metadata_path) if metadata_path is not None else "",
        "last_completed_source": str(progress_state.get("last_completed_source", "")),
        "processed_sources": sorted(progress_state["processed_sources"]),
        "skipped_no_mrz_sources": sorted(progress_state["skipped_no_mrz_sources"]),
        "error_sources": sorted(progress_state["error_sources"]),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _progress_path(output_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _recover_source_file_name_from_output(output_file_name: str, mode: MRZ_MODE) -> str | None:
    output_path = Path(output_file_name)
    expected_suffix = f"_{mode}"
    if not output_path.stem.lower().endswith(expected_suffix):
        return None
    source_stem = output_path.stem[: -len(expected_suffix)]
    if not source_stem:
        return None
    return f"{source_stem}{output_path.suffix.lower()}"


def _collect_processed_sources_from_metadata(
    output_metadata_by_file_name: dict[str, dict[str, Any]],
    mode: MRZ_MODE,
) -> set[str]:
    processed_sources: set[str] = set()
    for output_file_name in output_metadata_by_file_name:
        recovered = _recover_source_file_name_from_output(output_file_name, mode)
        if recovered:
            processed_sources.add(recovered.lower())
    return processed_sources


def _collect_processed_sources_from_existing_outputs(output_dir: Path, mode: MRZ_MODE) -> set[str]:
    processed_sources: set[str] = set()
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        recovered = _recover_source_file_name_from_output(path.name, mode)
        if recovered:
            processed_sources.add(recovered.lower())
    return processed_sources


def _write_checkpoint(
    *,
    output_dir: Path,
    mode: MRZ_MODE,
    input_path: Path,
    metadata_path: Path | None,
    output_metadata_by_file_name: dict[str, dict[str, Any]],
    progress_state: dict[str, Any],
) -> None:
    _write_metadata_file(
        output_dir,
        [
            output_metadata_by_file_name[file_name]
            for file_name in sorted(output_metadata_by_file_name)
        ],
    )
    _persist_progress_state(
        output_dir=output_dir,
        mode=mode,
        input_path=input_path,
        metadata_path=metadata_path,
        progress_state=progress_state,
    )


def mask_mrz_region(image_path: Path, destination_path: Path) -> bool:
    mrz_region = detect_mrz_region(image_path)
    if mrz_region is None:
        return False

    with Image.open(image_path) as image:
        working_image = ImageOps.exif_transpose(image).convert("RGB")
        drawer = ImageDraw.Draw(working_image)
        drawer.rectangle(
            [
                mrz_region["left"],
                mrz_region["top"],
                mrz_region["right"],
                mrz_region["bottom"],
            ],
            fill=MRZ_MASK_FILL_COLOR,
        )
        _save_image(working_image, destination_path)

    return True


def crop_mrz_region(image_path: Path, destination_path: Path) -> bool:
    mrz_region = detect_mrz_region(image_path)
    if mrz_region is None:
        return False

    crop_bottom = max(1, int(mrz_region["top"]))
    with Image.open(image_path) as image:
        working_image = ImageOps.exif_transpose(image).convert("RGB")
        if crop_bottom >= working_image.height:
            return False
        cropped_image = working_image.crop((0, 0, working_image.width, crop_bottom))
        _save_image(cropped_image, destination_path)

    return True


def process_mrz_images(
    *,
    input_path: Path,
    output_dir: Path,
    mode: MRZ_MODE,
    metadata_path: Path | None = None,
    recursive: bool = False,
) -> dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_dir = output_dir.expanduser().resolve()
    images = collect_supported_images(input_path, recursive)
    images = [image_path for image_path in images if not _is_relative_to(image_path, output_dir)]
    if not images:
        print(f"No supported images found at: {input_path}")
        return {"processed": 0, "skipped_no_mrz": 0, "errors": 0}

    processed_count = 0
    resumed_skip_count = 0
    skipped_count = 0
    error_count = 0
    missing_metadata_count = 0
    metadata_by_file_name = (
        _load_metadata_by_file_name(metadata_path.resolve())
        if metadata_path is not None
        else {}
    )
    output_metadata_by_file_name = _load_existing_output_metadata(output_dir)
    progress_state = _load_progress_state(output_dir)
    progress_state["processed_sources"].update(
        _collect_processed_sources_from_metadata(output_metadata_by_file_name, mode)
    )
    progress_state["processed_sources"].update(
        _collect_processed_sources_from_existing_outputs(output_dir, mode)
    )
    since_last_checkpoint = 0

    for image_path in images:
        output_suffix = "mask" if mode == "mask" else "crop"
        output_file_name = _build_output_file_name(image_path, output_suffix)
        destination_path = _ensure_output_path(
            image_path=image_path.with_name(output_file_name),
            input_path=input_path,
            output_dir=output_dir,
        )
        source_file_name = image_path.name.lower()

        if source_file_name in progress_state["processed_sources"] and (
            destination_path.exists() or source_file_name in progress_state["skipped_no_mrz_sources"]
        ):
            resumed_skip_count += 1
            continue

        try:
            if mode == "mask":
                changed = mask_mrz_region(image_path, destination_path)
            elif mode == "crop":
                changed = crop_mrz_region(image_path, destination_path)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if not changed:
                skipped_count += 1
                progress_state["processed_sources"].add(source_file_name)
                progress_state["skipped_no_mrz_sources"].add(source_file_name)
                progress_state["error_sources"].discard(source_file_name)
                progress_state["last_completed_source"] = image_path.name
                since_last_checkpoint += 1
                print(f"Skip (MRZ not found) -> {image_path.name}")
                if since_last_checkpoint >= CHECKPOINT_BATCH_SIZE:
                    _write_checkpoint(
                        output_dir=output_dir,
                        mode=mode,
                        input_path=input_path,
                        metadata_path=metadata_path,
                        output_metadata_by_file_name=output_metadata_by_file_name,
                        progress_state=progress_state,
                    )
                    since_last_checkpoint = 0
                continue

            processed_count += 1
            source_entry = metadata_by_file_name.get(image_path.name.lower())
            if source_entry is None:
                missing_metadata_count += 1
                print(f"Metadata missing -> {image_path.name}")
            else:
                output_metadata_by_file_name[output_file_name.lower()] = _build_output_metadata_entry(
                    source_entry,
                    output_file_name=output_file_name,
                )
            progress_state["processed_sources"].add(source_file_name)
            progress_state["skipped_no_mrz_sources"].discard(source_file_name)
            progress_state["error_sources"].discard(source_file_name)
            progress_state["last_completed_source"] = image_path.name
            since_last_checkpoint += 1
            print(f"Saved -> {destination_path}")
        except Exception as exc:
            error_count += 1
            progress_state["error_sources"].add(source_file_name)
            print(f"Error -> {image_path} | {exc}")

        if since_last_checkpoint >= CHECKPOINT_BATCH_SIZE:
            _write_checkpoint(
                output_dir=output_dir,
                mode=mode,
                input_path=input_path,
                metadata_path=metadata_path,
                output_metadata_by_file_name=output_metadata_by_file_name,
                progress_state=progress_state,
            )
            since_last_checkpoint = 0

    _write_checkpoint(
        output_dir=output_dir,
        mode=mode,
        input_path=input_path,
        metadata_path=metadata_path,
        output_metadata_by_file_name=output_metadata_by_file_name,
        progress_state=progress_state,
    )

    return {
        "processed": processed_count,
        "resumed_skip": resumed_skip_count,
        "skipped_no_mrz": skipped_count,
        "errors": error_count,
        "metadata_written": len(output_metadata_by_file_name),
        "metadata_missing": missing_metadata_count,
    }
