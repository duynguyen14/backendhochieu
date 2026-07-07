from __future__ import annotations

import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(folder_path: Path) -> list[Path]:
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def rename_images(folder_path: Path) -> int:
    image_paths = collect_images(folder_path)
    if not image_paths:
        print(f"No PNG/JPG images found in: {folder_path}")
        return 0

    temporary_paths: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        temporary_path = folder_path / f"__renaming_tmp__{index}{image_path.suffix.lower()}"
        image_path.rename(temporary_path)
        temporary_paths.append(temporary_path)

    for index, temporary_path in enumerate(temporary_paths, start=1):
        final_path = folder_path / f"{index}{temporary_path.suffix.lower()}"
        temporary_path.rename(final_path)
        print(f"Renamed -> {final_path.name}")

    return len(temporary_paths)


def collect_images_recursive(folder_path: Path) -> list[Path]:
    return sorted(
        path
        for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def copy_images_flattened(
    source_folder: Path,
    target_folder: Path,
    start_index: int,
) -> int:
    image_paths = collect_images_recursive(source_folder)
    if not image_paths:
        print(f"No PNG/JPG images found in: {source_folder}")
        return 0

    target_folder.mkdir(parents=True, exist_ok=True)
    next_index = start_index
    copied_count = 0

    for image_path in image_paths:
        destination_path = target_folder / f"{next_index}{image_path.suffix.lower()}"
        while destination_path.exists():
            next_index += 1
            destination_path = target_folder / f"{next_index}{image_path.suffix.lower()}"

        shutil.copy2(image_path, destination_path)
        print(f"Copied -> {destination_path.name}")
        next_index += 1
        copied_count += 1

    return copied_count
