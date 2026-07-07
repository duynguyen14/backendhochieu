from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_rename_image_dir
from app.services.image_rename_service import rename_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename all PNG/JPG images in a configured folder to 1..N."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the image folder. Defaults to RENAME_IMAGE_INPUT_DIR from .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_path = (args.input or get_rename_image_dir()).expanduser().resolve()

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder_path}")

    renamed_count = rename_images(folder_path)
    print(f"Done. renamed={renamed_count}")


if __name__ == "__main__":
    main()
