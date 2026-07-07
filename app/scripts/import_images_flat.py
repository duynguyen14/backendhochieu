from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_import_source_image_dir, get_import_target_image_dir
from app.services.image_rename_service import copy_images_flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy all PNG/JPG images from nested folders into one target folder and rename sequentially."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source folder to scan recursively. Defaults to IMPORT_SOURCE_IMAGE_INPUT_DIR from .env.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target folder to receive copied images. Defaults to IMPORT_TARGET_IMAGE_OUTPUT_DIR from .env.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First numeric filename to use in the target folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_folder = (args.source or get_import_source_image_dir()).expanduser().resolve()
    target_folder = (args.target or get_import_target_image_dir()).expanduser().resolve()

    copied_count = copy_images_flattened(
        source_folder=source_folder,
        target_folder=target_folder,
        start_index=args.start_index,
    )
    print(f"Done. copied={copied_count}")


if __name__ == "__main__":
    main()
