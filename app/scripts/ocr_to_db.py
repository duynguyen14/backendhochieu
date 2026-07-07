from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_image_input_dir
from app.services.ocr_service import process_images_to_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR on passport images and save the results to SQL Server."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to an image file or a directory of images. Defaults to OCR_IMAGE_INPUT_DIR from .env.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories when the input path is a folder.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Only process images whose numeric file name is greater than or equal to this value.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Only process images whose numeric file name is less than or equal to this value.",
    )
    parser.add_argument(
        "--generate-layoutlm",
        action="store_true",
        help="Generate and save layoutlm_json in the same OCR pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.start_index is None) != (args.end_index is None):
        raise ValueError("Both --start-index and --end-index must be provided together.")
    if args.start_index is not None and args.start_index > args.end_index:
        raise ValueError("--start-index must be less than or equal to --end-index.")

    input_path = (args.input or get_image_input_dir()).expanduser().resolve()
    result = process_images_to_database(
        input_path=input_path,
        recursive=args.recursive,
        start_index=args.start_index,
        end_index=args.end_index,
        generate_layoutlm=args.generate_layoutlm,
    )
    print(
        "Done. inserted={inserted}, skipped_duplicate={skipped_duplicate}, errors={errors}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
