from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import (
    get_mlz_mask_input_dir,
    get_mlz_mask_output_dir,
    get_mlz_metadata_input_path,
)
from app.services.mrz_image_service import process_mrz_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the passport MLZ/MRZ area and mask it in a new output folder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image file or folder. Defaults to MLZ_MASK_INPUT_DIR from .env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for masked images. Defaults to MLZ_MASK_OUTPUT_DIR from .env.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Source metadata.jsonl path. Defaults to MLZ_METADATA_INPUT_PATH from .env.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories when the input path is a folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = (args.input or get_mlz_mask_input_dir()).expanduser().resolve()
    output_dir = (args.output or get_mlz_mask_output_dir()).expanduser().resolve()
    metadata_path = (args.metadata or get_mlz_metadata_input_path()).expanduser().resolve()
    result = process_mrz_images(
        input_path=input_path,
        output_dir=output_dir,
        mode="mask",
        metadata_path=metadata_path,
        recursive=args.recursive,
    )
    print(
        "Done. processed={processed}, resumed_skip={resumed_skip}, skipped_no_mrz={skipped_no_mrz}, errors={errors}, metadata_written={metadata_written}, metadata_missing={metadata_missing}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
