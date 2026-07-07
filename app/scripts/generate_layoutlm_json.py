from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.layoutlm_service import (
    build_layoutlm_for_all_records,
    build_layoutlm_for_image_index_range,
    build_layoutlm_for_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LayoutLMV3 training JSON from OCR + reviewed passport data."
    )
    parser.add_argument(
        "--record-id",
        type=int,
        default=None,
        help="Generate LayoutLM JSON for a single record id.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Only process records whose image file name is numeric and greater than or equal to this value.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Only process records whose image file name is numeric and less than or equal to this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.record_id is not None and (args.start_index is not None or args.end_index is not None):
        raise ValueError("Use either --record-id or --start-index/--end-index, not both.")
    if (args.start_index is None) != (args.end_index is None):
        raise ValueError("Both --start-index and --end-index must be provided together.")
    if args.start_index is not None and args.start_index > args.end_index:
        raise ValueError("--start-index must be less than or equal to --end-index.")

    if args.record_id is not None:
        result = build_layoutlm_for_record(args.record_id)
        if result is None:
            print(f"Record not found: {args.record_id}")
            return
        print(
            "Generated layoutlm_json for record_id={record_id} | has_layoutlm_json={has_layoutlm_json}".format(
                record_id=args.record_id,
                has_layoutlm_json=result.get("has_layoutlm_json"),
            )
        )
        return

    if args.start_index is not None and args.end_index is not None:
        summary = build_layoutlm_for_image_index_range(args.start_index, args.end_index)
        print(
            "Done. updated={updated}, errors={errors}".format(
                **summary
            )
        )
        return

    summary = build_layoutlm_for_all_records()
    print(
        "Done. updated={updated}, errors={errors}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
