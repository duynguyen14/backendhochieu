from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import get_connection
from app.repositories import count_records


def main() -> None:
    with get_connection() as connection:
        cursor = connection.cursor()
        record_count = count_records(cursor)
        cursor.execute("SELECT DB_NAME()")
        database_name = str(cursor.fetchone()[0])

    print(f"database: {database_name}")
    print(f"passport_ocr_records: {record_count} record(s)")


if __name__ == "__main__":
    main()
