from __future__ import annotations

from pathlib import Path

import pyodbc

from app.models import ExistingRecord


def count_records(cursor: pyodbc.Cursor) -> int:
    cursor.execute("SELECT COUNT(*) FROM dbo.passport_ocr_records")
    return int(cursor.fetchone()[0])


def count_records_with_status(cursor: pyodbc.Cursor, status: str | None = None) -> int:
    if status:
        cursor.execute(
            "SELECT COUNT(*) FROM dbo.passport_ocr_records WHERE status = ?",
            status,
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM dbo.passport_ocr_records")
    return int(cursor.fetchone()[0])


def load_existing_records(cursor: pyodbc.Cursor) -> list[ExistingRecord]:
    cursor.execute("SELECT id, image_path FROM dbo.passport_ocr_records")
    records: list[ExistingRecord] = []

    for row in cursor.fetchall():
        image_path = str(row.image_path)
        records.append(
            ExistingRecord(
                id=int(row.id),
                image_path=image_path,
                normalized_path=image_path.replace("/", "\\").lower(),
                file_name=Path(image_path).name.lower(),
            )
        )

    return records


def list_records_paginated(
    cursor: pyodbc.Cursor,
    *,
    page: int,
    page_size: int,
    status: str | None = None,
) -> list[pyodbc.Row]:
    offset = (page - 1) * page_size

    if status:
        cursor.execute(
            """
            SELECT
                id,
                image_path,
                extracted_json,
                reviewed_json,
                layoutlm_json,
                layoutlm_reviewed_json,
                status,
                raw_ocr_text,
                created_at,
                updated_at
            FROM dbo.passport_ocr_records
            WHERE status = ?
            ORDER BY id
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            status,
            offset,
            page_size,
        )
    else:
        cursor.execute(
            """
            SELECT
                id,
                image_path,
                extracted_json,
                reviewed_json,
                layoutlm_json,
                layoutlm_reviewed_json,
                status,
                raw_ocr_text,
                created_at,
                updated_at
            FROM dbo.passport_ocr_records
            ORDER BY id
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            offset,
            page_size,
        )

    return list(cursor.fetchall())


def get_record_by_id(cursor: pyodbc.Cursor, record_id: int) -> pyodbc.Row | None:
    cursor.execute(
        """
        SELECT
            id,
            image_path,
            extracted_json,
            reviewed_json,
            layoutlm_json,
            layoutlm_reviewed_json,
            status,
            raw_ocr_text,
            error_message,
            created_at,
            updated_at
        FROM dbo.passport_ocr_records
        WHERE id = ?
        """,
        record_id,
    )
    return cursor.fetchone()


def get_previous_record_id(cursor: pyodbc.Cursor, record_id: int) -> int | None:
    cursor.execute(
        "SELECT TOP 1 id FROM dbo.passport_ocr_records WHERE id < ? ORDER BY id DESC",
        record_id,
    )
    row = cursor.fetchone()
    return int(row[0]) if row else None


def get_next_record_id(cursor: pyodbc.Cursor, record_id: int) -> int | None:
    cursor.execute(
        "SELECT TOP 1 id FROM dbo.passport_ocr_records WHERE id > ? ORDER BY id ASC",
        record_id,
    )
    row = cursor.fetchone()
    return int(row[0]) if row else None


def insert_record(
    cursor: pyodbc.Cursor,
    image_path: Path,
    extracted_json: str,
    raw_ocr_text: str,
) -> int:
    cursor.execute(
        """
        INSERT INTO dbo.passport_ocr_records (
            image_path,
            extracted_json,
            reviewed_json,
            status,
            raw_ocr_text,
            error_message
        )
        OUTPUT INSERTED.id
        VALUES (?, ?, NULL, ?, ?, NULL)
        """,
        str(image_path.resolve()),
        extracted_json,
        "ocr_done",
        raw_ocr_text,
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Could not insert OCR record for image: {image_path}")
    return int(row[0])


def insert_error_record(cursor: pyodbc.Cursor, image_path: Path, error_message: str) -> None:
    cursor.execute(
        """
        INSERT INTO dbo.passport_ocr_records (
            image_path,
            extracted_json,
            reviewed_json,
            status,
            raw_ocr_text,
            error_message
        )
        VALUES (?, NULL, NULL, ?, NULL, ?)
        """,
        str(image_path.resolve()),
        "error",
        error_message,
    )


def update_reviewed_record(
    cursor: pyodbc.Cursor,
    *,
    record_id: int,
    reviewed_json: str,
    status: str,
) -> None:
    cursor.execute(
        """
        UPDATE dbo.passport_ocr_records
        SET reviewed_json = ?, status = ?, updated_at = GETDATE()
        WHERE id = ?
        """,
        reviewed_json,
        status,
        record_id,
    )


def update_layoutlm_json(
    cursor: pyodbc.Cursor,
    *,
    record_id: int,
    layoutlm_json: str,
) -> None:
    cursor.execute(
        """
        UPDATE dbo.passport_ocr_records
        SET layoutlm_json = ?, updated_at = GETDATE()
        WHERE id = ?
        """,
        layoutlm_json,
        record_id,
    )


def update_layoutlm_reviewed_json(
    cursor: pyodbc.Cursor,
    *,
    record_id: int,
    layoutlm_reviewed_json: str,
) -> None:
    cursor.execute(
        """
        UPDATE dbo.passport_ocr_records
        SET layoutlm_reviewed_json = ?, updated_at = GETDATE()
        WHERE id = ?
        """,
        layoutlm_reviewed_json,
        record_id,
    )


def list_records_for_layoutlm(cursor: pyodbc.Cursor) -> list[pyodbc.Row]:
    cursor.execute(
        """
        SELECT
            id,
            image_path,
            extracted_json,
            reviewed_json,
            layoutlm_json,
            layoutlm_reviewed_json,
            status
        FROM dbo.passport_ocr_records
        ORDER BY id
        """
    )
    return list(cursor.fetchall())
