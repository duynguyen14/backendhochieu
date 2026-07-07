from __future__ import annotations

from app.database import get_connection
from app.repositories import list_country_code_values


def get_country_options() -> list[dict[str, str]]:
    with get_connection() as connection:
        cursor = connection.cursor()
        rows = list_country_code_values(cursor)

    options: list[dict[str, str]] = []
    for row in rows:
        value = str(row.CodeValue or "").strip()
        label = str(row.CodeValueDes or value).strip()
        options.append(
            {
                "value": value,
                "label": f"{value} - {label}" if value else label,
            }
        )

    return options
