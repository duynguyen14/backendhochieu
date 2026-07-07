from __future__ import annotations

import pyodbc


def list_country_code_values(cursor: pyodbc.Cursor) -> list[pyodbc.Row]:
    cursor.execute(
        """
        SELECT
            CodeValue,
            CodeValueDes
        FROM dbo.codevalues
        WHERE UPPER(CodeId) = 'COUNTRY'
        ORDER BY CodeValueDes, CodeValue
        """
    )
    return list(cursor.fetchall())
