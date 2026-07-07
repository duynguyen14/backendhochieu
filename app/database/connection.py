from __future__ import annotations

import pyodbc

from app.config import get_sql_server_connection_string


def get_connection_string() -> str:
    return get_sql_server_connection_string()


def get_connection() -> pyodbc.Connection:
    return pyodbc.connect(get_connection_string())
