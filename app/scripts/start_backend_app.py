from __future__ import annotations

import logging
import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_api_host, get_api_port, get_log_dir


def _configure_logging() -> Path:
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "backend_api.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return log_file_path


def main() -> None:
    log_file_path = _configure_logging()
    host = get_api_host()
    port = get_api_port()

    logging.info("Starting Passport OCR backend")
    logging.info("Host: %s", host)
    logging.info("Port: %s", port)
    logging.info("Log file: %s", log_file_path)

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
