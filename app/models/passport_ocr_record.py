from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExistingRecord:
    id: int
    image_path: str
    normalized_path: str
    file_name: str
