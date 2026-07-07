from __future__ import annotations


def rename_images(*args, **kwargs):
    from .image_rename_service import rename_images as _rename_images

    return _rename_images(*args, **kwargs)


def copy_images_flattened(*args, **kwargs):
    from .image_rename_service import copy_images_flattened as _copy_images_flattened

    return _copy_images_flattened(*args, **kwargs)


def process_images_to_database(*args, **kwargs):
    from .ocr_service import process_images_to_database as _process_images_to_database

    return _process_images_to_database(*args, **kwargs)


__all__ = ["copy_images_flattened", "process_images_to_database", "rename_images"]
