from __future__ import annotations

from fastapi import APIRouter

from app.services.code_value_service import get_country_options


router = APIRouter(tags=["code-values"])


@router.get("/code-values/countries")
def get_countries():
    return {"items": get_country_options()}
