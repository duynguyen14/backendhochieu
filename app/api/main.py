from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.code_values import router as code_values_router
from app.api.routes.passport_records import router as passport_records_router
from app.config import get_frontend_allowed_origins


app = FastAPI(title="Passport OCR Review API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_allowed_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(passport_records_router, prefix="/api")
app.include_router(code_values_router, prefix="/api")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
