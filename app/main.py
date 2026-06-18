from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api.routes import router as api_router
from .storage import db

app = FastAPI(title="Bilingualism Test")
app.include_router(api_router)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


def _page(name: str) -> FileResponse:
    return FileResponse(config.STATIC_DIR / name)


@app.get("/")
def index():
    return _page("index.html")


@app.get("/test")
def test_page():
    return _page("test.html")


@app.get("/result")
def result_page():
    return _page("result.html")


@app.get("/info")
def info_page():
    return _page("info.html")


@app.get("/admin")
def admin_page():
    return _page("admin.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
