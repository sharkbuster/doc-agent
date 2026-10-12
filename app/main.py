"""FastAPI 入口。

本地起服务：
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
然后打开 http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import router

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 内联一个 favicon，省掉一次 404，也省掉一个静态文件
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#1a7f6b"/>'
    '<path d="M9 17.5l4.5 4.5L23 12" stroke="#fff" stroke-width="3" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

DESCRIPTION = """电商售后客服智能体。

先检索平台售后政策条款，再按需调用工具查真实订单数据，最后基于条款和数据生成回答。
回答里的 [n] 是引用编号，对应检索到的条款；系统会校验编号是否存在，
模型编造的引用会被丢弃而不是展示给用户。
"""


def create_app() -> FastAPI:
    app = FastAPI(title="拾光优选 · 售后客服智能体", version=__version__, description=DESCRIPTION)
    app.include_router(router, prefix="/api")

    # 目录不存在时 StaticFiles 会直接抛异常，整个应用起不来。
    # 前端还没就位时也让它先能启动，接口是可以单独用的。
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(FAVICON, media_type="image/svg+xml")

    return app


app = create_app()
