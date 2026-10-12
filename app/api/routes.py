"""HTTP 路由。

接口刻意都写成同步的（def，不是 async def）。模型调用、SQLite 查询都是阻塞操作，
放在 async 函数里会卡住整个事件循环；交给 FastAPI 自动丢线程池反而更简单，
也省掉了 pytest-asyncio 这一层依赖。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import __version__, policy, runtime
from ..agent import tools as toolkit
from ..config import CORPUS_DIR
from ..kb.corpus import load_chunks
from ..store import queries
from .schemas import ChatRequest

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 有反向代理时不禁用缓冲，前端会等到全部生成完才一次性收到
    "X-Accel-Buffering": "no",
}


def sse_frame(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/health")
def health() -> dict:
    cfg = runtime.settings()
    view = cfg.public_view()
    documents = len(list(CORPUS_DIR.glob("*.md"))) if CORPUS_DIR.is_dir() else 0
    chunks = 0
    backend = "none"
    try:
        retriever = runtime.get_retriever()
        chunks = len(retriever.chunks)
        backend = retriever.backend
    except Exception as exc:  # noqa: BLE001 —— 健康检查本身不能因为语料缺失就 500
        backend = f"unavailable: {type(exc).__name__}"
    return {
        "status": "ok",
        "version": __version__,
        "llm": {
            "provider": view["llm_provider"],
            "model": view["llm_model"],
            "base_url": view["llm_base_url"],
            "api_key_configured": view["api_key_configured"],
        },
        "corpus": {"documents": documents, "chunks": chunks},
        "embedding": view["embedding"],
        "tools": [tool.name for tool in toolkit.TOOLS],
        "retrieval": {
            "backend": backend,
            "top_k": view["top_k"],
            "refusal_threshold": view["refusal_threshold"],
        },
    }


@router.get("/tools")
def list_tools() -> list[dict]:
    out = []
    for tool in toolkit.TOOLS:
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        parameters = {
            name: (spec.get("description") or spec.get("type", ""))
            for name, spec in (schema.get("properties") or {}).items()
        }
        out.append({
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": parameters,
        })
    return out


@router.get("/demo/orders")
def demo_orders() -> list[dict]:
    """演示用的订单列表，前端拿它拼示例问题。"""
    conn = runtime.get_connection()
    now = runtime.now()
    out = []
    for row in queries.list_orders(conn):
        order = queries.get_order(conn, row["order_id"])
        out.append({
            "order_id": row["order_id"],
            "status": row["status"],
            "status_label": toolkit.STATUS_LABEL.get(row["status"], row["status"]),
            "paid_yuan": policy.yuan(int(row["paid_cents"])),
            "receiver_city": row["receiver_city"],
            "signed_days_ago": policy.calendar_days_since(order.get("signed_at"), now),
        })
    return out


@router.post("/chat")
def chat(payload: ChatRequest) -> dict:
    history = [turn.model_dump() for turn in payload.history]
    try:
        answer = runtime.get_agent().run(payload.question, history)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"生成回答失败：{type(exc).__name__}: {exc}",
        ) from exc
    return answer.to_dict()


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    history = [turn.model_dump() for turn in payload.history]
    question = payload.question

    def frames():
        # Agent 在生成器里才创建：配置不对（比如没填密钥）时，
        # 这样还能发出一帧 error 事件，而不是让响应头都没发出去就 500
        try:
            agent = runtime.get_agent()
            for event in agent.stream(question, history):
                yield sse_frame(event.to_dict())
        except Exception as exc:  # noqa: BLE001
            yield sse_frame({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            yield sse_frame({"type": "done"})

    return StreamingResponse(frames(), media_type="text/event-stream", headers=SSE_HEADERS)
