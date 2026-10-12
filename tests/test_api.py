"""HTTP 接口。

走的是真实的 ASGI 栈（TestClient），但底下接的是内存数据库和离线假模型，
所以 CI 里既不需要密钥也不需要网络。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import runtime
from app.agent import tools as toolkit
from app.agent.llm import StubChatModel
from app.agent.loop import Agent
from app.config import CORPUS_DIR
from app.kb.chunker import split_corpus
from app.kb.embedder import HashingEmbedder
from app.kb.retriever import Retriever
from app.main import create_app
from app.store import db

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


class BoomAgent:
    """模拟模型侧彻底失败，用来验证接口的错误路径。"""

    def run(self, *args, **kwargs):
        raise RuntimeError("模型连接超时")

    def stream(self, *args, **kwargs):
        raise RuntimeError("模型连接超时")


@pytest.fixture(scope="module")
def retriever():
    return Retriever(split_corpus(CORPUS_DIR), HashingEmbedder(dim=1024), top_k=5)


@pytest.fixture
def client(retriever):
    connection = db.connect(":memory:")
    db.seed(connection, now=NOW)
    runtime.set_connection(connection)
    runtime.set_clock(lambda: NOW)
    runtime.set_retriever(retriever)
    runtime.set_agent(Agent(StubChatModel(), retriever, toolkit.TOOLS))
    with TestClient(create_app()) as test_client:
        yield test_client
    runtime.reset()


def frames_of(client, question: str) -> list[dict]:
    frames: list[dict] = []
    with client.stream("POST", "/api/chat/stream", json={"question": question}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[len("data: "):]))
    return frames


# ---------------------------------------------------------------- 基础接口


def test_health_reports_corpus_and_tools(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["corpus"]["documents"] == 8
    assert data["corpus"]["chunks"] > 10
    assert len(data["tools"]) == 6
    assert data["retrieval"]["backend"] in ("faiss", "numpy")


def test_health_does_not_leak_api_key(client):
    data = client.get("/api/health").json()
    assert data["llm"]["provider"] == "stub"
    assert data["llm"]["api_key_configured"] is False
    # 健康检查只允许暴露「配没配」，不许把密钥字段本身带出来
    assert set(data["llm"]) == {"provider", "model", "base_url", "api_key_configured"}


def test_tools_endpoint_describes_parameters(client):
    tools = client.get("/api/tools").json()
    assert len(tools) == 6
    by_name = {tool["name"]: tool for tool in tools}
    assert "order_id" in by_name["query_order"]["parameters"]
    assert "reason" in by_name["calc_refund"]["parameters"]
    assert all(tool["description"] for tool in tools)


def test_demo_orders_endpoint(client):
    orders = client.get("/api/demo/orders").json()
    assert len(orders) == 11
    first = next(order for order in orders if order["order_id"] == "SO20261001001")
    assert first["status_label"] == "已签收"
    assert first["paid_yuan"] == "249.00"
    assert first["signed_days_ago"] == 2


# ---------------------------------------------------------------- 非流式问答


def test_chat_policy_question(client):
    data = client.post("/api/chat", json={"question": "7 天无理由退货的期限是几天？"}).json()
    assert data["answer"]
    assert data["citations"]
    assert data["refused"] is False


def test_chat_order_question_uses_tool(client):
    data = client.post("/api/chat", json={
        "question": "SO20261001001 无理由退货能退多少钱？"
    }).json()
    assert data["tool_calls"][0]["name"] == "calc_refund"
    assert data["tool_calls"][0]["data"]["refund_yuan"] == "241.00"


def test_chat_off_topic_is_refused(client):
    data = client.post("/api/chat", json={"question": "怎么申请提高信用卡额度"}).json()
    assert data["refused"] is True


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_blank_question_is_rejected(client, question):
    assert client.post("/api/chat", json={"question": question}).status_code == 422


def test_overlong_question_is_rejected(client):
    assert client.post("/api/chat", json={"question": "退" * 501}).status_code == 422


def test_too_much_history_is_rejected(client):
    history = [{"role": "user", "content": "在吗"}] * 11
    response = client.post("/api/chat", json={"question": "能退吗", "history": history})
    assert response.status_code == 422


def test_history_is_accepted(client):
    history = [
        {"role": "user", "content": "SO20261001001 是什么状态"},
        {"role": "assistant", "content": "已签收。"},
    ]
    response = client.post("/api/chat", json={"question": "那能退吗", "history": history})
    assert response.status_code == 200


def test_model_failure_returns_502_not_500(client):
    runtime.set_agent(BoomAgent())
    response = client.post("/api/chat", json={"question": "能退吗"})
    assert response.status_code == 502
    assert "模型连接超时" in response.json()["detail"]


# ---------------------------------------------------------------- 流式问答


def test_stream_frame_order(client):
    frames = frames_of(client, "SO20261001001 快递到哪了")
    types = [frame["type"] for frame in frames]
    assert types[0] == "retrieval"
    assert types[-1] == "done"
    assert types.index("tool_result") < types.index("answer")
    assert frames[-1] == {"type": "done"}


def test_stream_deltas_rebuild_the_answer(client):
    frames = frames_of(client, "7 天无理由退货怎么算")
    deltas = "".join(f["text"] for f in frames if f["type"] == "answer_delta")
    final = next(f for f in frames if f["type"] == "answer")
    assert deltas == final["answer"]
    assert final["citations"]


def test_stream_retrieval_frame_carries_hits(client):
    frames = frames_of(client, "退货运费谁承担")
    retrieval = frames[0]
    assert retrieval["hits"], "政策类问题应该检索到条款"
    assert retrieval["hits"][0]["index"] == 1
    assert "label" in retrieval["hits"][0] and "text" in retrieval["hits"][0]


def test_stream_model_failure_emits_error_frame(client):
    runtime.set_agent(BoomAgent())
    frames = frames_of(client, "能退吗")
    types = [frame["type"] for frame in frames]
    assert "error" in types
    assert types[-1] == "done"
    error = next(f for f in frames if f["type"] == "error")
    assert "模型连接超时" in error["message"]


def test_stream_rejects_blank_question(client):
    response = client.post("/api/chat/stream", json={"question": "  "})
    assert response.status_code == 422
