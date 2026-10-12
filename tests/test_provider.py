"""非桩模型（ChatOpenAI）那条代码路径。

默认配置用离线桩，所以这条路径在 CI 里**从来没被执行过**：
工具 schema 的转换、流式分片的累积、ToolMessage 回灌、DeepSeek 的 base_url 拼接，
全都只在配了密钥之后才会走到。等面试现场配好密钥那一刻才发现问题就晚了。

这里用 httpx 的 MockTransport 在传输层拦下请求，喂进去的是 OpenAI 兼容接口
真实的报文形状（SSE 分片、tool_calls 分片累积），所以验证的是真实的解析代码，
不是我自己造的一个假接口。
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app import runtime
from app.agent import tools as toolkit
from app.agent.llm import build_model
from app.agent.loop import Agent
from app.config import Settings
from app.kb.chunker import split_corpus
from app.kb.embedder import HashingEmbedder
from app.kb.retriever import Retriever
from app.store import db

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def retriever():
    return Retriever(split_corpus(__import__("app.config", fromlist=["x"]).CORPUS_DIR),
                     HashingEmbedder(dim=512), top_k=5)


@pytest.fixture(autouse=True)
def wired():
    conn = db.connect(":memory:")
    db.seed(conn, now=NOW)
    runtime.set_connection(conn)
    runtime.set_clock(lambda: NOW)
    yield
    runtime.reset()


# ------------------------------------------------------------------ 报文构造


def _frame(delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def tool_call_stream(name: str, arguments: dict) -> str:
    """把工具调用拆成多个分片——真实的 OpenAI 接口就是一片一片吐参数的。"""
    raw = json.dumps(arguments, ensure_ascii=False)
    head, tail = raw[: len(raw) // 2], raw[len(raw) // 2 :]
    return "".join([
        _frame({"role": "assistant", "content": ""}),
        _frame({"tool_calls": [{"index": 0, "id": "call_test1", "type": "function",
                                "function": {"name": name, "arguments": ""}}]}),
        _frame({"tool_calls": [{"index": 0, "function": {"arguments": head}}]}),
        _frame({"tool_calls": [{"index": 0, "function": {"arguments": tail}}]}),
        _frame({}, "tool_calls"),
        "data: [DONE]\n\n",
    ])


def text_stream(*pieces: str) -> str:
    frames = [_frame({"role": "assistant", "content": ""})]
    frames += [_frame({"content": piece}) for piece in pieces]
    frames.append(_frame({}, "stop"))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def mock_model(responder):
    """responder(请求体) -> SSE 文本。返回 (模型, 请求体列表, 请求对象列表)。"""
    bodies: list[dict] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        requests.append(request)
        return httpx.Response(
            200,
            text=responder(bodies[-1], len(bodies)),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="deepseek-flash",
        api_key="sk-not-a-real-key",
        base_url="https://api.deepseek.com/v1",
        http_client=client,
        max_retries=0,
    )
    return model, bodies, requests


def has_tool_result(body: dict) -> bool:
    return any(message.get("role") == "tool" for message in body.get("messages", []))


# ------------------------------------------------------------------ build_model


def test_build_model_returns_chatopenai_for_deepseek():
    settings = replace(Settings.from_env(), llm_provider="deepseek",
                       deepseek_api_key="sk-abc", deepseek_model="deepseek-flash")
    model = build_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-flash"
    assert str(model.openai_api_base).rstrip("/") == "https://api.deepseek.com/v1"


def test_build_model_refuses_to_run_without_a_key():
    settings = replace(Settings.from_env(), llm_provider="deepseek", deepseek_api_key="")
    with pytest.raises(RuntimeError, match="密钥"):
        build_model(settings)


def test_build_model_keeps_stub_as_default():
    from app.agent.llm import StubChatModel

    assert isinstance(build_model(replace(Settings.from_env(), llm_provider="stub")), StubChatModel)


# ------------------------------------------------------------------ 工具调用往返


def test_tool_call_round_trip_against_openai_shaped_stream(retriever):
    model, bodies, headers = mock_model(
        lambda body, n: tool_call_stream("calc_refund", {"order_id": "SO20261001001", "reason": "no_reason"})
        if not has_tool_result(body)
        else text_stream("依据「7 天无理由退货」，", "应退 241.00 元 [1]。")
    )
    answer = Agent(model, retriever, toolkit.TOOLS).run("SO20261001001 无理由退货能退多少钱？")

    assert len(bodies) == 2, "应该先要工具、拿到结果后再生成回答"
    assert answer.tool_calls and answer.tool_calls[0]["name"] == "calc_refund"
    assert answer.tool_calls[0]["data"]["refund_yuan"] == "241.00"
    assert answer.text == "依据「7 天无理由退货」，应退 241.00 元 [1]。"
    assert [c["index"] for c in answer.citations] == [1]


def test_first_request_carries_the_real_tool_schemas(retriever):
    model, bodies, _ = mock_model(
        lambda body, n: tool_call_stream("query_order", {"order_id": "SO20261001001"})
        if not has_tool_result(body) else text_stream("好的。")
    )
    Agent(model, retriever, toolkit.TOOLS).run("SO20261001001 什么状态")

    sent = bodies[0]
    names = [tool["function"]["name"] for tool in sent["tools"]]
    assert set(names) == set(toolkit.TOOL_MAP)
    # 参数说明必须真的传给了模型，否则它不知道 reason 该填什么
    refund = next(t for t in sent["tools"] if t["function"]["name"] == "calc_refund")
    assert "order_id" in refund["function"]["parameters"]["properties"]
    assert "no_reason" in refund["function"]["description"]


def test_tool_result_is_fed_back_as_a_tool_message(retriever):
    model, bodies, _ = mock_model(
        lambda body, n: tool_call_stream("calc_refund", {"order_id": "SO20261001001", "reason": "no_reason"})
        if not has_tool_result(body) else text_stream("好了。")
    )
    Agent(model, retriever, toolkit.TOOLS).run("SO20261001001 能退多少")

    tool_messages = [m for m in bodies[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["refund_yuan"] == "241.00"
    assert tool_messages[0]["tool_call_id"] == "call_test1"


def test_request_goes_to_the_deepseek_compatible_path_with_bearer_key(retriever):
    model, _, requests = mock_model(lambda body, n: text_stream("好的。"))
    Agent(model, retriever, toolkit.TOOLS).run("7 天无理由退货是几天")

    # base_url 少了 /v1 是接 DeepSeek 最常见的错，这里把它钉死
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer sk-not-a-real-key"


def test_streaming_pieces_are_concatenated_in_order(retriever):
    model, _, _ = mock_model(lambda body, n: text_stream("第一段，", "第二段，", "第三段。"))
    answer = Agent(model, retriever, toolkit.TOOLS).run("7 天无理由退货是几天")
    assert answer.text == "第一段，第二段，第三段。"


def test_parallel_tool_calls_in_one_turn_are_all_executed(retriever):
    def two_calls() -> str:
        raw_a = json.dumps({"order_id": "SO20261001001"})
        raw_b = json.dumps({"order_id": "SO20261001001"})
        return "".join([
            _frame({"role": "assistant", "content": ""}),
            _frame({"tool_calls": [
                {"index": 0, "id": "call_a", "type": "function",
                 "function": {"name": "query_order", "arguments": raw_a}},
                {"index": 1, "id": "call_b", "type": "function",
                 "function": {"name": "query_logistics", "arguments": raw_b}},
            ]}),
            _frame({}, "tool_calls"),
            "data: [DONE]\n\n",
        ])

    model, bodies, _ = mock_model(
        lambda body, n: two_calls() if not has_tool_result(body) else text_stream("都查到了。")
    )
    answer = Agent(model, retriever, toolkit.TOOLS).run("SO20261001001 状态和物流")

    assert [c["name"] for c in answer.tool_calls] == ["query_order", "query_logistics"]
    tool_messages = [m for m in bodies[1]["messages"] if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_a", "call_b"}


def test_api_error_is_surfaced_not_swallowed(retriever):
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Authentication Fails"}})

    client = httpx.Client(transport=httpx.MockTransport(failing))
    model = ChatOpenAI(model="deepseek-flash", api_key="sk-bad",
                       base_url="https://api.deepseek.com/v1", http_client=client, max_retries=0)
    with pytest.raises(Exception) as info:
        Agent(model, retriever, toolkit.TOOLS).run("7 天无理由退货是几天")
    assert "401" in str(info.value) or "Authentication" in str(info.value)
