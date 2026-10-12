"""工具层 + 智能体循环。

这一组测试的价值在于：整条「检索 → 工具调用 → 回灌 → 生成回答」的链路
在没有任何 API 密钥的情况下被完整验证。CI 里跑的就是这些。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app import runtime
from app.agent import tools as toolkit
from app.agent.llm import StubChatModel
from app.agent.loop import Agent, clean_citations, citation_indices
from app.config import CORPUS_DIR
from app.kb.corpus import build_retriever
from app.kb.embedder import HashingEmbedder
from app.kb.retriever import Retriever
from app.kb.chunker import split_corpus
from app.store import db

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.seed(connection, now=NOW)
    return connection


@pytest.fixture(autouse=True)
def wired(conn):
    runtime.set_connection(conn)
    # 工具默认用真实系统时间。不固定住的话，「签收后第 7 天」这类断言会随
    # 运行时刻漂移——本机和 UTC 差 8 小时，跨不跨天都能让结果变。
    runtime.set_clock(lambda: NOW)
    yield
    runtime.reset()


@pytest.fixture(scope="module")
def retriever():
    chunks = split_corpus(CORPUS_DIR)
    assert chunks, "语料没加载上，后面的测试就没意义了"
    return Retriever(chunks, HashingEmbedder(dim=1024), top_k=5)


def call(tool_obj, **kwargs):
    """调用工具并解析它返回的 JSON。"""
    raw = tool_obj.invoke(kwargs)
    return json.loads(raw)


# ---------------------------------------------------------------- 参数归一化


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SO20261001001", "SO20261001001"),
        ("so20261001001", "SO20261001001"),
        (" SO-2026-1001001 ", "SO20261001001"),
        ("SO 2026 1001 001", "SO20261001001"),
    ],
)
def test_order_id_normalization(raw, expected):
    # 模型把订单号写成小写、带空格或连字符是常态，不该因此查不到
    assert toolkit.normalize_order_id(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("no_reason", "no_reason"),
        ("无理由", "no_reason"),
        ("7天无理由", "no_reason"),
        ("quality", "quality"),
        ("质量问题", "quality"),
    ],
)
def test_reason_normalization(raw, expected):
    assert toolkit.normalize_reason(raw) == expected


def test_reason_normalization_rejects_garbage():
    with pytest.raises(ValueError, match="退货原因"):
        toolkit.normalize_reason("看心情")


def test_tool_registry_is_consistent():
    names = [t.name for t in toolkit.TOOLS]
    assert len(names) == len(set(names)) == 6
    assert set(names) == set(toolkit.TOOL_MAP)


# ---------------------------------------------------------------- 订单查询


def test_query_order_returns_amounts_and_items():
    data = call(toolkit.query_order, order_id="SO20261001001")
    assert data["ok"] is True
    assert data["paid_yuan"] == "249.00"
    assert data["items"][0]["sku"] == "SKU-BED-18"
    assert "纯棉四件套" in data["summary"]


def test_query_order_lowercase_id_works():
    assert call(toolkit.query_order, order_id="so20261001001")["ok"] is True


def test_unknown_order_does_not_leak_other_orders():
    # 这是客服机器人：把别人的订单号说给用户听是事故，不是小毛病
    data = call(toolkit.query_order, order_id="SO99999999999")
    assert data["ok"] is False
    assert "核对" in data["summary"]
    for other in ("SO20261001001", "SO20261009011", "SO20261012006"):
        assert other not in json.dumps(data, ensure_ascii=False)


def test_query_logistics_returns_tracking():
    data = call(toolkit.query_logistics, order_id="SO20261001001")
    assert data["ok"] is True
    assert data["carrier"] == "顺丰速运"
    assert data["delivered"] is True


def test_query_logistics_reports_unshipped_order():
    data = call(toolkit.query_logistics, order_id="SO20261010005")
    assert data["ok"] is False
    assert "没有物流记录" in data["summary"]


# ---------------------------------------------------------------- 退货资格


def test_eligibility_within_window():
    data = call(toolkit.check_return_eligibility, order_id="SO20261001001", reason="无理由")
    assert data["ok"] is True
    assert data["eligible"] is True
    assert data["window_days"] == 7
    assert data["days_left"] == 5


def test_eligibility_blocked_for_custom_item():
    data = call(toolkit.check_return_eligibility, order_id="SO20261005003", reason="no_reason")
    assert data["eligible"] is False
    assert data["blocked_items"]


def test_eligibility_reports_expiry():
    data = call(toolkit.check_return_eligibility, order_id="SO20260915008", reason="quality")
    assert data["eligible"] is False
    assert data["expired"] is True
    assert "保修" in data["summary"]


def test_eligibility_rejects_unknown_reason_without_raising():
    data = call(toolkit.check_return_eligibility, order_id="SO20261001001", reason="想退")
    assert data["ok"] is False
    assert "退货原因" in data["summary"]


# ---------------------------------------------------------------- 退款


def test_calc_refund_detail():
    data = call(toolkit.calc_refund, order_id="SO20261001001", reason="no_reason")
    assert data["refund_yuan"] == "241.00"
    assert data["deduction_yuan"] == "8.00"
    labels = [line["label"] for line in data["lines"]]
    assert any("纯棉四件套" in label for label in labels)
    assert any("扣回发货运费" in label for label in labels)


def test_calc_refund_for_single_sku_of_two_item_order():
    data = call(toolkit.calc_refund, order_id="SO20261009011", reason="no_reason",
                skus=["SKU-BED-18"])
    assert data["goods_paid_cents"] == 22802


def test_calc_refund_reports_unknown_sku():
    data = call(toolkit.calc_refund, order_id="SO20261009011", reason="no_reason",
                skus=["SKU-NOPE"])
    assert data["missing_skus"] == ["SKU-NOPE"]
    assert data["refund_cents"] == 0


def test_calc_refund_of_expired_order_is_zero():
    data = call(toolkit.calc_refund, order_id="SO20260915008", reason="no_reason")
    assert data["refund_cents"] == 0
    assert data["lines"] == []


# ---------------------------------------------------------------- 换货 / 赔付


def test_check_exchange_ok():
    data = call(toolkit.check_exchange, order_id="SO20261001001")
    assert data["applicable"] is True
    assert data["items"][0]["can_exchange"] is True


def test_check_exchange_blocked_by_stock():
    data = call(toolkit.check_exchange, order_id="SO20261012006")
    assert data["applicable"] is False
    assert "只能选择退货" in data["items"][0]["blocker"]


def test_late_ship_compensation_amount():
    data = call(toolkit.calc_late_ship_compensation, order_id="SO20261010005")
    assert data["applicable"] is True
    assert data["amount_yuan"] == "6.95"


def test_late_ship_compensation_not_applicable_after_shipping():
    data = call(toolkit.calc_late_ship_compensation, order_id="SO20261001001")
    assert data["applicable"] is False


# ---------------------------------------------------------------- 引用校验


def test_citation_indices_are_deduped_and_sorted():
    assert citation_indices("见 [2] 和 [1]，还有 [2]") == [1, 2]


def test_clean_citations_drops_out_of_range():
    text, bogus = clean_citations("按政策 [1] 可以退，但 [9] 说不清。", total=3)
    assert bogus == [9]
    assert "[9]" not in text
    assert "[1]" in text


def test_clean_citations_keeps_valid_ones():
    text, bogus = clean_citations("依据 [1][2]。", total=2)
    assert bogus == []
    assert text == "依据 [1][2]。"


# ---------------------------------------------------------------- 假模型


class FixedModel(StubChatModel):
    """不管问什么都返回同一句话，用来验证循环的护栏。"""

    canned: str = ""
    calls: list = []

    def _next(self, messages):  # type: ignore[override]
        return AIMessage(content=self.canned)


class LoopModel(StubChatModel):
    """永远要求调用同一个工具，用来验证步数上限。"""

    def _next(self, messages):  # type: ignore[override]
        return AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "SO20261001001"},
                         "id": "call_loop", "type": "tool_call"}],
        )


# ---------------------------------------------------------------- 主循环


def agent(retriever, model=None, **kwargs) -> Agent:
    return Agent(
        model or StubChatModel(),
        retriever,
        toolkit.TOOLS,
        refusal_threshold=0.35,
        **kwargs,
    )


def test_policy_question_answers_from_knowledge_base(retriever):
    answer = agent(retriever).run("7 天无理由退货的期限是几天？")
    assert answer.tool_calls == []
    assert answer.text
    assert answer.citations, "政策类问题必须带上来源引用"
    assert all(1 <= c["index"] <= len(answer.hits) for c in answer.citations)
    assert answer.refused is False


def test_order_question_triggers_refund_tool(retriever):
    answer = agent(retriever).run("SO20261001001 这个订单无理由退货能退多少钱？")
    assert [c["name"] for c in answer.tool_calls] == ["calc_refund"]
    assert answer.tool_calls[0]["ok"] is True
    assert answer.tool_calls[0]["data"]["refund_yuan"] == "241.00"


def test_exchange_question_picks_exchange_tool(retriever):
    answer = agent(retriever).run("SO20261012006 我想换个同款，可以吗")
    assert answer.tool_calls and answer.tool_calls[0]["name"] == "check_exchange"


def test_logistics_question_picks_logistics_tool(retriever):
    answer = agent(retriever).run("SO20261001001 的快递到哪了")
    assert answer.tool_calls and answer.tool_calls[0]["name"] == "query_logistics"


def test_late_ship_question_is_not_hijacked_by_logistics(retriever):
    # 「说好 48 小时发货怎么还没发」同时命中「发货」和「超时」，
    # 意图顺序排错的话会被物流工具抢走
    answer = agent(retriever).run("SO20261010005 说好 48 小时发货，怎么到现在还没发，有赔付吗")
    assert answer.tool_calls and answer.tool_calls[0]["name"] == "calc_late_ship_compensation"


def test_quality_reason_is_detected(retriever):
    answer = agent(retriever).run("SO20261012006 商品有质量问题，能退多少")
    assert answer.tool_calls[0]["args"]["reason"] == "quality"


def test_off_topic_question_is_refused(retriever):
    answer = agent(retriever).run("怎么申请提高信用卡额度")
    assert answer.refused is True
    assert answer.tool_calls == []


def test_unshipped_order_flow_answers_without_raising(retriever):
    answer = agent(retriever).run("SO20261010005 我不想要了，能退吗")
    assert answer.tool_calls
    assert answer.tool_calls[0]["ok"] is True


def test_events_come_in_order(retriever):
    events = [e.type for e in agent(retriever).stream("SO20261001001 快递到哪了")]
    assert events[0] == "retrieval"
    assert "tool_call" in events and "tool_result" in events
    assert events.index("tool_result") < events.index("answer")
    assert events[-1] == "done"


def test_answer_deltas_join_into_the_final_answer(retriever):
    pieces = []
    final = ""
    for event in agent(retriever).stream("7 天无理由退货怎么算"):
        if event.type == "answer_delta":
            pieces.append(event.data["text"])
        elif event.type == "answer":
            final = event.data["answer"]
    assert final == "".join(pieces)


def test_bogus_citation_is_dropped_and_reported(retriever):
    model = FixedModel(canned="根据政策 [1] 可以退，另外 [9] 也提到了。")
    events = list(agent(retriever, model).stream("7 天无理由退货"))
    warning = [e for e in events if e.type == "citation_warning"]
    assert warning and warning[0].data["dropped"] == [9]
    final = [e for e in events if e.type == "answer"][0].data
    assert "[9]" not in final["answer"]
    assert final["citations"][0]["index"] == 1


def test_max_steps_stops_runaway_tool_calls(retriever):
    answer = agent(retriever, LoopModel(), max_steps=3).run("SO20261001001 帮我查一下")
    assert len(answer.tool_calls) == 3
    assert "转人工" in answer.text


def test_tool_exception_is_fed_back_not_raised(retriever):
    @tool
    def broken(order_id: str) -> str:
        """故意坏掉的工具。"""
        raise RuntimeError("模拟工具崩溃")

    instance = Agent(StubChatModel(), retriever, [broken])
    raw, parsed = instance._run_tool({"name": "broken", "args": {"order_id": "SO1"}})
    assert parsed["ok"] is False
    assert "模拟工具崩溃" in parsed["summary"]
    assert json.loads(raw)["ok"] is False


def test_unknown_tool_name_is_reported():
    instance = Agent(StubChatModel(), Retriever([], HashingEmbedder(dim=64)), toolkit.TOOLS)
    _, parsed = instance._run_tool({"name": "drop_database", "args": {}})
    assert parsed["ok"] is False
    assert "没有名为" in parsed["summary"]


def test_agent_requires_at_least_one_tool(retriever):
    with pytest.raises(ValueError, match="工具"):
        Agent(StubChatModel(), retriever, [])


def test_history_is_accepted(retriever):
    history = [
        {"role": "user", "content": "SO20261001001 是什么状态"},
        {"role": "assistant", "content": "已签收。"},
    ]
    answer = agent(retriever).run("那能退吗", history)
    assert answer.text
