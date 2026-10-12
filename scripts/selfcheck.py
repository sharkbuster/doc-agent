"""端到端自检：一条命令确认整个项目是活的。

    .venv\\Scripts\\python.exe -m scripts.selfcheck

不需要网络，也不需要任何密钥——用离线桩模型跑通全链路。
任何一项失败都会返回非 0，方便挂到 CI 或者部署后做冒烟。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 中文 Windows 的控制台默认是 GBK，输出里只要有一个它编码不了的字符，
# 整个脚本就会以 UnicodeEncodeError 崩掉——而这时候自检其实已经全过了。
# 宁可把那个字符变成问号，也不要让结果丢掉。
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from fastapi.testclient import TestClient  # noqa: E402

from app import policy, runtime  # noqa: E402
from app.agent import tools as toolkit  # noqa: E402
from app.agent.llm import StubChatModel  # noqa: E402
from app.agent.loop import Agent  # noqa: E402
from app.config import CORPUS_DIR  # noqa: E402
from app.kb.corpus import load_chunks  # noqa: E402
from app.main import create_app  # noqa: E402
from app.store import db, queries  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(label)
        print(f"  [OK]   {label}{('  ' + detail) if detail else ''}")
    else:
        FAILED.append(label)
        print(f"  [FAIL] {label}{('  ' + detail) if detail else ''}")


def main() -> int:
    print("=" * 62)
    print("端到端自检")
    print("=" * 62)

    print("\n[1] 知识库")
    documents = sorted(CORPUS_DIR.glob("*.md")) if CORPUS_DIR.is_dir() else []
    check("语料目录存在", bool(documents), f"{len(documents)} 篇文档")
    chunks = load_chunks()
    check("切分产出片段", len(chunks) > 10, f"{len(chunks)} 个片段")
    check("每个片段都有来源标签", all(c.label for c in chunks))

    print("\n[2] 检索")
    runtime.set_retriever(None)
    settings = runtime.settings()
    retriever = runtime.get_retriever()
    result = retriever.search("退货运费谁承担")
    check("查得到结果", bool(result.hits),
          f"覆盖度={result.top_coverage:.3f} 余弦={result.top_dense:.3f} 模式={settings.retrieval_mode}")
    check("判据取值在合理区间", 0.0 <= result.refusal_score <= 1.0,
          f"{settings.refusal_signal}={result.refusal_score:.3f}")
    answerable = retriever.search("退款要等几天才能到账")
    check("知识库内的问题不触发拒答",
          not answerable.below(settings.refusal_threshold),
          f"{answerable.refusal_score:.3f} >= {settings.refusal_threshold}")
    off_topic = retriever.search("怎么申请提高信用卡额度")
    check("超纲问题识别为知识库外",
          off_topic.below(settings.refusal_threshold),
          f"{off_topic.refusal_score:.3f} < {settings.refusal_threshold}")

    print("\n[3] 订单数据")
    clock = policy.now_utc()
    connection = db.connect(":memory:")
    count = db.seed(connection, now=clock)
    runtime.set_connection(connection)
    runtime.set_clock(lambda: clock)
    runtime.set_retriever(retriever)
    check("播种订单", count == 11, f"{queries.count_orders(connection)} 条")

    print("\n[4] 工具")
    samples = {
        "query_order": {"order_id": "SO20261001001"},
        "query_logistics": {"order_id": "SO20261001001"},
        "check_return_eligibility": {"order_id": "SO20261001001", "reason": "no_reason"},
        "calc_refund": {"order_id": "SO20261001001", "reason": "no_reason"},
        "check_exchange": {"order_id": "SO20261001001"},
        "calc_late_ship_compensation": {"order_id": "SO20261010005"},
    }
    for name, args in samples.items():
        payload = json.loads(toolkit.TOOL_MAP[name].invoke(args))
        check(f"{name} 可用", bool(payload.get("ok")) and bool(payload.get("summary")),
              f"→ {payload.get('summary', '')[:28]}…")

    print("\n[5] 智能体")
    agent = Agent(StubChatModel(), retriever, toolkit.TOOLS)
    policy_answer = agent.run("7 天无理由退货的期限是几天？")
    check("政策问题给出引用", bool(policy_answer.citations), f"{len(policy_answer.citations)} 条引用")
    refund_answer = agent.run("SO20261001001 无理由退货能退多少钱？")
    check("订单问题触发工具", bool(refund_answer.tool_calls),
          f"→ {refund_answer.tool_calls[0]['name'] if refund_answer.tool_calls else '无'}")
    check("退款金额正确",
          bool(refund_answer.tool_calls)
          and refund_answer.tool_calls[0]["data"].get("refund_yuan") == "241.00",
          "241.00 元")
    refused = agent.run("怎么申请提高信用卡额度")
    check("超纲问题拒答", refused.refused)
    check("没有越界引用", all(1 <= c["index"] <= len(policy_answer.hits)
                              for c in policy_answer.citations))

    print("\n[6] HTTP 接口")
    runtime.set_agent(agent)
    with TestClient(create_app()) as client:
        health = client.get("/api/health")
        check("GET /api/health", health.status_code == 200 and health.json()["status"] == "ok")
        tools_response = client.get("/api/tools")
        check("GET /api/tools", tools_response.status_code == 200
              and len(tools_response.json()) == 6)
        orders_response = client.get("/api/demo/orders")
        check("GET /api/demo/orders", orders_response.status_code == 200
              and len(orders_response.json()) == 11)
        chat = client.post("/api/chat", json={"question": "7 天无理由退货是几天"})
        check("POST /api/chat", chat.status_code == 200 and bool(chat.json()["answer"]))
        check("空问题被拒", client.post("/api/chat", json={"question": "  "}).status_code == 422)

        frames = []
        with client.stream("POST", "/api/chat/stream",
                           json={"question": "SO20261001001 快递到哪了"}) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[6:]))
        types = [frame["type"] for frame in frames]
        check("POST /api/chat/stream 事件齐全",
              types[:1] == ["retrieval"] and types[-1:] == ["done"]
              and "tool_result" in types and "answer" in types,
              f"{len(frames)} 帧")

        index = client.get("/")
        if (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").is_file():
            check("GET / 返回页面", index.status_code == 200)
        else:
            print("  [skip] 前端还没生成，跳过 GET /")

    print()
    print("=" * 62)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for label in FAILED:
            print(f"  [FAIL] {label}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
