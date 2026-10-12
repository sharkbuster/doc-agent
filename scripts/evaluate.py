"""跑评测集，输出检索、工具选择、拒答三组指标。

    .venv\\Scripts\\python.exe -m scripts.evaluate
    .venv\\Scripts\\python.exe -m scripts.evaluate --gate    # CI 用：跌破阈值就退出码非 0

三类指标分开看，因为它们坏起来的原因完全不同：
检索差是把不相干的条款塞给了模型；工具选错是订单数据根本没查到；
拒答失灵是对着知识库外面的问题硬编一个答案出来。

评测全程用内存数据库 + 固定的「现在」，所以同一份代码任何时候跑出来的
数字都一样，不受运行日期影响。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 中文 Windows 的控制台是 GBK，输出里出现它编码不了的字符就会整个脚本崩掉
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from app import policy, runtime  # noqa: E402
from app.agent import tools as toolkit  # noqa: E402
from app.agent.llm import StubChatModel, build_model  # noqa: E402
from app.agent.loop import Agent  # noqa: E402
from app.config import CORPUS_DIR, DATA_DIR, EVAL_DIR  # noqa: E402
from app.kb.chunker import split_corpus  # noqa: E402
from app.kb.embedder import HashingEmbedder  # noqa: E402
from app.kb.retriever import Retriever  # noqa: E402
from app.store import db  # noqa: E402

EVAL_CSV = EVAL_DIR / "qa_eval.csv"
RESULT_JSON = EVAL_DIR / "eval_result.json"

TOP_K = 5
MRR_DEPTH = 10
CHUNK_SIZES = (200, 320, 480)

# CI 门禁。定在实测值下方留一点余量：门禁的意义是发现「改坏了」，
# 不是要求每次提交都刷出新高分。citation_leak 必须是 0，这个没有余量可留。
GATES = {
    "hybrid_recall@3": 0.90,
    "hybrid_mrr": 0.85,
    "tool_accuracy": 0.90,
    "refusal_accuracy": 0.85,
    "citation_leak": 0,
}


def load_cases(path: Path = EVAL_CSV) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"找不到评测集：{path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("question") or "").strip()]
    for row in rows:
        row["expected_tool"] = (row.get("expected_tool") or "").strip()
        row["expected_doc"] = (row.get("expected_doc") or "").strip()
        row["expected_keyword"] = (row.get("expected_keyword") or "").strip()
        row["should_refuse"] = (row.get("should_refuse") or "0").strip() == "1"
        row["type"] = (row.get("type") or "").strip()
    return rows


def matches(hit, case: dict) -> bool:
    """命中判定：文档对得上，且关键词真的出现在这个片段里。

    用「文档 + 关键词」而不是片段 id：片段 id 会随切分参数变化，
    一旦改了 max_chars，所有标注就全废了。
    """
    return (
        hit.chunk.doc == case["expected_doc"]
        and case["expected_keyword"] in hit.chunk.text
    )


def retrieval_metrics(retriever, cases: list[dict], depth: int = MRR_DEPTH) -> dict:
    relevant = [c for c in cases if c["expected_doc"] and c["expected_keyword"]]
    if not relevant:
        return {"total": 0}
    recall = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    misses: list[str] = []
    for case in relevant:
        result = retriever.search(case["question"], k=depth)
        rank = None
        for position, hit in enumerate(result.hits, start=1):
            if matches(hit, case):
                rank = position
                break
        if rank is None:
            misses.append(case["id"])
            continue
        for k in recall:
            if rank <= k:
                recall[k] += 1
        reciprocal += 1.0 / rank
    total = len(relevant)
    return {
        "total": total,
        "recall@1": round(recall[1] / total, 4),
        "recall@3": round(recall[3] / total, 4),
        "recall@5": round(recall[5] / total, 4),
        "mrr": round(reciprocal / total, 4),
        "misses": misses,
    }


def compare_modes(cases: list[dict]) -> dict:
    """三种检索模式在同一份语料和同一批问题上对比。"""
    chunks = split_corpus(CORPUS_DIR)
    out = {}
    for mode in ("hybrid", "dense", "lexical"):
        retriever = Retriever(chunks, HashingEmbedder(dim=1024), mode=mode, top_k=TOP_K)
        out[mode] = retrieval_metrics(retriever, cases)
        out[mode]["backend"] = retriever.backend
    return out


def sweep_chunk_size(cases: list[dict]) -> dict:
    """切分粒度对召回的影响。

    不扫一遍就是拍脑袋定 max_chars；扫完才知道 320 是不是真的比 200 好。
    """
    out = {}
    for size in CHUNK_SIZES:
        chunks = split_corpus(CORPUS_DIR, max_chars=size, overlap=size // 5)
        retriever = Retriever(chunks, HashingEmbedder(dim=1024), top_k=TOP_K)
        metrics = retrieval_metrics(retriever, cases)
        out[str(size)] = {
            "chunks": len(chunks),
            "recall@1": metrics["recall@1"],
            "recall@3": metrics["recall@3"],
            "mrr": metrics["mrr"],
        }
    return out


def build_agent() -> tuple[Agent, str]:
    settings = runtime.settings()
    model = build_model(settings) if settings.llm_provider != "stub" else StubChatModel()
    agent = Agent(
        model,
        runtime.get_retriever(),
        toolkit.TOOLS,
        refusal_threshold=settings.refusal_threshold,
    )
    return agent, settings.llm_provider


def agent_metrics(agent: Agent, cases: list[dict]) -> dict:
    tool_cases = [c for c in cases if c["expected_tool"]]
    correct = 0
    confusion: Counter[str] = Counter()
    mismatches: list[dict] = []
    answers: dict[str, object] = {}
    for case in tool_cases:
        answer = agent.run(case["question"])
        answers[case["id"]] = answer
        picked = answer.tool_calls[0]["name"] if answer.tool_calls else ""
        if picked == case["expected_tool"]:
            correct += 1
        else:
            confusion[f"{case['expected_tool']} -> {picked or '（没调工具）'}"] += 1
            mismatches.append({
                "id": case["id"], "question": case["question"],
                "expected": case["expected_tool"], "picked": picked,
            })

    refuse_cases = [c for c in cases if c["should_refuse"]]
    refused_right = 0
    for case in refuse_cases:
        answer = agent.run(case["question"])
        answers[case["id"]] = answer
        if answer.refused:
            refused_right += 1

    answerable = [c for c in cases if not c["should_refuse"]]
    false_refusals = []
    citation_covered = 0
    citation_expected = 0
    leaks = 0
    for case in answerable:
        answer = answers.get(case["id"])
        if answer is None:
            answer = agent.run(case["question"])
            answers[case["id"]] = answer
        if answer.refused:
            false_refusals.append(case["id"])
        if case["type"] in ("policy", "hybrid"):
            citation_expected += 1
            if answer.citations:
                citation_covered += 1
        # 越界引用在循环里已经被丢掉，这里再确认一次没有漏网的
        leaks += sum(1 for c in answer.citations if c["index"] < 1 or c["index"] > len(answer.hits))

    return {
        "tool_total": len(tool_cases),
        "tool_accuracy": round(correct / len(tool_cases), 4) if tool_cases else 0.0,
        "tool_confusion": dict(confusion),
        "tool_mismatches": mismatches[:15],
        "refuse_total": len(refuse_cases),
        "refusal_accuracy": round(refused_right / len(refuse_cases), 4) if refuse_cases else 0.0,
        "false_refusals": false_refusals,
        "citation_expected": citation_expected,
        "citation_coverage": round(citation_covered / citation_expected, 4) if citation_expected else 0.0,
        "citation_leak": leaks,
    }


def print_report(report: dict) -> None:
    modes = report["retrieval_by_mode"]
    print("=" * 66)
    print("检索指标（同一批问题，三种检索模式对比）")
    print("=" * 66)
    print(f"{'模式':<10}{'后端':<8}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>8}")
    for mode in ("hybrid", "dense", "lexical"):
        m = modes[mode]
        print(f"{mode:<10}{m['backend']:<8}{m['recall@1']:>10.3f}{m['recall@3']:>10.3f}"
              f"{m['recall@5']:>10.3f}{m['mrr']:>8.3f}")
    print(f"（共 {modes['hybrid']['total']} 条有标准答案的问题）")

    print()
    print("=" * 66)
    print("切分粒度对比")
    print("=" * 66)
    print(f"{'max_chars':<12}{'片段数':>8}{'Recall@1':>12}{'Recall@3':>12}{'MRR':>10}")
    for size, m in report["chunk_sweep"].items():
        print(f"{size:<12}{m['chunks']:>8}{m['recall@1']:>12.3f}{m['recall@3']:>12.3f}{m['mrr']:>10.3f}")

    agent_part = report["agent"]
    print()
    print("=" * 66)
    print(f"工具选择 / 拒答 / 引用（模型：{report['llm_provider']}）")
    print("=" * 66)
    print(f"  工具选择准确率 : {agent_part['tool_accuracy']:.3f}  "
          f"({agent_part['tool_total']} 条)")
    print(f"  拒答准确率     : {agent_part['refusal_accuracy']:.3f}  "
          f"({agent_part['refuse_total']} 条)")
    print(f"  引用覆盖率     : {agent_part['citation_coverage']:.3f}  "
          f"({agent_part['citation_expected']} 条政策类问题)")
    print(f"  越界引用泄漏   : {agent_part['citation_leak']}  （必须为 0）")
    if agent_part["false_refusals"]:
        print(f"  误拒答         : {', '.join(agent_part['false_refusals'])}")
    if agent_part["tool_confusion"]:
        print("  工具选错明细   :")
        for pair, count in sorted(agent_part["tool_confusion"].items()):
            print(f"      {pair} × {count}")
    if report["llm_provider"] == "stub":
        print()
        print("  注意：当前是离线桩模型，工具选择这一项测的是桩的关键词规则，")
        print("        不代表真实模型的能力。要看真实数字请把 LLM_PROVIDER 切到 deepseek。")


def check_gates(report: dict) -> list[str]:
    failures = []
    hybrid = report["retrieval_by_mode"]["hybrid"]
    actual = {
        "hybrid_recall@3": hybrid["recall@3"],
        "hybrid_mrr": hybrid["mrr"],
        "tool_accuracy": report["agent"]["tool_accuracy"],
        "refusal_accuracy": report["agent"]["refusal_accuracy"],
        "citation_leak": report["agent"]["citation_leak"],
    }
    for name, floor in GATES.items():
        value = actual[name]
        if name == "citation_leak":
            if value > floor:
                failures.append(f"{name}: {value} > {floor}")
            continue
        if value < floor:
            failures.append(f"{name}: {value:.4f} < {floor}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评测集")
    parser.add_argument("--gate", action="store_true", help="指标跌破阈值时返回非 0")
    parser.add_argument("--out", default=str(RESULT_JSON), help="结果 JSON 的写到哪里")
    args = parser.parse_args()

    cases = load_cases()
    by_type = Counter(case["type"] for case in cases)

    # 固定「现在」并重新播种，否则「签收后第 N 天」会随运行日期漂移，
    # 订单类问题的标准答案就不成立了
    clock = policy.now_utc()
    connection = db.connect(":memory:")
    db.seed(connection, now=clock)
    runtime.set_connection(connection)
    runtime.set_clock(lambda: clock)
    runtime.set_settings(None)
    runtime.set_retriever(None)

    agent, provider = build_agent()
    report = {
        "cases": len(cases),
        "cases_by_type": dict(by_type),
        "llm_provider": provider,
        "config": {
            "top_k": TOP_K,
            "embedding": runtime.settings().public_view()["embedding"],
            "chunks": len(runtime.get_retriever().chunks),
            "documents": len(list(CORPUS_DIR.glob("*.md"))),
            "refusal_threshold": runtime.settings().refusal_threshold,
        },
        "retrieval_by_mode": compare_modes(cases),
        "chunk_sweep": sweep_chunk_size(cases),
        "agent": agent_metrics(agent, cases),
        "gates": GATES,
    }

    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"结果已写入 {out}")

    failures = check_gates(report)
    if args.gate:
        print()
        if failures:
            print("指标门禁未通过：")
            for item in failures:
                print(f"  [FAIL] {item}")
            return 1
        print("指标门禁全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
