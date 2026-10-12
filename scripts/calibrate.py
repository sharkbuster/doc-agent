"""扫参数：RRF 里的稠密权重、以及拒答阈值。

    .venv\\Scripts\\python.exe -m scripts.calibrate

两个参数都不该拍脑袋定：

- **稠密权重**。RRF 默认等权，但这个项目里的稠密向量是自研的字符 n-gram 哈希，
  明显弱于 BM25。等权融合的结果是弱信号把强信号拽下来（实测 Recall@3 从 0.83 掉到 0.78）。
- **拒答阈值**。这是「该拒答的判成能答」和「能答的判成拒答」之间的取舍，
  只有把两条错误率都算出来才谈得上选。第一版凭感觉填了 0.35，
  结果 36 条政策问题全被判成拒答——因为那个信号的取值范围根本不在 0.35 附近。

扫完把推荐值写进 .env 或 app/config.py 的默认值。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from app.config import CORPUS_DIR, EVAL_DIR  # noqa: E402
from app.kb.chunker import split_corpus  # noqa: E402
from app.kb.embedder import HashingEmbedder  # noqa: E402
from app.kb.retriever import Retriever  # noqa: E402

DENSE_WEIGHTS = (0.0, 0.15, 0.3, 0.5, 0.7, 1.0)
THRESHOLDS = [round(step / 100, 2) for step in range(5, 101, 1)]


def load_cases() -> list[dict]:
    with (EVAL_DIR / "qa_eval.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("question")]


def is_answerable(row: dict) -> bool:
    """政策类问题必须能答；订单类由工具兜底，不参与拒答阈值的标定。"""
    return row.get("should_refuse") != "1" and row.get("type") in ("policy", "hybrid")


def is_refusable(row: dict) -> bool:
    return row.get("should_refuse") == "1"


def retrieval_score(retriever: Retriever, cases: list[dict], depth: int = 10) -> dict:
    relevant = [c for c in cases if c.get("expected_doc") and c.get("expected_keyword")]
    hit_at = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    for case in relevant:
        result = retriever.search(case["question"], k=depth)
        rank = None
        for position, hit in enumerate(result.hits, start=1):
            if hit.chunk.doc == case["expected_doc"] and case["expected_keyword"] in hit.chunk.text:
                rank = position
                break
        if rank is None:
            continue
        for k in hit_at:
            if rank <= k:
                hit_at[k] += 1
        reciprocal += 1.0 / rank
    total = len(relevant) or 1
    return {
        "recall@1": hit_at[1] / total,
        "recall@3": hit_at[3] / total,
        "recall@5": hit_at[5] / total,
        "mrr": reciprocal / total,
    }


def sweep_dense_weight(cases: list[dict]) -> dict:
    chunks = split_corpus(CORPUS_DIR)
    out = {}
    for weight in DENSE_WEIGHTS:
        retriever = Retriever(chunks, HashingEmbedder(dim=1024), dense_weight=weight, top_k=5)
        out[weight] = retrieval_score(retriever, cases)
    return out


def collect_scores(cases: list[dict], signal: str) -> tuple[list[float], list[float]]:
    chunks = split_corpus(CORPUS_DIR)
    retriever = Retriever(chunks, HashingEmbedder(dim=1024), top_k=5)
    answerable: list[float] = []
    refusable: list[float] = []
    for case in cases:
        result = retriever.search(case["question"])
        value = result.top_coverage if signal == "coverage" else result.top_dense
        if is_answerable(case):
            answerable.append(value)
        elif is_refusable(case):
            refusable.append(value)
    return answerable, refusable


def sweep_threshold(answerable: list[float], refusable: list[float], signal: str) -> dict:
    rows = []
    for threshold in THRESHOLDS:
        false_refusal = sum(1 for s in answerable if s < threshold) / len(answerable)
        detected = sum(1 for s in refusable if s < threshold) / len(refusable)
        rows.append({
            "threshold": threshold,
            "false_refusal": false_refusal,
            "detected": detected,
            # 约登指数：两个错误率之差，越大越好。只看单项一定会走极端
            # （阈值取 0 就永不误拒答，取 1 就永不漏拒答）。
            "youden": detected - false_refusal,
            "balanced": (detected + (1 - false_refusal)) / 2,
        })
    best = max(rows, key=lambda row: (row["youden"], -row["false_refusal"], row["threshold"]))
    return {"signal": signal, "rows": rows, "best": best}


# ----------------------------------------------------------------------
# 拒答判据的不同算法对比
#
# 这一段是**实验代码**，刻意没有去复用 Retriever.coverage：生产路径上只需要一种判据，
# 为了「可替换」把它做成带一堆开关的接口，等于让每个调用点都多背几个用不上的参数。
# 实验代码重复十来行、换生产接口干净，划算。
# ----------------------------------------------------------------------

SIGNAL_VARIANTS = [
    ("bigram", (2,), True, 3),
    ("bigram", (2,), True, 5),
    ("unigram", (1,), True, 3),
    ("unigram+bigram", (1, 2), True, 3),
    ("unigram+bigram", (1, 2), True, 5),
    ("bigram 不加权", (2,), False, 3),
    ("unigram+bigram 不加权", (1, 2), False, 5),
]


def _variant_scores(cases: list[dict], sizes: tuple[int, ...], weighted: bool, top_n: int):
    from app.kb.text import tokenize

    chunks = split_corpus(CORPUS_DIR)
    retriever = Retriever(chunks, None, mode="lexical", top_k=5)
    idf = retriever.lexical.idf
    fallback = max(idf.values()) if idf else 1.0
    pool_by_chunk = [set(tokenize(c.text, sizes)) for c in chunks]
    index_of = {id(c): i for i, c in enumerate(chunks)}

    def score(question: str) -> float:
        hits = retriever.search(question).hits[:top_n]
        if not hits:
            return 0.0
        wanted = set(tokenize(question, sizes))
        if not wanted:
            return 0.0
        available: set[str] = set()
        for hit in hits:
            available |= pool_by_chunk[index_of[id(hit.chunk)]]
        if not weighted:
            return len(wanted & available) / len(wanted)
        total = sum(idf.get(t, fallback) for t in wanted)
        if total <= 0:
            return 0.0
        return sum(idf.get(t, fallback) for t in wanted if t in available) / total

    answerable, refusable = [], []
    for case in cases:
        value = score(case["question"])
        if is_refusable(case):
            refusable.append(value)
        elif is_answerable(case):
            answerable.append(value)
    return answerable, refusable


def sweep_signals(cases: list[dict]) -> list[dict]:
    out = []
    for label, sizes, weighted, top_n in SIGNAL_VARIANTS:
        answerable, refusable = _variant_scores(cases, sizes, weighted, top_n)
        report = sweep_threshold(answerable, refusable, label)
        report["label"] = f"{label} · top{top_n}"
        report["sizes"] = sizes
        report["weighted"] = weighted
        report["top_n"] = top_n
        out.append(report)
    return out


def print_signal_table(reports: list[dict], n_answerable: int) -> None:
    print()
    print("=" * 68)
    print("拒答判据的不同算法对比")
    print("=" * 68)
    print(f"  {'判据':<26}{'阈值':>7}{'误拒答':>9}{'拒答识别':>10}{'约登':>9}")
    for report in reports:
        best = report["best"]
        print(f"  {report['label']:<26}{best['threshold']:>7.3f}"
              f"{best['false_refusal']:>9.1%}{best['detected']:>10.1%}{best['youden']:>9.3f}")
    current = reports[0]
    winner = max(reports, key=lambda r: r["best"]["youden"])
    print()
    print(f"  当前生产用的是「{current['label']}」，约登 {current['best']['youden']:.3f}")
    if winner is current:
        print("  → 它同时就是最好的一档，不用动。")
        return
    gain = winner["best"]["youden"] - current["best"]["youden"]
    fewer = (current["best"]["false_refusal"] - winner["best"]["false_refusal"]) * n_answerable
    print(f"  最好的一档是「{winner['label']}」，约登 {winner['best']['youden']:.3f}"
          f"（高 {gain:.3f}），相当于少错 {fewer:.0f} 条")
    if fewer < 2:
        print("  → 这个样本量下少错 1 条说明不了什么，而且换过去要放弃 IDF 加权——")
        print("    那是用来兜住「评测集里没出现过的离题问法」的。维持现状。")
    else:
        print("  → 差距够大，值得换。换完记得同步改 app/config.py 的默认值和 REFUSAL_THRESHOLD。")


def print_weight_table(table: dict) -> None:
    print("=" * 68)
    print("稠密权重扫描（RRF 融合里稠密分量占的权重，0 = 纯词法）")
    print("=" * 68)
    print(f"{'权重':>6}{'Recall@1':>12}{'Recall@3':>12}{'Recall@5':>12}{'MRR':>10}")
    best = max(table, key=lambda w: (table[w]["recall@3"], table[w]["mrr"]))
    for weight, metrics in table.items():
        mark = "  ← 推荐" if weight == best else ""
        print(f"{weight:>6}{metrics['recall@1']:>12.3f}{metrics['recall@3']:>12.3f}"
              f"{metrics['recall@5']:>12.3f}{metrics['mrr']:>10.3f}{mark}")


def print_threshold_table(report: dict) -> None:
    signal = report["signal"]
    print()
    print("=" * 68)
    print(f"拒答阈值扫描（判据：{signal}）")
    print("=" * 68)
    print(f"  {'阈值':>6}{'能答却被拒':>14}{'该拒被识别':>14}{'约登指数':>12}")
    interesting = [row for row in report["rows"]
                   if row["threshold"] * 100 % 5 == 0 or row is report["best"]]
    for row in interesting:
        mark = "  ← 推荐" if row is report["best"] else ""
        print(f"  {row['threshold']:>6.2f}{row['false_refusal']:>14.3f}"
              f"{row['detected']:>14.3f}{row['youden']:>12.3f}{mark}")


def main() -> int:
    cases = load_cases()
    print(f"评测集：{len(cases)} 条")
    print()

    weights = sweep_dense_weight(cases)
    print_weight_table(weights)
    best_weight = max(weights, key=lambda w: (weights[w]["recall@3"], weights[w]["mrr"]))

    reports = {}
    for signal in ("coverage", "dense"):
        answerable, refusable = collect_scores(cases, signal)
        reports[signal] = sweep_threshold(answerable, refusable, signal)
        print_threshold_table(reports[signal])

    print_signal_table(sweep_signals(cases), sum(1 for c in cases if is_answerable(c)))

    print()
    print("=" * 68)
    print("结论")
    print("=" * 68)
    print(f"  DENSE_WEIGHT 建议取   : {best_weight}")
    for signal, report in reports.items():
        best = report["best"]
        print(f"  REFUSAL_SIGNAL={signal:<9}: 阈值 {best['threshold']:.2f}"
              f"  误拒答 {best['false_refusal']:.1%}  拒答识别 {best['detected']:.1%}"
              f"  约登 {best['youden']:.3f}")
    winner = max(reports.values(), key=lambda r: r["best"]["youden"])
    print()
    print(f"  → 判据用 {winner['signal']}，阈值取 {winner['best']['threshold']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
