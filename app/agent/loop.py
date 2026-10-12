"""智能体循环。

一轮问答的顺序：先检索知识库，把命中的条款塞进系统提示；再把问题交给模型；
模型要调工具就执行、把结果回灌，直到它给出最终回答。

三个必须有的护栏，都不是为了好看：

- **迭代上限**。模型可能反复调同一个工具，没有上限就是死循环加账单。
- **工具异常回灌**。工具炸了要变成一条 ToolMessage 交给模型，而不是中断整个请求——
  模型还能告诉用户「订单查不到」，直接 500 就什么都说不出来了。
- **引用校验**。回答里的 [7] 如果没有第 7 条对应，就是编的。
  编造引用比编造内容更隐蔽：用户看到有出处就信了。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .prompts import build_system_prompt

MAX_STEPS = 4
_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Event:
    type: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.data}


@dataclass
class Answer:
    text: str
    citations: list[dict]
    hits: list[dict]
    tool_calls: list[dict]
    refused: bool
    top_dense: float
    steps: int

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "citations": self.citations,
            "hits": self.hits,
            "tool_calls": self.tool_calls,
            "refused": self.refused,
            "top_dense": round(self.top_dense, 4),
            "steps": self.steps,
        }


def citation_indices(text: str) -> list[int]:
    return sorted({int(m.group(1)) for m in _CITATION.finditer(text)})


def clean_citations(text: str, total: int) -> tuple[str, list[int]]:
    """把越界的引用编号从回答里抹掉。

    宁可让回答少一个出处，也不能让它挂着一个不存在的来源——
    用户没有办法分辨那是不是编的。
    """
    bogus = [index for index in citation_indices(text) if index < 1 or index > total]
    for index in bogus:
        text = text.replace(f"[{index}]", "")
    # 抹掉编号后可能留下多余空格
    return re.sub(r"[ \t]{2,}", " ", text).strip(), bogus


def hit_dicts(hits) -> list[dict]:
    return [
        {
            "index": index,
            "label": hit.chunk.label,
            "doc": hit.chunk.doc,
            "heading": list(hit.chunk.heading),
            "score": round(hit.fused, 5),
            "dense": round(hit.dense, 4),
            "lexical": round(hit.lexical, 3),
            "text": hit.chunk.text,
        }
        for index, hit in enumerate(hits, start=1)
    ]


class Agent:
    def __init__(
        self,
        model,
        retriever,
        tools: Sequence,
        *,
        max_steps: int = MAX_STEPS,
        refusal_threshold: float = 0.35,
    ) -> None:
        if not tools:
            raise ValueError("至少要有一个工具")
        self.tool_map = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(list(tools))
        self.retriever = retriever
        self.max_steps = max(1, max_steps)
        self.refusal_threshold = refusal_threshold

    def _run_tool(self, call: dict) -> tuple[str, dict]:
        name = call.get("name") or ""
        args = call.get("args") or {}
        tool = self.tool_map.get(name)
        if tool is None:
            summary = f"没有名为 {name} 的工具"
            return json.dumps({"ok": False, "summary": summary}, ensure_ascii=False), {
                "ok": False, "summary": summary
            }
        try:
            # 参数校验失败、工具内部异常都走这条路：变成一条工具消息回灌给模型，
            # 让它自己决定是换参数重试还是如实告诉用户
            raw = tool.invoke(args)
        except Exception as exc:  # noqa: BLE001 —— 工具是可插拔的，什么异常都可能有
            summary = f"工具 {name} 执行失败：{type(exc).__name__}: {exc}"
            return json.dumps({"ok": False, "summary": summary}, ensure_ascii=False), {
                "ok": False, "summary": summary, "error": type(exc).__name__
            }
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = {"ok": True, "summary": text}
        return text, parsed

    def stream(self, question: str, history: Sequence[dict] | None = None) -> Iterator[Event]:
        result = self.retriever.search(question)
        low_confidence = result.below(self.refusal_threshold)
        kb_matched = bool(result.hits) and not low_confidence
        hits = hit_dicts(result.hits)
        yield Event("retrieval", {
            "query": question,
            "top_dense": round(result.top_dense, 4),
            "top_coverage": round(result.top_coverage, 4),
            "refusal_score": round(result.refusal_score, 4),
            "kb_matched": kb_matched,
            "hits": hits,
        })

        messages: list = [SystemMessage(content=build_system_prompt(result.hits, kb_matched))]
        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content") or ""
            if not content:
                continue
            messages.append(HumanMessage(content=content) if role == "user"
                            else AIMessage(content=content))
        messages.append(HumanMessage(content=question))

        collected_calls: list[dict] = []
        text = ""
        steps = 0
        for step in range(self.max_steps):
            steps = step + 1
            pieces: list[str] = []
            tool_calls: list[dict] = []
            for chunk in self.model.stream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    pieces.append(content)
                    yield Event("answer_delta", {"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            pieces.append(part["text"])
                            yield Event("answer_delta", {"text": part["text"]})
                for call in getattr(chunk, "tool_calls", None) or []:
                    tool_calls.append(dict(call))

            text = "".join(pieces)
            if not tool_calls:
                break

            messages.append(AIMessage(content=text, tool_calls=tool_calls))
            for call in tool_calls:
                name = call.get("name") or ""
                args = call.get("args") or {}
                call_id = call.get("id") or f"call_{len(collected_calls)}"
                yield Event("tool_call", {"id": call_id, "name": name, "args": args})
                raw, parsed = self._run_tool(call)
                collected_calls.append({
                    "id": call_id, "name": name, "args": args,
                    "ok": bool(parsed.get("ok", True)),
                    "summary": parsed.get("summary", ""),
                    "data": parsed,
                })
                yield Event("tool_result", collected_calls[-1])
                # 少了这条消息，下一轮模型会以为自己没调用过工具，然后重复调用
                messages.append(ToolMessage(content=raw, tool_call_id=call_id))
        else:
            # 步数用尽：工具还在被反复调用，如实说明，不要编一个答案
            note = "连续调用了多次工具仍然没有得到确定结果，建议转人工客服核实。"
            yield Event("answer_delta", {"text": note})
            text = note

        text, bogus = clean_citations(text, len(hits))
        if bogus:
            yield Event("citation_warning", {"dropped": bogus, "available": len(hits)})

        total = len(hits)
        citations = [
            {"index": i, "label": hits[i - 1]["label"], "text": hits[i - 1]["text"]}
            for i in citation_indices(text)
            if 1 <= i <= total
        ]
        # 拒答判据是结构性的：知识库没匹配上、又没调用任何工具，这一轮就没有依据。
        # 不去猜回答里的措辞——「查不到」这三个字可能出现在任何一句解释里，
        # 提示词自己就写着这句话，按关键词判会把正常回答也判成拒答。
        refused = (not text.strip()) or (low_confidence and not collected_calls)

        yield Event("answer", Answer(
            text=text,
            citations=citations,
            hits=hits,
            tool_calls=collected_calls,
            refused=refused,
            top_dense=result.top_dense,
            steps=steps,
        ).to_dict())
        yield Event("done", {})

    def run(self, question: str, history: Sequence[dict] | None = None) -> Answer:
        """把事件流收成一个结果对象。评测和测试用这个，不必自己拼事件。"""
        final: dict = {}
        for event in self.stream(question, history):
            if event.type == "answer":
                final = event.data
        return Answer(
            text=final.get("answer", ""),
            citations=final.get("citations", []),
            hits=final.get("hits", []),
            tool_calls=final.get("tool_calls", []),
            refused=bool(final.get("refused")),
            top_dense=float(final.get("top_dense", 0.0)),
            steps=int(final.get("steps", 0)),
        )
