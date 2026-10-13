"""模型层。

默认 `stub`：一个离线的确定性假模型。它不聪明，但会把「检索 → 工具调用 →
读工具结果 → 生成回答」这条链路完整走一遍，于是测试和 CI 不需要任何密钥。
一个开源仓库如果得先申请 API key 才能跑 pytest，基本不会有人跑。

`deepseek` / `openai` 走 langchain_openai 的 ChatOpenAI。DeepSeek 是 OpenAI 兼容接口，
换个 base_url 就行，不需要额外的适配层。
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

_ORDER_ID = re.compile(r"\bSO\d{6,}\b", re.I)

# 顺序有讲究：越具体的意图越靠前。
# 「说好 48 小时发货怎么还没发」同时命中「发货」和「超时」，必须先判赔付，
# 否则会被物流工具抢走。
_INTENTS: list[tuple[re.Pattern[str], str, dict]] = [
    (re.compile(r"赔付|赔偿|超时.{0,6}发货|一直没发货|还不发货|说好.{0,4}小时"), "calc_late_ship_compensation", {}),
    (re.compile(r"换货|换一个|换同款|换个|换新"), "check_exchange", {}),
    (re.compile(r"退款|退多少|退多少钱|退钱|能退多少|到账"), "calc_refund", {}),
    (re.compile(r"能不能退|可以退|能退吗|无理由|退货|退款政策|超期|过期|期限"), "check_return_eligibility", {}),
    (re.compile(r"物流|快递|到哪|运单|签收没|收到"), "query_logistics", {}),
]

_REASON_IN_TEXT = [
    (re.compile(r"质量|坏了|破损|有问题|故障|不能用|发错"), "quality"),
    (re.compile(r"不喜欢|不想要|无理由|七天|7天|退货"), "no_reason"),
]


class StubChatModel(BaseChatModel):
    """离线假模型：按关键词挑工具，回答就直接复述工具或检索结果的要点。

    它存在的意义是让整条链路可以在没有网络、没有密钥的情况下被测试，
    而不是冒充一个能用的客服。真实回答质量请把 LLM_PROVIDER 切到 deepseek。
    """

    tools: list[dict] = []
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "stub"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "StubChatModel":
        formatted = [
            tool if isinstance(tool, dict) else convert_to_openai_tool(tool)
            for tool in tools
        ]
        return self.model_copy(update={"tools": formatted})

    # ---- 决策 ----

    @staticmethod
    def _text_of(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))

    def _last_human(self, messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if message.type == "human":
                return self._text_of(message)
        return ""

    def _tool_names(self) -> set[str]:
        names = set()
        for spec in self.tools:
            name = spec.get("function", {}).get("name") or spec.get("name")
            if name:
                names.add(name)
        return names

    def _pick(self, question: str) -> tuple[str, dict] | None:
        order = _ORDER_ID.search(question)
        if not order:
            return None
        order_id = order.group(0).upper()
        available = self._tool_names()
        for pattern, name, extra in _INTENTS:
            if name in available and pattern.search(question):
                args = {"order_id": order_id, **extra}
                if name in ("check_return_eligibility", "calc_refund"):
                    for reason_pattern, code in _REASON_IN_TEXT:
                        if reason_pattern.search(question):
                            args["reason"] = code
                            break
                    else:
                        args["reason"] = "no_reason"
                return name, args
        if "query_order" in available:
            return "query_order", {"order_id": order_id}
        return None

    # ---- 生成 ----

    def _from_context(self, messages: Sequence[BaseMessage]) -> str:
        """没有工具可用时的兜底：把系统提示里最相关的那段原文摘出来。

        真正的总结是模型干的活。这里只是让链路在没有密钥时也能跑通，
        所以宁可摘抄也不要假装自己在推理。

        匹配时必须带上「来源：」这个标记。只认 [数字] 的话，
        会先命中提示词里「例如 [1][2]」那句示例，然后摘出一段指令当答案。
        """
        system = ""
        for message in messages:
            if message.type == "system":
                system = self._text_of(message)
                break
        blocks = re.findall(r"\[(\d+)\] 来源：(.+?)(?=\n\[\d+\] 来源：|\Z)", system, re.S)
        if not blocks:
            return "知识库里没有找到相关条款，建议转人工客服确认。"
        index, body = blocks[0]
        lines = [line for line in body.strip().splitlines() if line.strip()]
        text = "\n".join(lines[1:])[:400] or lines[0]
        return f"依据 [{index}]：{text}"

    def _compose(self, tool_message: ToolMessage) -> str:
        raw = self._text_of(tool_message)
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        summary = data.get("summary", "")
        if not data.get("ok", True):
            return summary or "这个请求没能查到数据。"
        return summary

    def _next(self, messages: Sequence[BaseMessage]) -> AIMessage:
        last_tool = None
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                last_tool = message
                break
        if last_tool is not None:
            return AIMessage(content=self._compose(last_tool))

        picked = self._pick(self._last_human(messages))
        if picked is None:
            return AIMessage(content=self._from_context(messages))
        name, args = picked
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}",
                         "type": "tool_call"}],
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        if message.tool_calls:
            # 工具调用轮没有正文可流，一次性把调用信息吐出去
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=message.tool_calls))
            return
        text = message.content if isinstance(message.content, str) else ""
        step = 24
        for start in range(0, len(text), step):
            yield ChatGenerationChunk(message=AIMessageChunk(content=text[start : start + step]))


def build_model(settings) -> BaseChatModel:
    provider = (settings.llm_provider or "stub").lower()
    if provider == "stub":
        return StubChatModel()
    api_key, base_url, model = settings.llm_target()
    if not api_key:
        raise RuntimeError(
            f"LLM_PROVIDER={provider} 但没有读到密钥，检查 .env 里的 "
            f"{provider.upper()}_API_KEY"
        )
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0.2, "timeout": 60}
    if base_url:
        kwargs["base_url"] = base_url
    # DeepSeek 的思考模式开关走 extra_body（OpenAI 兼容格式）。
    # 注意：思考模式下 temperature 不生效，官方文档明确写了「不报错但也没用」。
    if provider == "deepseek" and settings.deepseek_thinking in ("enabled", "disabled"):
        kwargs["extra_body"] = {"thinking": {"type": settings.deepseek_thinking}}
    # 失败重试交给上层决定要不要重来，模型层重试只会让超时叠加
    kwargs["max_retries"] = 1
    return ChatOpenAI(**kwargs)
