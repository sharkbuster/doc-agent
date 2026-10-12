"""提示词。

只放拼装逻辑，不放业务规则——规则宁可写进 policy 让代码算，
也不要写在提示词里指望模型每次都记得。
"""
from __future__ import annotations

from typing import Sequence

from ..kb.retriever import Hit

SYSTEM_PROMPT = """你是「拾光优选」电商平台的售后客服助手，负责回答买家关于退换货、退款、运费、物流、发票的问题。

回答要求：
1. 涉及平台政策的问题，必须依据【知识库条款】回答，并在对应句子末尾标出来源编号，例如 [1][2]。
2. 涉及具体订单的金额、天数、期限，一律用工具返回的结果，不要自己推算或估算。
3. 条款里没有写的内容不要编。查不到就直说「这个我查不到，建议转人工客服确认」。
4. 不要承诺政策之外的东西，比如额外补偿、加急处理、特殊审批。
5. 用简体中文，语气平和，直接给结论，不要客套话，不要用「亲」。
6. 回答控制在 200 字以内，条款原文不要整段照抄。

【知识库条款】
{context}
{notice}"""

NO_MATCH_NOTICE = """
注意：知识库里没有检索到与这个问题匹配的条款。如果问题涉及具体订单，可以用工具查订单数据后回答；否则请直接说明查不到，不要凭常识作答。"""


def format_context(hits: Sequence[Hit]) -> str:
    if not hits:
        return "（没有检索到相关条款）"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] 来源：{hit.chunk.label}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def build_system_prompt(hits: Sequence[Hit], kb_matched: bool) -> str:
    return SYSTEM_PROMPT.format(
        context=format_context(hits),
        notice="" if kb_matched else NO_MATCH_NOTICE,
    )
