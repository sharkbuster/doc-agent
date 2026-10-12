"""分词。

只在这里实现一次，索引和查询都调它。两边用不同的切法会出现
「文档里明明有这句话却检索不到」这种最难查的问题。
"""
from __future__ import annotations

import re

# 全角字符（ＡＢ，３）转半角。用户从微信/Word 里粘过来的问题经常是全角的，
# 不统一的话 "３天" 和 "3天" 是两个完全无关的词。
_FULLWIDTH = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
_FULLWIDTH[0x3000] = 0x20

_ASCII_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPACE = re.compile(r"\s+")


def to_halfwidth(text: str) -> str:
    return text.translate(_FULLWIDTH)


def normalize(text: str) -> str:
    return _SPACE.sub(" ", to_halfwidth(text).lower()).strip()


def base_tokens(text: str) -> list[str]:
    """切成最小单元：汉字一字一个，英文数字连成一个词。"""
    text = normalize(text)
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _CJK.match(ch):
            tokens.append(ch)
            i += 1
            continue
        m = _ASCII_WORD.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        i += 1  # 标点和空白不进词表：它们对区分文档没帮助，只会灌水
    return tokens


def tokenize(text: str, sizes: tuple[int, ...] = (1, 2)) -> list[str]:
    """单元 + n-gram。中文靠 bigram 才能表达「退货」和「退款」的区别。"""
    base = base_tokens(text)
    if not base:
        return []
    out = list(base) if 1 in sizes else []
    for n in sizes:
        if n <= 1:
            continue
        for i in range(len(base) - n + 1):
            out.append("".join(base[i : i + n]))
    return out
