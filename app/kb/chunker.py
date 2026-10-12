"""把知识库文档切成可检索的片段。

三条约束，都是实际调出来的：

1. **片段必须带标题路径**。回答时要说清依据是哪条政策的哪一小节
   （退换货政策 > 7 天无理由 > 不适用商品）。这个信息只有切分时顺手记下来，
   切完再回头从原文里找是找不回来的。

2. **中文词间没有空格**。按字符数硬切经常把一句话劈成两半，两边都只剩半个语义。
   所以顺序是：先按标题分节，再按段落和列表项分块，实在超长了才按句子拆，
   最后才允许硬切。

3. **相邻片段要留重叠**。跨段落的答案（前段结论 + 后段例外）不留重叠时，
   两个片段各自都不完整，谁都排不到前面。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_CHARS = 320
DEFAULT_OVERLAP = 60

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# 列表项也算块边界：政策文档里一条一款，合在一起检索会互相稀释
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.、)]|第[一二三四五六七八九十百]+[条款项])\s*")
# 句末标点。中文为主，英文顺带认一下
_SENT_END = "。！？；!?;"
# 标点后面可能还跟着引号或括号，切分点要往后挪，别让下一句以 」开头
_TRAILING = "”’」』）)】》”"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    heading: tuple[str, ...]
    text: str

    @property
    def label(self) -> str:
        """引用时给人看的来源，例如：退换货政策 > 7 天无理由退货 > 不适用商品。"""
        return " > ".join(p for p in (self.doc, *self.heading) if p)


def _first_h1(text: str) -> str:
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
        if line.strip() and not m:
            break  # 正文已经开始，H1 只认开头
    return ""


def _sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    """按标题把文档切成 (标题路径, 正文) 序列。"""
    out: list[tuple[tuple[str, ...], str]] = []
    stack: list[str] = []
    buf: list[str] = []
    heading: tuple[str, ...] = ()

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((heading, body))

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if not m:
            buf.append(line)
            continue
        flush()
        buf = []
        level = len(m.group(1))
        title = m.group(2).strip()
        stack = stack[: level - 1]
        # 标题跳级（## 直接跟在一级后面时缺了中间层）要补空位，否则层级会串
        while len(stack) < level - 1:
            stack.append("")
        stack.append(title)
        heading = tuple(t for t in stack if t)
    flush()
    return out


def _blocks(body: str) -> list[str]:
    """把一节正文拆成块：空行分段，列表项各成一块。"""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        lines = para.splitlines()
        if len(lines) == 1 or not any(_LIST_ITEM.match(ln) for ln in lines):
            blocks.append(para)
            continue
        # 混排：列表项单独成块，非列表行归到前一块里
        current: list[str] = []
        for ln in lines:
            if _LIST_ITEM.match(ln) and current:
                blocks.append("\n".join(current).strip())
                current = [ln]
            else:
                current.append(ln)
        if current:
            blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，标点连同后面的引号一起留在前一句。"""
    out: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in _SENT_END:
            j = i + 1
            while j < len(text) and text[j] in _TRAILING:
                j += 1
            piece = text[start:j].strip()
            if piece:
                out.append(piece)
            start = j
            i = j
        else:
            i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _tail(text: str, overlap: int) -> str:
    """取末尾一小段做重叠。

    优先整句取；一句话就比 overlap 还长时退回按字符截。截断的那一刀会切在句子中间，
    但重叠区本来就是给检索当冗余用的，不值得为此把重叠长度放大到不可控。
    """
    if overlap <= 0 or not text:
        return ""
    acc: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)):
        if total + len(sentence) > overlap:
            break
        acc.insert(0, sentence)
        total += len(sentence)
    return "".join(acc) or text[-overlap:]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """没有标点的超长文本（比如一整行表格数据）只能按长度硬切，这是最后手段。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_text(
    text: str,
    *,
    doc: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """切一份文档。doc 是文档名，用于引用展示和片段 id。"""
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正数")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap >= max_chars:
        # 重叠比块还长的话，打包逻辑会退化成无限循环式的重叠，必须拦住
        raise ValueError(f"overlap({overlap}) 必须小于 max_chars({max_chars})")

    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        units: list[str] = []
        for block in _blocks(body):
            if len(block) <= max_chars:
                units.append(block)
                continue
            # 超长块按句拆开，交给下面的打包逻辑去拼。
            #
            # 这里最初写的是「先按 max_chars 预分组」，结果是每组都贴着上限，
            # 「重叠尾巴 + 下一组」永远超出预算，重叠被静默丢弃——测试里
            # 表现为相邻块完全不重叠、检索跨段答案的能力直接没了。
            # 改成句级单元之后，打包器才有余量去留重叠。
            for sentence in split_sentences(block):
                if len(sentence) <= max_chars:
                    units.append(sentence)
                else:
                    units.extend(_hard_wrap(sentence, max_chars))

        pending = ""
        for unit in units:
            if not pending:
                pending = unit
                continue
            if len(pending) + 1 + len(unit) <= max_chars:
                pending = f"{pending}\n{unit}"
                continue
            chunks.append(_make(doc, heading, pending, len(chunks)))
            tail = _tail(pending, overlap)
            # 只有「重叠尾巴 + 新块」装得下才带重叠。装不下就宁可不要重叠——
            # 一个内容全是重叠的片段信息量极低，只会白占一个召回位。
            if tail and len(tail) + 1 + len(unit) <= max_chars:
                pending = f"{tail}\n{unit}"
            else:
                pending = unit
        if pending.strip():
            chunks.append(_make(doc, heading, pending, len(chunks)))
    return chunks


def _make(doc: str, heading: tuple[str, ...], text: str, index: int) -> Chunk:
    return Chunk(id=f"{doc}#{index:03d}", doc=doc, heading=heading, text=text.strip())


def split_corpus(
    directory: Path,
    *,
    pattern: str = "*.md",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """切一整个目录。文件按名字排序，保证每次构建的片段顺序一致。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"知识库目录不存在: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        title = _first_h1(text) or path.stem
        for chunk in split_text(text, doc=title, max_chars=max_chars, overlap=overlap):
            # H1 已经当作文档名了，别再在标题路径里重复一遍
            heading = chunk.heading[1:] if chunk.heading[:1] == (title,) else chunk.heading
            chunks.append(
                Chunk(id=chunk.id, doc=chunk.doc, heading=heading, text=chunk.text)
            )
    return chunks
