"""切分器的边界测试。

重点不在「正常文档能切开」，而在几个真会出问题的地方：
无标点长文本、跳级标题、只有标题没有正文、重叠预算装不下。
"""
from __future__ import annotations

import pytest

from app.kb.chunker import (
    DEFAULT_MAX_CHARS,
    Chunk,
    split_corpus,
    split_sentences,
    split_text,
)

DOC = "退换货政策"


def texts(chunks: list[Chunk]) -> str:
    return "".join(c.text for c in chunks)


# ---------------------------------------------------------------- 基本切分


def test_empty_document_yields_nothing():
    assert split_text("", doc=DOC) == []


def test_whitespace_only_document_yields_nothing():
    assert split_text("   \n\n \t \n", doc=DOC) == []


def test_heading_without_body_yields_nothing():
    # 只有标题没有正文时不产生片段，否则会召回一堆标题党空块
    assert split_text("# 退换货政策\n\n## 7 天无理由\n\n## 运费\n", doc=DOC) == []


def test_short_document_stays_in_one_chunk():
    text = "## 7 天无理由\n\n签收之日起 7 天内可无理由退货。\n"
    chunks = split_text(text, doc=DOC)
    assert len(chunks) == 1
    assert chunks[0].heading == ("7 天无理由",)
    assert "7 天内可无理由退货" in chunks[0].text


def test_paragraph_not_split_when_it_fits():
    body = "签收之日起 7 天内可无理由退货。" * 3
    chunks = split_text(f"## 规则\n\n{body}\n", doc=DOC, max_chars=200)
    assert len(chunks) == 1
    assert body in chunks[0].text


# ---------------------------------------------------------------- 标题路径


def test_heading_path_nests():
    text = "# 退换货政策\n\n## 7 天无理由\n\n### 不适用商品\n\n定制类商品不支持无理由退货。\n"
    chunks = split_text(text, doc=DOC)
    assert chunks[0].heading == ("退换货政策", "7 天无理由", "不适用商品")


def test_skipped_heading_level_does_not_crash():
    # ## 直接跟在一级标题下、中间缺一层时，层级不能串位
    text = "# A\n\n### C\n\n正文内容在这里。\n"
    chunks = split_text(text, doc=DOC)
    assert chunks[0].heading == ("A", "C")


def test_heading_path_resets_on_sibling():
    text = (
        "# A\n\n## B\n\n正文一。\n\n## C\n\n正文二。\n"
    )
    chunks = split_text(text, doc=DOC)
    assert chunks[0].heading == ("A", "B")
    assert chunks[1].heading == ("A", "C")


def test_label_joins_doc_and_heading():
    text = "## 7 天无理由\n\n### 不适用商品\n\n正文。\n"
    chunk = split_text(text, doc=DOC)[0]
    assert chunk.label == "退换货政策 > 7 天无理由 > 不适用商品"


# ---------------------------------------------------------------- 长度与重叠


def test_every_chunk_within_max_chars():
    # 这是切分器最重要的不变量：块长受控，否则向量化和提示词都会爆
    body = "这是一句用来凑长度的话。" * 120
    chunks = split_text(f"## 长篇\n\n{body}\n", doc=DOC, max_chars=300, overlap=60)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 300 for c in chunks)


def test_long_paragraph_split_at_sentence_boundary():
    body = "".join(f"第{i}条规则说明内容在此处。" for i in range(40))
    chunks = split_text(f"## 规则\n\n{body}\n", doc=DOC, max_chars=200, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        # 除最后一块外，都应当切在句末标点上，而不是从句子里劈开
        assert chunk.text.endswith("。"), chunk.text[-12:]


def test_overlap_repeats_tail_of_previous_chunk():
    body = "".join(f"第{i}条：签收后七日内可申请退货。" for i in range(30))
    chunks = split_text(f"## 规则\n\n{body}\n", doc=DOC, max_chars=200, overlap=40)
    assert len(chunks) > 2
    # 用整句比对：拿尾巴 20 个字符去比，边界正好落在半个句子上时会误报
    last_sentence = chunks[0].text.split("\n")[-1]
    assert last_sentence in chunks[1].text


def test_overlap_larger_than_max_chars_rejected():
    # 不拦的话打包逻辑会退化成块块互相重叠，直接死循环式的膨胀
    with pytest.raises(ValueError):
        split_text("## A\n\n正文。\n", doc=DOC, max_chars=100, overlap=100)


def test_negative_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("## A\n\n正文。\n", doc=DOC, overlap=-1)


def test_zero_max_chars_rejected():
    with pytest.raises(ValueError):
        split_text("## A\n\n正文。\n", doc=DOC, max_chars=0)


# ---------------------------------------------------------------- 无标点 / 硬切


def test_text_without_punctuation_still_bounded():
    # 一整行表格数据、乱码、连续编号——没有标点也不能切出超长块
    body = "A1B2C3D4E5" * 200
    chunks = split_text(f"## 数据\n\n{body}\n", doc=DOC, max_chars=250, overlap=30)
    assert chunks
    assert all(len(c.text) <= 250 for c in chunks)
    assert "A1B2C3D4E5" in chunks[0].text


def test_hard_wrap_keeps_all_content():
    body = "X" * 1000
    chunks = split_text(f"## 数据\n\n{body}\n", doc=DOC, max_chars=200, overlap=50)
    assert sum(c.text.count("X") for c in chunks) >= 1000


def test_sentence_split_keeps_trailing_quote_with_sentence():
    # 「好。」里的句号不是句子边界，切在它后面会让下一句以 」 开头
    assert split_sentences("他说「好。」然后走了。") == ["他说「好。」", "然后走了。"]


def test_ascii_period_is_not_a_boundary():
    # 刻意不把 "." 当句末：政策文档里全是 3.5 折、v1.2、7.5 折、A.1 条款这类写法，
    # 按半角句点切会把数字和版本号劈开，检索时会变成两个都不匹配的碎片。
    assert split_sentences("有效期为 3.5 年。下一句。") == ["有效期为 3.5 年。", "下一句。"]
    assert split_sentences("升级到 v1.2 之后再说") == ["升级到 v1.2 之后再说"]


def test_sentence_split_handles_exclamation_and_question():
    assert split_sentences("真的吗？不可能！好吧；那就这样。") == [
        "真的吗？",
        "不可能！",
        "好吧；",
        "那就这样。",
    ]


# ---------------------------------------------------------------- 块边界


def test_list_items_become_separate_blocks_when_they_do_not_fit():
    body = "\n".join(f"- 第{i}项：这一类商品不支持无理由退货，请下单前确认。" for i in range(10))
    chunks = split_text(f"## 不适用\n\n{body}\n", doc=DOC, max_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_list_intro_stays_with_first_item():
    body = "以下商品不适用：\n- 定制类\n- 生鲜类\n"
    chunks = split_text(f"## 不适用\n\n{body}\n", doc=DOC, max_chars=500)
    assert len(chunks) == 1
    assert "以下商品不适用" in chunks[0].text


# ---------------------------------------------------------------- 整个目录


def test_split_corpus_uses_h1_as_doc_name_and_strips_it(tmp_path):
    (tmp_path / "a.md").write_text(
        "# 退换货政策\n\n## 7 天无理由\n\n签收后 7 天内可退。\n", encoding="utf-8"
    )
    chunks = split_corpus(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].doc == "退换货政策"
    # H1 已经当文档名了，标题路径里不该再出现一次
    assert chunks[0].heading == ("7 天无理由",)
    assert chunks[0].label == "退换货政策 > 7 天无理由"


def test_split_corpus_falls_back_to_filename(tmp_path):
    (tmp_path / "shipping.md").write_text("## 时效\n\n一线城市次日达。\n", encoding="utf-8")
    chunks = split_corpus(tmp_path)
    assert chunks[0].doc == "shipping"


def test_split_corpus_is_deterministic_and_ids_unique(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.md").write_text(
            f"# 文档{i}\n\n## 小节\n\n" + "内容。" * 80 + "\n", encoding="utf-8"
        )
    first = split_corpus(tmp_path, max_chars=200, overlap=40)
    second = split_corpus(tmp_path, max_chars=200, overlap=40)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert len(first) > 3


def test_split_corpus_rejects_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        split_corpus(tmp_path / "nope")


def test_default_max_chars_is_reasonable():
    # 别被人悄悄改成 20 或者 5000
    assert 100 <= DEFAULT_MAX_CHARS <= 1000
