"""分词、向量化、索引、检索。

这几个模块的问题都很隐蔽——索引和查询切法不一致、向量没归一化、
FAISS 和 numpy 两套后端给出不同结果——所以测试尽量往「两边必须一致」上压。
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from app.kb.chunker import Chunk
from app.kb.embedder import ApiEmbedder, HashingEmbedder
from app.kb.index import VectorIndex
from app.kb.retriever import BM25, Retriever
from app.kb.text import base_tokens, normalize, to_halfwidth, tokenize


# ------------------------------------------------------------------ 分词


def test_fullwidth_and_halfwidth_are_the_same_token():
    # 微信里粘过来的问题经常是全角，不统一就会出现「３天」查不到「3天」
    assert tokenize("３天无理由") == tokenize("3天无理由")
    assert to_halfwidth("ＡＢ，３") == "AB,3"


def test_normalize_lowercases_and_collapses_space():
    assert normalize("  IPHONE   15\tPro ") == "iphone 15 pro"


def test_ascii_words_stay_whole():
    assert base_tokens("iphone 15 pro") == ["iphone", "15", "pro"]


def test_decimal_number_is_one_token():
    assert "3.5" in base_tokens("有效期 3.5 年")


def test_punctuation_is_dropped():
    assert base_tokens("退货，还是换货？") == list("退货还是换货")


def test_cjk_chars_split_one_by_one():
    assert base_tokens("退货") == ["退", "货"]


def test_bigrams_are_produced():
    assert tokenize("退货", (2,)) == ["退货"]


def test_tokenize_can_drop_unigrams():
    assert "退" not in tokenize("退货", (2,))


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []
    assert tokenize("   ，。！ ") == []


def test_emoji_and_symbols_do_not_crash():
    assert base_tokens("😀🎉 ✔ 退货") == ["退", "货"]


# ------------------------------------------------------------------ 向量化


def test_embed_shape_and_normalization():
    emb = HashingEmbedder(dim=256)
    matrix = emb.embed(["退货政策", "运费谁承担"])
    assert matrix.shape == (2, 256)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_embed_empty_text_is_zero_vector_not_nan():
    matrix = HashingEmbedder(dim=64).embed([""])
    assert matrix.shape == (1, 64)
    assert not np.isnan(matrix).any()
    assert float(np.linalg.norm(matrix[0])) == 0.0


def test_embed_is_deterministic():
    emb = HashingEmbedder(dim=128)
    assert np.array_equal(emb.embed(["同一句话"])[0], HashingEmbedder(dim=128).embed(["同一句话"])[0])


def test_fit_produces_idf_with_expected_length():
    emb = HashingEmbedder(dim=64)
    emb.fit(["退货", "运费", "退货"])
    assert emb.idf is not None and emb.idf.shape == (64,)


def test_related_text_scores_higher_than_unrelated():
    docs = ["签收之日起 7 天内可申请无理由退货", "一线城市次日达，偏远地区三天"]
    emb = HashingEmbedder(dim=512)
    emb.fit(docs)
    vectors = emb.embed(docs + ["无理由退货怎么申请"])
    related = float(vectors[2] @ vectors[0])
    unrelated = float(vectors[2] @ vectors[1])
    assert related > unrelated


def test_dim_must_be_positive():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


# ------------------------------------------------------------------ API embedder


def _mock_httpx(monkeypatch, handler, counter=None):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(**kwargs):
        if counter is not None:
            counter.append(1)
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_api_embedder_normalizes_and_respects_index_order(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        data = [{"index": i, "embedding": [3.0, 4.0]} for i, _ in enumerate(payload["input"])]
        # 故意倒序返回：按 index 排序是调用方的责任
        return httpx.Response(200, json={"data": list(reversed(data))})

    _mock_httpx(monkeypatch, handler)
    emb = ApiEmbedder(api_key="k", base_url="https://example.com/v1", model="m")
    matrix = emb.embed(["a", "b"])
    assert matrix.shape == (2, 2)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    assert np.allclose(matrix[0], [0.6, 0.8], atol=1e-6)


def test_api_embedder_batches_requests(monkeypatch):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(len(payload["input"]))
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(payload["input"]))]},
        )

    _mock_httpx(monkeypatch, handler)
    emb = ApiEmbedder(api_key="k", base_url="https://example.com/v1", model="m", batch_size=32)
    emb.embed([f"t{i}" for i in range(40)])
    assert calls == [32, 8]


def test_api_embedder_requires_key_and_url():
    with pytest.raises(ValueError):
        ApiEmbedder(api_key="", base_url="https://x/v1", model="m")
    with pytest.raises(ValueError):
        ApiEmbedder(api_key="k", base_url="", model="m")


def test_api_embedder_rejects_dim_mismatch(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(payload["input"]))]},
        )

    _mock_httpx(monkeypatch, handler)
    emb = ApiEmbedder(api_key="k", base_url="https://x/v1", model="m", dim=128)
    with pytest.raises(ValueError, match="维"):
        emb.embed(["a"])


# ------------------------------------------------------------------ 索引


@pytest.fixture
def vectors() -> np.ndarray:
    rng = np.random.default_rng(20261013)
    raw = rng.normal(size=(12, 16)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_index_finds_the_most_similar_vector(vectors, backend):
    index = VectorIndex(vectors, backend=backend)
    hits = index.search(vectors[3], k=1)
    assert hits and hits[0][0] == 3
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_numpy_and_faiss_backends_agree(vectors):
    query = vectors[7]
    numpy_hits = VectorIndex(vectors, backend="numpy").search(query, k=5)
    faiss_hits = VectorIndex(vectors, backend="faiss").search(query, k=5)
    assert [i for i, _ in numpy_hits] == [i for i, _ in faiss_hits]
    assert [s for _, s in numpy_hits] == pytest.approx([s for _, s in faiss_hits], abs=1e-5)


def test_auto_backend_prefers_faiss_when_available(vectors):
    # 装了 faiss 就该用它；这条断言的意义是：哪天 faiss 装不上了，
    # 这里会失败，提醒去确认 numpy 兜底是否还在正常工作
    assert VectorIndex(vectors).backend == "faiss"


def test_empty_index_returns_nothing():
    index = VectorIndex(np.zeros((0, 8), dtype=np.float32))
    assert index.size == 0
    assert index.search(np.zeros(8, dtype=np.float32), k=3) == []


def test_index_clamps_k_to_size(vectors):
    assert len(VectorIndex(vectors).search(vectors[0], k=999)) == len(vectors)


def test_index_rejects_dim_mismatch(vectors):
    with pytest.raises(ValueError, match="维"):
        VectorIndex(vectors).search(np.zeros(3, dtype=np.float32), k=1)


def test_index_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        VectorIndex(np.zeros(5, dtype=np.float32))


def test_index_rejects_unknown_backend(vectors):
    with pytest.raises(ValueError):
        VectorIndex(vectors, backend="milvus")


def test_search_with_zero_k_is_empty(vectors):
    assert VectorIndex(vectors).search(vectors[0], k=0) == []


# ------------------------------------------------------------------ 检索


CORPUS = [
    ("退换货政策", ("7 天无理由",), "签收之日起 7 天内，商品不影响二次销售的可申请无理由退货。"),
    ("退换货政策", ("不适用商品",), "定制类商品、生鲜类商品、已拆封的贴身用品不适用 7 天无理由退货。"),
    ("运费说明", ("退货运费",), "无理由退货的运费由买家承担；因商品质量问题退货的，运费由平台承担。"),
    ("发票说明", ("开票时间",), "订单完成后 30 天内可以申请开具电子发票。"),
    ("物流时效", ("配送时间",), "一线城市次日达，偏远地区 3 到 5 天送达。"),
]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", doc=doc, heading=heading, text=text)
        for i, (doc, heading, text) in enumerate(CORPUS)
    ]


def retriever(chunks, **kwargs) -> Retriever:
    kwargs.setdefault("embedder", HashingEmbedder(dim=512))
    return Retriever(chunks, **kwargs)


def test_exact_phrase_query_hits_the_right_chunk(chunks):
    result = retriever(chunks).search("退货的运费谁承担")
    assert result.top is not None
    assert result.top.chunk.doc == "运费说明"


def test_hybrid_uses_both_rankings(chunks):
    result = retriever(chunks).search("定制商品能不能无理由退货")
    labels = {h.chunk.doc for h in result.hits}
    assert "退换货政策" in labels
    assert any(h.dense_rank is not None and h.lexical_rank is not None for h in result.hits)


def test_dense_and_lexical_modes_run_without_the_other(chunks):
    dense = retriever(chunks, mode="dense").search("发票什么时候能开")
    assert dense.hits and dense.hits[0].chunk.doc == "发票说明"
    assert all(h.lexical_rank is None for h in dense.hits)

    lexical = Retriever(chunks, mode="lexical").search("发票什么时候能开")
    assert lexical.hits and lexical.hits[0].chunk.doc == "发票说明"
    assert all(h.dense_rank is None for h in lexical.hits)


def test_lexical_mode_needs_no_embedder(chunks):
    assert Retriever(chunks, mode="lexical").backend == "none"


def test_dense_mode_requires_embedder(chunks):
    with pytest.raises(ValueError, match="embedder"):
        Retriever(chunks, mode="dense")


def test_unknown_mode_rejected(chunks):
    with pytest.raises(ValueError, match="模式"):
        retriever(chunks, mode="magic")


def test_top_dense_is_the_global_maximum(chunks):
    result = retriever(chunks).search("退货运费")
    assert result.top_dense >= max(h.dense for h in result.hits)


def test_top_dense_separates_related_from_unrelated(chunks):
    engine = retriever(chunks)
    related = engine.search("无理由退货的运费谁出").top_dense
    unrelated = engine.search("怎么申请提高信用卡额度").top_dense
    assert related > unrelated


def test_below_threshold_marks_off_topic_questions(chunks):
    engine = retriever(chunks)
    assert engine.search("怎么申请提高信用卡额度").below(0.99) is True
    assert engine.search("无理由退货的运费谁出").below(0.0) is False


def test_below_threshold_raises_without_dense_scores(chunks):
    # 词法模式没有绝对量纲，不能悄悄返回一个「看起来能用」的布尔值
    result = Retriever(chunks, mode="lexical").search("运费")
    assert result.has_dense is False
    with pytest.raises(ValueError, match="拒答"):
        result.below(0.5)


def test_empty_corpus_returns_empty_result():
    result = retriever([]).search("随便问点什么")
    assert result.hits == []
    assert result.top is None
    assert result.top_dense == 0.0


def test_empty_query_returns_no_hits(chunks):
    assert retriever(chunks).search("") .hits == []


def test_retrieval_is_deterministic(chunks):
    query = "定制商品退货"
    first = [h.chunk.id for h in retriever(chunks).search(query).hits]
    second = [h.chunk.id for h in retriever(chunks).search(query).hits]
    assert first == second


def test_k_limits_hits(chunks):
    assert len(retriever(chunks).search("退货", k=2).hits) == 2


def test_zero_k_returns_nothing(chunks):
    assert retriever(chunks).search("退货", k=0).hits == []


def test_bm25_scores_matching_doc_above_non_matching():
    bm = BM25([tokenize("退货运费由买家承担"), tokenize("发票三十天内开具")])
    scores = bm.score(tokenize("运费"))
    assert scores[0] > scores[1]


def test_bm25_handles_empty_input():
    assert BM25([]).score(tokenize("运费")).tolist() == []
    assert BM25([tokenize("运费")]).score([]).tolist() == [0.0]


def test_hits_carry_readable_labels(chunks):
    result = retriever(chunks).search("退货运费")
    assert result.top.chunk.label == "运费说明 > 退货运费"
