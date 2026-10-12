"""检索：BM25 与稠密向量各自排一遍，再用 RRF 融合。

几个刻意的选择：

- **融合用 RRF（倒数排名融合）而不是加权求和**。BM25 的分数量纲没有上界，
  余弦分在 [0,1]，两者直接加权得先做归一化，而归一化又会把
  「这次查询整体都很差」这个信息抹掉——那正是拒答要用的信号。
  RRF 只看名次，天然免疫量纲问题。

- **拒答判据用稠密余弦的最大值，不用融合分**。融合分是名次的函数，
  检索质量整体很烂时它照样能给出很高的值；余弦是有绝对意义的相似度，
  才能用来判断「知识库里到底有没有」。

- **保留 dense / lexical / hybrid 三种模式**。不是为了炫技，
  是为了能在评测集上把三者拉出来对比——README 里的指标表就是这么来的。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chunker import Chunk
from .text import tokenize

RRF_K = 60
_FETCH_FACTOR = 4
_FETCH_MIN = 20


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    fused: float
    dense: float
    lexical: float
    dense_rank: int | None
    lexical_rank: int | None


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    top_dense: float
    has_dense: bool = True

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None

    def below(self, threshold: float) -> bool:
        """知识库里没有相关内容——该拒答，而不是让模型硬编一个。

        只有稠密检索才给出有绝对意义的相似度，所以纯词法模式下这个方法没有依据，
        宁可报错也不返回一个看起来能用、实际取决于 BM25 量纲的布尔值。
        """
        if not self.has_dense:
            raise ValueError("词法检索模式没有稠密分，判断不了是否该拒答")
        return self.top_dense < threshold


class BM25:
    """标准 BM25（k1/b 可调）。语料是几百个片段，所以直接用最朴素的写法，
    可读性比常数优化值钱；真到几万片段再上倒排表。"""

    def __init__(
        self,
        docs: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        # 加 1 的写法保证 IDF 非负：某个词在所有片段里都出现时它不该有惩罚性的负权重
        self.idf = {t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(doc) for doc in docs]

    def score(self, tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        if not self.n or not tokens:
            return scores
        avg = self.avg_len or 1.0
        for token in set(tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                freq = tf.get(token)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / avg)
                scores[i] += idf * freq * (self.k1 + 1.0) / denom
        return scores


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder=None,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        rrf_k: int = RRF_K,
        backend: str = "auto",
    ) -> None:
        if mode not in ("hybrid", "dense", "lexical"):
            raise ValueError(f"未知检索模式: {mode}")
        self.chunks = list(chunks)
        self.mode = mode
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.embedder = embedder
        self.lexical = BM25([tokenize(c.text) for c in self.chunks])
        self.index = None
        if mode != "lexical" and self.chunks:
            if embedder is None:
                raise ValueError("dense/hybrid 模式需要 embedder")
            embedder.fit([c.text for c in self.chunks])
            from .index import VectorIndex

            self.index = VectorIndex(embedder.embed([c.text for c in self.chunks]), backend=backend)

    @property
    def backend(self) -> str:
        return self.index.backend if self.index else "none"

    def search(self, query: str, k: int | None = None) -> SearchResult:
        k = self.top_k if k is None else k
        has_dense = self.index is not None
        query_tokens = tokenize(query)
        if not self.chunks or k <= 0 or not query_tokens:
            # 查不出词就没有可检索的东西。不拦住的话，全零查询向量和每个片段的
            # 余弦都是 0，排序退化成「按下标」，会返回一堆纯属凑数的来源。
            return SearchResult(query=query, hits=[], top_dense=0.0, has_dense=has_dense)

        fetch = min(len(self.chunks), max(k * _FETCH_FACTOR, _FETCH_MIN))
        dense_ranked: list[tuple[int, float]] = []
        top_dense = 0.0
        if self.index is not None:
            raw = self.index.search(self.embedder.embed([query])[0], fetch)
            if raw:
                # 索引返回的是降序，所以第一个就是全局最高的余弦分
                top_dense = raw[0][1]
            # 余弦非正说明两者正交甚至相反，不是弱相关，是无关，直接不算候选
            dense_ranked = [(i, s) for i, s in raw if s > 0]

        lexical_ranked: list[tuple[int, float]] = []
        if self.mode != "dense":
            scores = self.lexical.score(query_tokens)
            order = np.argsort(-scores, kind="stable")[:fetch]
            lexical_ranked = [(int(i), float(scores[i])) for i in order if scores[i] > 0]

        if self.mode == "dense":
            lexical_ranked = []
        elif self.mode == "lexical":
            dense_ranked = []

        dense_pos = {idx: r for r, (idx, _) in enumerate(dense_ranked, start=1)}
        lexical_pos = {idx: r for r, (idx, _) in enumerate(lexical_ranked, start=1)}
        dense_score = dict(dense_ranked)
        lexical_score = dict(lexical_ranked)

        fused: list[tuple[float, int]] = []
        for idx in set(dense_pos) | set(lexical_pos):
            total = 0.0
            if idx in dense_pos:
                total += 1.0 / (self.rrf_k + dense_pos[idx])
            if idx in lexical_pos:
                total += 1.0 / (self.rrf_k + lexical_pos[idx])
            fused.append((total, idx))
        # 融合分相同时按下标升序，保证同一份语料每次排序结果一致
        fused.sort(key=lambda item: (-item[0], item[1]))

        hits = [
            Hit(
                chunk=self.chunks[idx],
                fused=score,
                dense=dense_score.get(idx, 0.0),
                lexical=lexical_score.get(idx, 0.0),
                dense_rank=dense_pos.get(idx),
                lexical_rank=lexical_pos.get(idx),
            )
            for score, idx in fused[:k]
        ]
        return SearchResult(query=query, hits=hits, top_dense=top_dense, has_dense=has_dense)
