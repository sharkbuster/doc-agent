"""文本向量化。

为什么默认不用 sentence-transformers / bge：

- 会把克隆者拖进 2GB 级别的 torch 依赖，CI 从一分钟变成十分钟，
  大多数人走到这一步就放弃了；
- DeepSeek 没有 embeddings 接口，走 API 向量就得再申请一家的密钥，
  和「克隆下来直接能跑」这个目标直接冲突。

所以内置一个自己实现的：字符 n-gram → 特征哈希 → TF 亚线性缩放 → IDF 加权 → L2 归一化。
效果当然比不上 bge，但本仓库的场景是「在同一套政策文档里区分小节」，
它够用，而且有评测集上的数字撑腰（见 README 的检索指标表）。

想要更好的效果就把 EMBEDDING_BACKEND 切到 api，接口是 OpenAI 兼容的。
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize

DEFAULT_DIM = 1024


def _bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, corpus: Sequence[str]) -> None: ...
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """把 n-gram 哈希进固定维度的桶里，不需要词表，也就不存在 OOV。"""

    def __init__(self, dim: int = DEFAULT_DIM, sizes: tuple[int, ...] = (1, 2)) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正数")
        self.dim = dim
        self.sizes = sizes
        self.name = f"hashing-{dim}d"
        self.idf: np.ndarray | None = None

    def fit(self, corpus: Sequence[str]) -> None:
        """统计每个桶的文档频率，算 IDF。

        没有 IDF 的话，高频字（的、是、可）会在每个片段里都贡献权重，
        稠密相似度会被它们拉平，检索质量还不如 BM25——这是实测出来的。
        """
        dim = self.dim
        df = np.zeros(dim, dtype=np.float64)
        for text in corpus:
            buckets = {_bucket(t) % dim for t in tokenize(text, self.sizes)}
            for b in buckets:
                df[b] += 1.0
        n = max(len(corpus), 1)
        # 平滑过的 IDF：出现过一次的桶权重最大，全都出现的桶权重趋近 1
        self.idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        idf = self.idf
        for row, text in enumerate(texts):
            counts = Counter(_bucket(t) % self.dim for t in tokenize(text, self.sizes))
            if not counts:
                continue
            for bucket, count in counts.items():
                # 亚线性 TF：一个词出现 10 次不该比出现 1 次重要 10 倍
                weight = 1.0 + np.log(count)
                if idf is not None:
                    weight *= float(idf[bucket])
                out[row, bucket] = weight
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class ApiEmbedder:
    """OpenAI 兼容的 /embeddings 接口。

    只在 EMBEDDING_BACKEND=api 且三项配置都给全时才用得上。
    没有 fit：向量维度由服务端决定，IDF 也不需要。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dim: int = 0,
        batch_size: int = 32,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY 没配")
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL 没配")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.name = f"api-{model}"

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                if not batch:
                    continue
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in data)
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.dim and matrix.size and matrix.shape[1] != self.dim:
            raise ValueError(f"服务端返回 {matrix.shape[1]} 维，配置写的是 {self.dim} 维")
        if matrix.size:
            self.dim = matrix.shape[1]
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            # 有些网关返回的向量没归一化，这里统一处理，检索时才能直接用内积当余弦
            np.divide(matrix, norms, out=matrix, where=norms > 0)
        return matrix


def build_embedder(settings) -> Embedder:
    """按配置造 embedder。没配 api 就老实用内置的。"""
    if settings.embedding_backend == "api":
        return ApiEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return HashingEmbedder(dim=settings.embedding_dim)
