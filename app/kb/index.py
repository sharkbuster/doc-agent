"""向量索引。

两套后端，同一套接口：

- **faiss**：装了就用 IndexFlatIP。向量在建索引前已经 L2 归一化，
  所以内积就是余弦相似度，不需要额外算模长。
- **numpy**：没装 faiss 时退化成一次矩阵乘法。几百个片段的规模它一点都不慢，
  留着是为了让「某个平台装不上 faiss」不至于让整个项目跑不起来。

不落盘。片段只有几百条时重建一次是毫秒级的事，而持久化要额外处理
「语料改了但索引没重建」的一致性问题，收益不抵麻烦。
"""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, vectors: np.ndarray, backend: str = "auto") -> None:
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if matrix.ndim != 2:
            raise ValueError(f"向量矩阵要二维，拿到的是 {matrix.ndim} 维")
        self.vectors = matrix
        self.size, self.dim = matrix.shape if matrix.size else (0, matrix.shape[1] if matrix.ndim == 2 else 0)
        self.backend = self._pick_backend(backend)
        self._faiss = None
        if self.backend == "faiss" and self.size:
            import faiss

            index = faiss.IndexFlatIP(self.dim)
            index.add(matrix)
            self._faiss = index

    @staticmethod
    def _pick_backend(backend: str) -> str:
        if backend == "numpy":
            return "numpy"
        if backend not in ("auto", "faiss"):
            raise ValueError(f"未知的索引后端: {backend}")
        try:
            import faiss  # noqa: F401
        except ImportError:
            return "numpy"
        return "faiss"

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """返回 [(下标, 余弦分)]，按分数从高到低。"""
        if self.size == 0 or k <= 0:
            return []
        k = min(k, self.size)
        vec = np.ascontiguousarray(np.asarray(query, dtype=np.float32)).reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f"查询向量 {vec.shape[1]} 维，索引是 {self.dim} 维")

        if self._faiss is not None:
            scores, ids = self._faiss.search(vec, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

        scores = (self.vectors @ vec[0]).astype(np.float64)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]
