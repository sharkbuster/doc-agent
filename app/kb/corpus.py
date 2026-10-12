"""知识库的装配：读语料目录 → 切分 → 建检索器。"""
from __future__ import annotations

from functools import lru_cache

from .chunker import split_corpus
from .embedder import build_embedder
from .retriever import Retriever


def load_chunks(directory=None):
    from ..config import CORPUS_DIR

    return split_corpus(directory or CORPUS_DIR)


def build_retriever(settings, directory=None) -> Retriever:
    chunks = load_chunks(directory)
    if not chunks:
        # 语料目录空了还照常启动的话，每个问题都会走「知识库里没有」，
        # 现象是「答什么都拒答」，极难联想到是语料没加载上
        raise RuntimeError("知识库是空的，检查 data/corpus 下有没有 .md 文件")
    embedder = None if settings.retrieval_mode == "lexical" else build_embedder(settings)
    return Retriever(
        chunks,
        embedder,
        mode=settings.retrieval_mode,
        top_k=settings.top_k,
        dense_weight=settings.dense_weight,
        refusal_signal=settings.refusal_signal,
    )
