"""运行配置。

全部走环境变量，默认值保证「什么都不配也能跑起来」——这是刻意的：
一个开源仓库如果需要先申请密钥才能 pytest，那它基本不会有人跑。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"


def load_dotenv_file() -> None:
    """有 .env 就读，没装 python-dotenv 也不报错。

    不想为了读一个文件把 dotenv 变成硬依赖：CI 里直接给环境变量更省事。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 需要整数，实际是 {raw!r}") from None


def _float(name: str, default: float) -> float:
    raw = _str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"环境变量 {name} 需要小数，实际是 {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_backend: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int
    db_path: str
    retrieval_mode: str
    dense_weight: float
    top_k: int
    refusal_signal: str
    refusal_threshold: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()
        return cls(
            llm_provider=_str("LLM_PROVIDER", "stub").lower(),
            deepseek_api_key=_str("DEEPSEEK_API_KEY"),
            deepseek_base_url=_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            # 官方现在的模型名是 deepseek-flash。旧的 deepseek-chat 仍然接受
            # （官方文档写明由同一个模型服务、按同一价格计费），所以老配置不会被改坏。
            deepseek_model=_str("DEEPSEEK_MODEL", "deepseek-flash"),
            openai_api_key=_str("OPENAI_API_KEY"),
            openai_base_url=_str("OPENAI_BASE_URL"),
            openai_model=_str("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_backend=_str("EMBEDDING_BACKEND", "hashing").lower(),
            embedding_api_key=_str("EMBEDDING_API_KEY"),
            embedding_base_url=_str("EMBEDDING_BASE_URL"),
            embedding_model=_str("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=_int("EMBEDDING_DIM", 1024),
            db_path=_str("DB_PATH", str(DATA_DIR / "demo.db")),
            # 默认纯词法检索。这是 scripts/calibrate.py 扫出来的结论，不是偷懒：
            # 内置的字符 n-gram 哈希向量在这个语料上任何权重配置都跑不赢 BM25。
            # 接了真实嵌入模型之后应当把这个值调回 hybrid 并重扫权重。
            retrieval_mode=_str("RETRIEVAL_MODE", "lexical").lower(),
            dense_weight=_float("DENSE_WEIGHT", 0.0),
            top_k=_int("RETRIEVAL_TOP_K", 5),
            refusal_signal=_str("REFUSAL_SIGNAL", "coverage").lower(),
            refusal_threshold=_float("REFUSAL_THRESHOLD", 0.05),
        )

    def llm_target(self) -> tuple[str, str, str]:
        """返回 (api_key, base_url, model)。stub 是空三元组。"""
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key, self.deepseek_base_url, self.deepseek_model
        if self.llm_provider == "openai":
            return self.openai_api_key, self.openai_base_url, self.openai_model
        return "", "", ""

    def public_view(self) -> dict:
        """给 /api/health 用。密钥只暴露「有没有」，绝不回显内容。"""
        _, base_url, model = self.llm_target()
        embedding = (
            self.embedding_model
            if self.embedding_backend == "api"
            else f"hashing-{self.embedding_dim}d"
        )
        return {
            "llm_provider": self.llm_provider,
            "llm_model": model or "stub",
            "llm_base_url": base_url,
            "api_key_configured": bool(self.llm_target()[0]),
            "embedding": embedding,
            "retrieval_mode": self.retrieval_mode,
            "dense_weight": self.dense_weight,
            "top_k": self.top_k,
            "refusal_signal": self.refusal_signal,
            "refusal_threshold": self.refusal_threshold,
        }
