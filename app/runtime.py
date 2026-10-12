"""运行期共享的东西：数据库连接和检索器。

做成模块级单例是刻意的。工具函数会被大模型按任意顺序调用，
把连接一层层当参数传下去，只会让工具的签名变脏——而那个签名是要给模型看的，
多一个用不上的参数就多一次它填错的机会。测试里用 set_* 换成内存对象。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable

from .config import Settings, load_dotenv_file
from .policy import now_utc
from .store import db, queries

_settings: Settings | None = None
_connection: sqlite3.Connection | None = None
_retriever = None
_clock: Callable[[], datetime] | None = None
_agent = None


def now() -> datetime:
    """工具算日期窗口时用的「现在」。

    做成可替换的，是因为日期类断言必须能固定时间：否则「签收后第 7 天」
    这种测试的结果会随运行时刻漂移，而且本机时区和 UTC 差 8 小时，
    跨不跨天都能让结果变。
    """
    return _clock() if _clock is not None else now_utc()


def set_clock(value: Callable[[], datetime] | None) -> None:
    global _clock
    _clock = value


def settings() -> Settings:
    global _settings
    if _settings is None:
        load_dotenv_file()
        _settings = Settings.from_env()
    return _settings


def set_settings(value: Settings | None) -> None:
    global _settings
    _settings = value


def get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        conn = db.connect(settings().db_path)
        db.init_schema(conn)
        # 空库自动播种。不这么做的话，刚克隆下来的人直接起服务，会看到一个
        # 能聊天但一个订单都查不到的机器人——他没法判断这是坏了还是本来就没数据。
        # 要用新日期重新造演示数据，跑 scripts/seed.py 即可。
        if queries.count_orders(conn) == 0:
            db.seed(conn)
        _connection = conn
    return _connection


def set_connection(conn: sqlite3.Connection | None) -> None:
    global _connection
    _connection = conn


def get_retriever():
    """懒加载：建索引要读一遍语料，不能在每个请求里做。"""
    global _retriever
    if _retriever is None:
        from .kb.corpus import build_retriever

        _retriever = build_retriever(settings())
    return _retriever


def set_retriever(value) -> None:
    global _retriever
    _retriever = value


def get_agent():
    """整个进程共用一个 Agent。

    模型对象每轮重建的话，HTTP 连接池、重试计数、schema 转换都要重来一遍，
    而这些恰恰是每次请求里最贵又不产生价值的部分。
    """
    global _agent
    if _agent is None:
        from .agent.llm import build_model
        from .agent.loop import Agent
        from .agent.tools import TOOLS

        cfg = settings()
        _agent = Agent(
            build_model(cfg),
            get_retriever(),
            TOOLS,
            refusal_threshold=cfg.refusal_threshold,
        )
    return _agent


def set_agent(value) -> None:
    global _agent
    _agent = value


def reset() -> None:
    global _settings, _connection, _retriever, _clock, _agent
    _settings = None
    _connection = None
    _retriever = None
    _clock = None
    _agent = None
