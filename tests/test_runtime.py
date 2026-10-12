"""运行期单例的装配。

这些行为单看都很小，但每一条出问题都会表现成「接口莫名其妙返回空数据」，
而且从接口层面完全看不出原因。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app import runtime
from app.config import Settings
from app.store import queries


@pytest.fixture(autouse=True)
def clean():
    runtime.reset()
    yield
    runtime.reset()


def settings_for(tmp_path: Path) -> Settings:
    return replace(Settings.from_env(), db_path=str(tmp_path / "demo.db"))


def test_empty_database_is_seeded_automatically(tmp_path):
    # 刚克隆下来直接起服务，不应该看到一个查不到任何订单的机器人
    runtime.set_settings(settings_for(tmp_path))
    conn = runtime.get_connection()
    assert queries.count_orders(conn) == 11


def test_existing_data_is_not_overwritten(tmp_path):
    runtime.set_settings(settings_for(tmp_path))
    conn = runtime.get_connection()
    conn.execute("DELETE FROM orders WHERE order_id = ?", ("SO20261001001",))
    conn.commit()
    runtime.set_connection(None)

    # 已经有数据（哪怕是残缺的）就不再自动播种，免得把用户改过的库冲掉
    again = runtime.get_connection()
    assert queries.count_orders(again) == 10


def test_connection_is_reused(tmp_path):
    runtime.set_settings(settings_for(tmp_path))
    assert runtime.get_connection() is runtime.get_connection()


def test_database_file_is_created(tmp_path):
    settings = settings_for(tmp_path)
    runtime.set_settings(settings)
    runtime.get_connection()
    assert Path(settings.db_path).is_file()


def test_clock_defaults_to_real_time():
    assert runtime.now().tzinfo is not None


def test_clock_can_be_replaced():
    from datetime import datetime, timezone

    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runtime.set_clock(lambda: fixed)
    assert runtime.now() == fixed


def test_reset_clears_everything(tmp_path):
    runtime.set_settings(settings_for(tmp_path))
    runtime.set_connection(runtime.get_connection())
    runtime.set_clock(lambda: None)
    runtime.set_retriever(object())
    runtime.set_agent(object())
    runtime.reset()
    assert runtime._settings is None
    assert runtime._connection is None
    assert runtime._retriever is None
    assert runtime._agent is None
    assert runtime._clock is None


def test_settings_are_cached_and_replaceable(tmp_path):
    runtime.set_settings(settings_for(tmp_path))
    assert runtime.settings().db_path.endswith("demo.db")
    runtime.set_settings(None)
    assert runtime.settings().db_path  # 重新从环境变量读出来
