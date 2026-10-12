"""工具层的 SQL 集中在这里。

不把 SQL 直接写进工具函数：工具入参来自大模型，出错概率远高于前端，
把「查数据」和「怎么把错误讲给模型听」分开，前者才能单独测。
"""
from __future__ import annotations

import sqlite3


def get_order(conn: sqlite3.Connection, order_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return None
    items = conn.execute(
        "SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,)
    ).fetchall()
    order = dict(row)
    order["items"] = [dict(item) for item in items]
    return order


def get_logistics(conn: sqlite3.Connection, order_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM logistics WHERE order_id = ?", (order_id,)).fetchone()
    return dict(row) if row else None


def get_stock(conn: sqlite3.Connection, sku: str) -> dict | None:
    row = conn.execute("SELECT * FROM stock WHERE sku = ?", (sku,)).fetchone()
    return dict(row) if row else None


def list_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT order_id, status, paid_cents, receiver_city FROM orders"
                        " ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def count_orders(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
