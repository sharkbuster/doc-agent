"""演示数据的播种脚本。

    .venv\\Scripts\\python.exe -m scripts.seed

订单里的时间是相对「执行这一刻」倒推生成的（「签收后第 3 天」这类），
所以隔几天再演示之前重跑一次，日期窗口才会落在合理的位置。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import runtime  # noqa: E402
from app.store import db, queries  # noqa: E402


def main() -> int:
    settings = runtime.settings()
    path = Path(settings.db_path)
    conn = db.connect(path)
    count = db.seed(conn)
    runtime.set_connection(conn)

    print(f"数据库：{path}")
    print(f"已写入 {count} 个订单、{queries.count_orders(conn)} 条记录、"
          f"{len(queries.list_orders(conn))} 条可查询订单")
    print()
    print(f"{'订单号':<16}{'状态':<10}{'实付':>10}  收货城市")
    for row in queries.list_orders(conn):
        label = row["status"]
        print(f"{row['order_id']:<16}{label:<10}{row['paid_cents'] / 100:>10.2f}  {row['receiver_city']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
