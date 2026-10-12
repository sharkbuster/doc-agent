"""订单库：建表 + 造演示数据。

金额一律用整数「分」。这个选择在别的地方也许还有得争，在这里没有：
「退款 45.30 元」只要用浮点算过一次，和账单对不上就是迟早的事。

时间戳存 UTC 的 ISO 字符串。演示数据里的时间不是写死的日期，而是**按播种那一刻
倒推**出来的（「签收后第 3 天」这类），这样刚播种完演示就是合理的；
隔几天再演示，重跑一次 seed 刷新一下就行。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    paid_cents          INTEGER NOT NULL,
    goods_cents         INTEGER NOT NULL,
    shipping_cents      INTEGER NOT NULL,
    coupon_cents        INTEGER NOT NULL,
    receiver_city       TEXT NOT NULL,
    promised_ship_hours INTEGER NOT NULL,
    shipped_at          TEXT,
    signed_at           TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id          TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    sku               TEXT NOT NULL,
    name              TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    unit_price_cents  INTEGER NOT NULL,
    kind              TEXT NOT NULL,
    is_big_appliance  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);

CREATE TABLE IF NOT EXISTS logistics (
    order_id      TEXT PRIMARY KEY REFERENCES orders(order_id) ON DELETE CASCADE,
    carrier       TEXT NOT NULL,
    tracking_no   TEXT NOT NULL,
    last_event    TEXT NOT NULL,
    last_event_at TEXT NOT NULL,
    delivered     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stock (
    sku               TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    supports_exchange INTEGER NOT NULL DEFAULT 1
);
"""

# kind 决定这件商品能不能走 7 天无理由：
#   normal    普通商品
#   custom    定制类
#   fresh     生鲜易腐
#   opened    已拆封的贴身用品
#   digital   已激活的数码产品
#   clearance 标注「不退不换」的特价清仓
_ORDERS: list[dict] = [
    {
        # 签收 2 天：无理由窗口内，包邮，有优惠券 —— 用来演示「扣 8 元发货运费」
        "id": "SO20261001001", "city": "杭州", "status": "signed",
        "created_hours_ago": 24 * 6, "promised_ship_hours": 48,
        "shipped_hours_after_created": 6, "signed_days_ago": 2,
        "shipping_cents": 0, "coupon_cents": 1000,
        "items": [{"sku": "SKU-BED-18", "name": "纯棉四件套 1.8m 床", "qty": 1,
                   "unit_price_cents": 25900, "kind": "normal"}],
        "logistics": {"carrier": "顺丰速运", "tracking_no": "SF1024001",
                      "last_event": "已签收，感谢使用顺丰", "delivered": True},
    },
    {
        # 签收 20 天：无理由和 15 天质量问题都超了
        "id": "SO20260920002", "city": "武汉", "status": "signed",
        "created_hours_ago": 24 * 24, "promised_ship_hours": 48,
        "shipped_hours_after_created": 10, "signed_days_ago": 20,
        "shipping_cents": 800, "coupon_cents": 0,
        "items": [{"sku": "SKU-SHOE-42", "name": "轻量跑步鞋 42 码", "qty": 1,
                   "unit_price_cents": 32900, "kind": "normal"}],
        "logistics": {"carrier": "中通快递", "tracking_no": "ZT7788002",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
    {
        # 签收 3 天：定制商品，不适用无理由
        "id": "SO20261005003", "city": "上海", "status": "signed",
        "created_hours_ago": 24 * 7, "promised_ship_hours": 72,
        "shipped_hours_after_created": 40, "signed_days_ago": 3,
        "shipping_cents": 0, "coupon_cents": 0,
        "items": [{"sku": "SKU-CUP-DIY", "name": "定制刻字马克杯", "qty": 2,
                   "unit_price_cents": 4900, "kind": "custom"}],
        "logistics": {"carrier": "圆通速递", "tracking_no": "YT5566003",
                      "last_event": "已签收，签收人：前台代收", "delivered": True},
    },
    {
        # 签收 1 天：生鲜，不适用无理由
        "id": "SO20261008004", "city": "成都", "status": "signed",
        "created_hours_ago": 24 * 3, "promised_ship_hours": 24,
        "shipped_hours_after_created": 8, "signed_days_ago": 1,
        "shipping_cents": 1200, "coupon_cents": 0,
        "items": [{"sku": "SKU-FRUIT-01", "name": "阳山水蜜桃 4 只装", "qty": 1,
                   "unit_price_cents": 6800, "kind": "fresh"}],
        "logistics": {"carrier": "京东物流", "tracking_no": "JD9908004",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
    {
        # 待发货且已超过承诺时间 52 小时：超时未发货赔付场景
        "id": "SO20261010005", "city": "西安", "status": "pending_ship",
        "created_hours_ago": 100, "promised_ship_hours": 48,
        "shipped_hours_after_created": None, "signed_days_ago": None,
        "shipping_cents": 0, "coupon_cents": 2000,
        "items": [{"sku": "SKU-LAMP-01", "name": "护眼台灯 国AA级", "qty": 1,
                   "unit_price_cents": 15900, "kind": "normal"}],
        "logistics": None,
    },
    {
        # 签收当天 + 质量问题 + 非包邮：退货运费由平台承担、发货运费要退
        "id": "SO20261012006", "city": "南京", "status": "signed",
        "created_hours_ago": 24 * 4, "promised_ship_hours": 48,
        "shipped_hours_after_created": 12, "signed_days_ago": 0,
        "shipping_cents": 1000, "coupon_cents": 0,
        "items": [{"sku": "SKU-PAN-28", "name": "不粘炒锅 28cm", "qty": 1,
                   "unit_price_cents": 19900, "kind": "normal"}],
        "logistics": {"carrier": "韵达快递", "tracking_no": "YD3312006",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
    {
        # 签收 8 天：普通商品已过 7 天无理由，但大家电有 30 天质量问题窗口
        "id": "SO20261002007", "city": "广州", "status": "signed",
        "created_hours_ago": 24 * 12, "promised_ship_hours": 48,
        "shipped_hours_after_created": 20, "signed_days_ago": 8,
        "shipping_cents": 0, "coupon_cents": 0,
        "items": [{"sku": "SKU-FRIDGE-01", "name": "双门冰箱 452L", "qty": 1,
                   "unit_price_cents": 289900, "kind": "normal", "is_big_appliance": True}],
        "logistics": {"carrier": "日日顺物流", "tracking_no": "RRS7002007",
                      "last_event": "已签收，已安装调试", "delivered": True},
    },
    {
        # 签收 35 天：什么都超了，只能走保修
        "id": "SO20260915008", "city": "兰州", "status": "signed",
        "created_hours_ago": 24 * 40, "promised_ship_hours": 48,
        "shipped_hours_after_created": 30, "signed_days_ago": 35,
        "shipping_cents": 1500, "coupon_cents": 0,
        "items": [{"sku": "SKU-KETTLE-01", "name": "恒温电热水壶 1.7L", "qty": 1,
                   "unit_price_cents": 12900, "kind": "normal"}],
        "logistics": {"carrier": "邮政快递包裹", "tracking_no": "YZ1109008",
                      "last_event": "已签收，签收人：门卫", "delivered": True},
    },
    {
        # 签收 12 天：已激活数码，不适用无理由；15 天质量问题窗口内
        "id": "SO20260928009", "city": "深圳", "status": "signed",
        "created_hours_ago": 24 * 15, "promised_ship_hours": 24,
        "shipped_hours_after_created": 5, "signed_days_ago": 12,
        "shipping_cents": 0, "coupon_cents": 3000,
        "items": [{"sku": "SKU-PHONE-X", "name": "智能手机 X1 256G（已激活）", "qty": 1,
                   "unit_price_cents": 399900, "kind": "digital"}],
        "logistics": {"carrier": "顺丰速运", "tracking_no": "SF6655009",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
    {
        # 签收 2 天 + 特价清仓：标注不退不换
        "id": "SO20261011010", "city": "长沙", "status": "signed",
        "created_hours_ago": 24 * 5, "promised_ship_hours": 48,
        "shipped_hours_after_created": 15, "signed_days_ago": 2,
        "shipping_cents": 0, "coupon_cents": 0,
        "items": [{"sku": "SKU-COAT-XL", "name": "羽绒服 XL（清仓特价）", "qty": 1,
                   "unit_price_cents": 19900, "kind": "clearance"}],
        "logistics": {"carrier": "中通快递", "tracking_no": "ZT2211010",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
    {
        # 一单两件 + 优惠券：专门用来验证优惠券按商品金额比例分摊
        # （单件订单里分摊逻辑永远是「全给它」，测不出比例算错）
        "id": "SO20261009011", "city": "北京", "status": "signed",
        "created_hours_ago": 24 * 4, "promised_ship_hours": 48,
        "shipped_hours_after_created": 9, "signed_days_ago": 1,
        "shipping_cents": 0, "coupon_cents": 5000,
        "items": [
            {"sku": "SKU-BED-18", "name": "纯棉四件套 1.8m 床", "qty": 1,
             "unit_price_cents": 25900, "kind": "normal"},
            {"sku": "SKU-LAMP-01", "name": "护眼台灯 国AA级", "qty": 1,
             "unit_price_cents": 15900, "kind": "normal"},
        ],
        "logistics": {"carrier": "顺丰速运", "tracking_no": "SF8809011",
                      "last_event": "已签收，签收人：本人", "delivered": True},
    },
]

_STOCK: list[tuple[str, str, int, int]] = [
    ("SKU-BED-18", "纯棉四件套 1.8m 床", 42, 1),
    ("SKU-SHOE-42", "轻量跑步鞋 42 码", 0, 1),      # 缺货，换货只能改成退货
    ("SKU-CUP-DIY", "定制刻字马克杯", 15, 1),
    ("SKU-PAN-28", "不粘炒锅 28cm", 0, 1),          # 缺货，用来覆盖「在期限内但没库存」
    ("SKU-LAMP-01", "护眼台灯 国AA级", 120, 1),
    ("SKU-KETTLE-01", "恒温电热水壶 1.7L", 3, 1),
    ("SKU-FRUIT-01", "阳山水蜜桃 4 只装", 0, 0),     # 生鲜，本来就不支持换货
]


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False：接口都是同步 def，FastAPI 会把它们丢到线程池里执行，
    # 于是「建连接的主线程」和「跑查询的 worker 线程」不是同一个，
    # 默认设置下会直接抛 ProgrammingError。
    #
    # 这里敢关掉是因为**这个应用的数据库是只读的**（数据由 scripts/seed.py 写入），
    # 而 CPython 自带的 SQLite 是 serialized 模式，多线程各自调用是安全的。
    # 以后一旦要在请求路径里写库，必须改成每线程一个连接或者加写锁。
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # SQLite 默认不校验外键，不显式打开的话「给不存在的订单插物流」会悄悄成功，
    # 开发和测试环境的表现就会分叉
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def seed(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """重建演示数据。会先清空，所以可以反复调用。"""
    now = now or datetime.now(timezone.utc)
    init_schema(conn)

    # 先删子表再删父表：反过来会撞外键约束
    conn.execute("DELETE FROM order_items")
    conn.execute("DELETE FROM logistics")
    conn.execute("DELETE FROM orders")
    conn.execute("DELETE FROM stock")

    def stamp(delta: timedelta) -> str:
        return (now - delta).isoformat(timespec="seconds")

    for order in _ORDERS:
        goods = sum(item["qty"] * item["unit_price_cents"] for item in order["items"])
        shipping = order["shipping_cents"]
        coupon = order["coupon_cents"]
        # 实付不允许被优惠券抵成负数
        paid = max(goods + shipping - coupon, 0)
        shipped_at = None
        if order["shipped_hours_after_created"] is not None:
            shipped_at = stamp(timedelta(hours=order["created_hours_ago"] - order["shipped_hours_after_created"]))
        signed_at = (
            stamp(timedelta(days=order["signed_days_ago"]))
            if order["signed_days_ago"] is not None
            else None
        )
        conn.execute(
            "INSERT INTO orders (order_id, status, created_at, paid_cents, goods_cents,"
            " shipping_cents, coupon_cents, receiver_city, promised_ship_hours,"
            " shipped_at, signed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                order["id"], order["status"], stamp(timedelta(hours=order["created_hours_ago"])),
                paid, goods, shipping, coupon, order["city"], order["promised_ship_hours"],
                shipped_at, signed_at,
            ),
        )
        for item in order["items"]:
            conn.execute(
                "INSERT INTO order_items (order_id, sku, name, qty, unit_price_cents, kind,"
                " is_big_appliance) VALUES (?,?,?,?,?,?,?)",
                (order["id"], item["sku"], item["name"], item["qty"],
                 item["unit_price_cents"], item["kind"],
                 1 if item.get("is_big_appliance") else 0),
            )
        if order["logistics"]:
            log = order["logistics"]
            conn.execute(
                "INSERT INTO logistics (order_id, carrier, tracking_no, last_event,"
                " last_event_at, delivered) VALUES (?,?,?,?,?,?)",
                (order["id"], log["carrier"], log["tracking_no"], log["last_event"],
                 stamp(timedelta(hours=4)), 1 if log["delivered"] else 0),
            )
    conn.executemany("INSERT INTO stock (sku, name, qty, supports_exchange) VALUES (?,?,?,?)", _STOCK)
    conn.commit()
    return len(_ORDERS)
