"""售后规则。

这个文件里的断言几乎都是「差一天」「差一分钱」这类边界。
规则算错在演示时看不出来，只有在真实对账或者面试官追问时才会暴露，
所以宁可在这里写细一点。

所有测试都注入了固定的 now，不依赖运行时刻。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.policy import (
    NO_REASON,
    QUALITY,
    allocate_coupon,
    calendar_days_since,
    evaluate_exchange,
    evaluate_return,
    item_paid_cents,
    late_ship_compensation,
    parse_time,
    refund_breakdown,
    yuan,
)
from app.store import db, queries

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)

# 播种数据里的「签收后第 N 天」都相对 NOW 生成
SEEDED = [
    "SO20261001001",  # 签收 2 天，普通商品，包邮，有 10 元券
    "SO20260920002",  # 签收 20 天，普通商品
    "SO20261005003",  # 签收 3 天，定制商品
    "SO20261008004",  # 签收 1 天，生鲜
    "SO20261010005",  # 待发货，已超时 52 小时
    "SO20261012006",  # 签收当天，普通商品，非包邮
    "SO20261002007",  # 签收 8 天，大家电
    "SO20260915008",  # 签收 35 天，普通商品
    "SO20260928009",  # 签收 12 天，已激活数码
    "SO20261011010",  # 签收 2 天，特价清仓
    "SO20261009011",  # 签收 1 天，两件商品 + 50 元券
]


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.seed(connection, now=NOW)
    return connection


def order_of(conn, order_id: str) -> dict:
    order = queries.get_order(conn, order_id)
    assert order is not None, order_id
    return order


# ------------------------------------------------------------- 基础工具


def test_yuan_formats_two_decimals():
    assert yuan(24100) == "241.00"
    assert yuan(0) == "0.00"
    assert yuan(5) == "0.05"


def test_parse_time_tolerates_missing_timezone():
    # 手写测试数据常常忘了带时区，不能因此抛异常
    assert parse_time("2026-10-13T12:00:00").tzinfo is not None
    assert parse_time(None) is None


def test_calendar_days_uses_dates_not_duration():
    # 今天 00:30 签收、现在 23:30，是「第 0 天」而不是第 1 天
    signed = datetime(2026, 10, 13, 0, 30, tzinfo=timezone.utc)
    assert calendar_days_since(signed.isoformat(), NOW) == 0
    assert calendar_days_since(None, NOW) is None


# ------------------------------------------------------------- 优惠券分摊


def test_coupon_allocation_sums_exactly_to_coupon(conn):
    order = order_of(conn, "SO20261009011")
    shares = allocate_coupon(order["items"], order["coupon_cents"])
    assert sum(shares.values()) == order["coupon_cents"] == 5000


def test_coupon_allocation_is_proportional(conn):
    order = order_of(conn, "SO20261009011")
    shares = allocate_coupon(order["items"], 5000)
    bed, lamp = order["items"]
    # 25900 : 15900 ≈ 1.63 : 1
    assert shares[bed["id"]] == 3098
    assert shares[lamp["id"]] == 1902


def test_coupon_allocation_caps_at_goods_total(conn):
    # 券比商品还贵时按商品总额封顶，否则会算出负的实付金额
    order = order_of(conn, "SO20261009011")
    shares = allocate_coupon(order["items"], 999999)
    assert sum(shares.values()) == 41800
    assert all(share >= 0 for share in shares.values())


def test_coupon_allocation_handles_no_items_and_no_coupon(conn):
    assert allocate_coupon([], 500) == {}
    order = order_of(conn, "SO20261012006")
    assert allocate_coupon(order["items"], 0) == {order["items"][0]["id"]: 0}


def test_item_paid_is_goods_minus_its_share(conn):
    order = order_of(conn, "SO20261009011")
    shares = allocate_coupon(order["items"], 5000)
    assert item_paid_cents(order["items"][0], shares) == 22802
    assert item_paid_cents(order["items"][1], shares) == 13998


def test_paid_amount_equals_sum_of_item_paid(conn):
    # 这条最值钱：明细加起来必须等于订单实付，否则对账没法解释
    order = order_of(conn, "SO20261009011")
    shares = allocate_coupon(order["items"], order["coupon_cents"])
    assert sum(item_paid_cents(i, shares) for i in order["items"]) == order["paid_cents"]


# ------------------------------------------------------------- 退货资格


def test_within_no_reason_window(conn):
    result = evaluate_return(order_of(conn, "SO20261001001"), NO_REASON, NOW)
    assert result["eligible"] is True
    assert result["window_days"] == 7
    assert result["days_since_signed"] == 2
    assert result["days_left"] == 5


def test_last_day_of_window_is_still_eligible(conn):
    # 播种时签收时间 = NOW − 2 天，所以 NOW + 5 天正好是「签收后第 7 天」
    order = order_of(conn, "SO20261001001")
    day7 = NOW + timedelta(days=5)
    assert calendar_days_since(order["signed_at"], day7) == 7
    last = evaluate_return(order, NO_REASON, day7)
    assert last["eligible"] is True
    assert last["days_left"] == 0

    day8 = NOW + timedelta(days=6)
    assert calendar_days_since(order["signed_at"], day8) == 8
    after = evaluate_return(order, NO_REASON, day8)
    assert after["expired"] is True and after["eligible"] is False


def test_day_after_window_is_expired(conn):
    # 签收后第 8 天：无理由超了，质量问题还在 15 天里
    order = order_of(conn, "SO20261001001")
    day8 = NOW + timedelta(days=6)
    no_reason = evaluate_return(order, NO_REASON, day8)
    quality = evaluate_return(order, QUALITY, day8)
    assert no_reason["expired"] is True and no_reason["eligible"] is False
    assert quality["eligible"] is True and quality["window_days"] == 15


def test_big_appliance_gets_thirty_days(conn):
    order = order_of(conn, "SO20261002007")
    result = evaluate_return(order, QUALITY, NOW)
    assert result["eligible"] is True
    assert result["window_days"] == 30
    assert "大家电" in result["policy"]
    # 8 天已经超出 7 天无理由了
    assert evaluate_return(order, NO_REASON, NOW)["expired"] is True


def test_beyond_every_window_is_rejected(conn):
    result = evaluate_return(order_of(conn, "SO20260915008"), QUALITY, NOW)
    assert result["eligible"] is False
    assert "保修" in result["message"]


@pytest.mark.parametrize(
    "order_id,keyword",
    [
        ("SO20261005003", "定制"),
        ("SO20261008004", "生鲜"),
        ("SO20260928009", "数码"),
        ("SO20261011010", "清仓"),
    ],
)
def test_excluded_kinds_block_no_reason_return(conn, order_id, keyword):
    result = evaluate_return(order_of(conn, order_id), NO_REASON, NOW)
    assert result["eligible"] is False
    assert result["blocked_items"] and keyword in result["blocked_items"][0]["why"]
    # 但不影响质量问题通道
    assert evaluate_return(order_of(conn, order_id), QUALITY, NOW)["eligible"] is True


def test_unshipped_order_goes_to_cancel_not_return(conn):
    result = evaluate_return(order_of(conn, "SO20261010005"), NO_REASON, NOW)
    assert result["eligible"] is False
    assert "取消订单" in result["message"]


def test_unsigned_order_is_rejected(conn):
    order = order_of(conn, "SO20261001001")
    order["signed_at"] = None
    result = evaluate_return(order, NO_REASON, NOW)
    assert result["eligible"] is False
    assert "签收" in result["message"]


def test_unknown_reason_raises(conn):
    with pytest.raises(ValueError, match="reason"):
        evaluate_return(order_of(conn, "SO20261001001"), "换个心情", NOW)


# ------------------------------------------------------------- 退款金额


def test_free_shipping_no_reason_deducts_eight_yuan(conn):
    # 259.00 商品 − 10.00 优惠券 = 249.00 实付，包邮再扣 8.00 发货运费 → 241.00
    result = refund_breakdown(order_of(conn, "SO20261001001"), NO_REASON, NOW)
    assert result["goods_paid_cents"] == 24900
    assert result["deduction_cents"] == 800
    assert result["refund_cents"] == 24100
    assert result["refund_yuan"] == "241.00"


def test_non_free_shipping_no_reason_does_not_double_charge(conn):
    # 非包邮：运费本来就不在商品实付里，退款不该再扣一次
    result = refund_breakdown(order_of(conn, "SO20261012006"), NO_REASON, NOW)
    assert result["deduction_cents"] == 0
    assert result["refund_cents"] == 19900
    assert any("不予退还" in note for note in result["notes"])


def test_quality_refund_has_no_deduction(conn):
    result = refund_breakdown(order_of(conn, "SO20261001001"), QUALITY, NOW)
    assert result["deduction_cents"] == 0
    assert result["refund_cents"] == 24900


def test_quality_refund_does_not_subsidize_shipping_twice(conn):
    result = refund_breakdown(order_of(conn, "SO20261012006"), QUALITY, NOW)
    assert result["refund_cents"] == 19900  # 不加回 10 元运费，也不扣
    assert any("平台承担" in note for note in result["notes"])


def test_refund_of_whole_two_item_order_equals_paid_minus_deduction(conn):
    order = order_of(conn, "SO20261009011")
    result = refund_breakdown(order, NO_REASON, NOW)
    assert result["goods_paid_cents"] == order["paid_cents"] == 36800
    assert result["refund_cents"] == 36800 - 800


def test_refund_of_single_item_uses_its_own_coupon_share(conn):
    order = order_of(conn, "SO20261009011")
    result = refund_breakdown(order, NO_REASON, NOW, skus=["SKU-BED-18"])
    assert result["goods_paid_cents"] == 22802
    assert result["refund_cents"] == 22802 - 800


def test_unknown_sku_is_reported_not_silently_ignored(conn):
    result = refund_breakdown(order_of(conn, "SO20261009011"), NO_REASON, NOW, skus=["SKU-NOPE"])
    assert result["refund_cents"] == 0
    assert result["missing_skus"] == ["SKU-NOPE"]
    assert any("没有这些商品" in note for note in result["notes"])


def test_excluded_item_is_left_out_of_the_refund(conn):
    result = refund_breakdown(order_of(conn, "SO20261005003"), NO_REASON, NOW)
    assert result["refund_cents"] == 0
    assert result["excluded"] == ["定制刻字马克杯"]
    assert any("未计入退款" in note for note in result["notes"])


def test_expired_order_refunds_nothing(conn):
    result = refund_breakdown(order_of(conn, "SO20260915008"), QUALITY, NOW)
    assert result["refund_cents"] == 0
    assert result["lines"] == []
    assert "保修" in result["message"]
    assert "保修" in result["notes"][0]


def test_unshipped_order_refunds_nothing(conn):
    result = refund_breakdown(order_of(conn, "SO20261010005"), NO_REASON, NOW)
    assert result["refund_cents"] == 0
    assert "取消订单" in result["message"]


def test_refund_never_goes_negative(conn):
    # 1.00 元的包邮商品走无理由退货，要扣 8 元发货运费，退款不能变成负数
    order = order_of(conn, "SO20261001001")
    order["coupon_cents"] = 0
    order["items"][0]["unit_price_cents"] = 100
    result = refund_breakdown(order, NO_REASON, NOW)
    assert result["goods_paid_cents"] == 100
    assert result["deduction_cents"] == 800
    assert result["refund_cents"] == 0


def test_partial_return_notes_coupon_is_not_returned(conn):
    result = refund_breakdown(order_of(conn, "SO20261001001"), NO_REASON, NOW)
    assert any("优惠券不返还" in note for note in result["notes"])


# ------------------------------------------------------------- 换货


def test_exchange_ok_when_in_stock_within_window(conn):
    order = order_of(conn, "SO20261001001")
    stock = {"SKU-BED-18": queries.get_stock(conn, "SKU-BED-18")}
    result = evaluate_exchange(order, stock, NOW)
    assert result["applicable"] is True
    assert result["items"][0]["can_exchange"] is True
    assert "平台承担" in result["message"]


def test_exchange_blocked_when_out_of_stock(conn):
    order = order_of(conn, "SO20261012006")
    stock = {"SKU-PAN-28": queries.get_stock(conn, "SKU-PAN-28")}
    result = evaluate_exchange(order, stock, NOW)
    assert result["applicable"] is False
    assert "只能选择退货" in result["items"][0]["blocker"]


def test_exchange_blocked_for_fresh_goods(conn):
    order = order_of(conn, "SO20261008004")
    stock = {"SKU-FRUIT-01": queries.get_stock(conn, "SKU-FRUIT-01")}
    result = evaluate_exchange(order, stock, NOW)
    assert result["applicable"] is False
    assert "生鲜" in result["items"][0]["blocker"]


def test_exchange_window_is_fifteen_days(conn):
    order = order_of(conn, "SO20260920002")  # 签收 20 天
    result = evaluate_exchange(order, {}, NOW)
    assert result["applicable"] is False
    assert "保修" in result["message"]


def test_exchange_missing_stock_row_is_reported(conn):
    order = order_of(conn, "SO20261001001")
    result = evaluate_exchange(order, {}, NOW)
    assert result["items"][0]["can_exchange"] is False
    assert "库存记录" in result["items"][0]["blocker"]


# ------------------------------------------------------------- 超时未发货赔付


def test_late_ship_compensation_is_five_percent(conn):
    result = late_ship_compensation(order_of(conn, "SO20261010005"), NOW)
    assert result["applicable"] is True
    assert result["amount_cents"] == 695  # 139.00 × 5%
    assert result["amount_yuan"] == "6.95"


def _fake_order(paid_cents: int, hours_ago: float = 100, promised: int = 48) -> dict:
    return {
        "status": "pending_ship",
        "created_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "paid_cents": paid_cents,
        "promised_ship_hours": promised,
    }


def test_late_ship_compensation_floor_and_ceiling():
    assert late_ship_compensation(_fake_order(1000), NOW)["amount_cents"] == 500
    assert late_ship_compensation(_fake_order(200000), NOW)["amount_cents"] == 5000


def test_late_ship_not_applicable_before_grace_period():
    result = late_ship_compensation(_fake_order(13900, hours_ago=60), NOW)
    assert result["applicable"] is False
    assert result["overdue_hours"] == 12.0


def test_late_ship_not_applicable_for_shipped_order(conn):
    result = late_ship_compensation(order_of(conn, "SO20261001001"), NOW)
    assert result["applicable"] is False
    assert "已经发货" in result["message"]
