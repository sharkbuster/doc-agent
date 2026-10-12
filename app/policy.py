"""售后规则的计算。

这个项目里最容易算错的就是这里：日期窗口的边界、优惠券怎么分摊、扣不扣发货运费。
所以它刻意不依赖 HTTP、不依赖大模型——工具函数只负责把结果讲给模型听，
规则本身可以单独测。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

NO_REASON = "no_reason"
QUALITY = "quality"

REASONS = {
    NO_REASON: "7 天无理由退货",
    QUALITY: "质量问题退换货",
}

NO_REASON_WINDOW_DAYS = 7
QUALITY_WINDOW_DAYS = 15
BIG_APPLIANCE_WINDOW_DAYS = 30
EXCHANGE_WINDOW_DAYS = 15

# 包邮商品走无理由退货时要扣回的发货运费。按订单扣一次，不是每件扣一次——
# 它是「一次发货」的成本，和退几件没关系。
FREE_SHIPPING_DEDUCTION_CENTS = 800

LATE_SHIP_GRACE_HOURS = 48
LATE_SHIP_RATE = 0.05
LATE_SHIP_MIN_CENTS = 500
LATE_SHIP_MAX_CENTS = 5000

EXCLUDED_KINDS = {
    "custom": "定制类商品",
    "fresh": "生鲜及易腐商品",
    "opened": "已拆封的贴身用品",
    "digital": "已激活的数码产品",
    "clearance": "标注「不退不换」的特价清仓商品",
}


def yuan(cents: int) -> str:
    return f"{cents / 100:.2f}"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    # 库里存的是带时区的 UTC，但手写的测试数据经常忘了带，这里统一补上，
    # 免得和 now 相减时抛 "can't subtract offset-naive and offset-aware"
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def item_goods_cents(item: dict) -> int:
    return int(item["qty"]) * int(item["unit_price_cents"])


def calendar_days_since(signed_at: str | None, now: datetime) -> int | None:
    """按日期差算，不按精确时长。

    政策写的是「自然日」：按 24 小时精确相减的话，「签收第 7 天下午」会被判超期。
    两边都取 UTC 日期，这样 seed 造出来的「签收后第 N 天」恒等于 N，
    测试结果也不会随运行时刻漂移。
    """
    moment = parse_time(signed_at)
    if moment is None:
        return None
    return (now.astimezone(timezone.utc).date() - moment.astimezone(timezone.utc).date()).days


def allocate_coupon(items: Sequence[dict], coupon_cents: int) -> dict[int, int]:
    """按商品金额比例分摊优惠券。

    最后一件吃掉四舍五入的余数，保证各件分摊之和**正好**等于券面金额。
    不这么做的话，退款明细加起来和订单实付对不上，对账时没法解释。
    """
    if not items:
        return {}
    total = sum(item_goods_cents(item) for item in items)
    if total <= 0 or coupon_cents <= 0:
        return {item["id"]: 0 for item in items}
    # 券面金额高于商品总额时按商品总额封顶，否则后面会算出负的实付金额
    coupon_cents = min(int(coupon_cents), total)
    shares: dict[int, int] = {}
    allocated = 0
    for item in items[:-1]:
        share = round(coupon_cents * item_goods_cents(item) / total)
        shares[item["id"]] = share
        allocated += share
    shares[items[-1]["id"]] = coupon_cents - allocated
    return shares


def item_paid_cents(item: dict, shares: dict[int, int]) -> int:
    """单件商品的实付金额 = 商品金额 − 它分摊到的优惠券。"""
    return item_goods_cents(item) - shares.get(item["id"], 0)


def _blocked(items: Iterable[dict]) -> list[dict]:
    return [
        {
            "sku": item["sku"],
            "name": item["name"],
            "why": EXCLUDED_KINDS.get(item["kind"], item["kind"]),
        }
        for item in items
        if item["kind"] != "normal"
    ]


def evaluate_return(
    order: dict,
    reason: str,
    now: datetime | None = None,
    items: Sequence[dict] | None = None,
) -> dict:
    """这批商品能不能退，依据哪一条，还剩几天。"""
    now = now or now_utc()
    if reason not in REASONS:
        raise ValueError(f"reason 只能是 {NO_REASON}（无理由）或 {QUALITY}（质量问题）")
    selected = list(order["items"] if items is None else items)
    days = calendar_days_since(order.get("signed_at"), now)

    base = {
        "reason": reason,
        "reason_label": REASONS[reason],
        "days_since_signed": days,
    }

    if order["status"] == "pending_ship":
        return {**base, "eligible": False, "window_days": 0, "days_left": 0, "expired": False,
                "blocked_items": [], "policy": "未发货订单的处理方式",
                "message": "订单还没发货。未发货的订单走「取消订单」，不走走退货流程。"}
    if days is None:
        return {**base, "eligible": False, "window_days": 0, "days_left": 0, "expired": False,
                "blocked_items": [], "policy": "签收后起算",
                "message": "这个订单还没有签收记录，签收之后才能申请退货。"}

    if reason == QUALITY:
        big = any(item.get("is_big_appliance") for item in selected)
        window = BIG_APPLIANCE_WINDOW_DAYS if big else QUALITY_WINDOW_DAYS
        blocked: list[dict] = []
        policy = (f"冰箱、洗衣机、空调等大家电存在质量问题的，自签收次日起 "
                  f"{BIG_APPLIANCE_WINDOW_DAYS} 个自然日内可申请退换货"
                  if big else
                  f"商品存在质量问题、与描述不符或发错货的，自签收次日起 "
                  f"{QUALITY_WINDOW_DAYS} 个自然日内可申请退货或换货")
    else:
        window = NO_REASON_WINDOW_DAYS
        blocked = _blocked(selected)
        policy = (f"7 天无理由退货：自签收次日起 {NO_REASON_WINDOW_DAYS} 个自然日内，"
                  f"商品完好、不影响二次销售")

    expired = days > window
    eligible = (not expired) and not blocked
    days_left = max(window - days, 0)

    if expired:
        message = (f"已经超过 {window} 个自然日的期限（现在是签收后第 {days} 天），"
                   f"平台不再受理退换货，只能走厂家保修渠道。")
    elif blocked:
        names = "、".join(f"{b['name']}（{b['why']}）" for b in blocked)
        message = (f"{names}不适用 7 天无理由退货。"
                   f"如果商品存在质量问题，可以按质量问题走 15 个自然日的窗口。")
    else:
        message = f"符合{REASONS[reason]}，期限还剩 {days_left} 天。"

    return {
        **base,
        "eligible": eligible,
        "window_days": window,
        "days_left": days_left,
        "expired": expired,
        "blocked_items": blocked,
        "policy": policy,
        "message": message,
    }


def refund_breakdown(
    order: dict,
    reason: str,
    now: datetime | None = None,
    skus: Sequence[str] | None = None,
) -> dict:
    """退款金额明细。"""
    now = now or now_utc()
    all_items = list(order["items"])
    missing: list[str] = []
    if skus:
        wanted = {s.strip().upper() for s in skus if s and s.strip()}
        selected = [i for i in all_items if i["sku"].upper() in wanted]
        missing = sorted(wanted - {i["sku"].upper() for i in selected})
    else:
        selected = all_items

    # 优惠券是整单施加的，分摊基数必须是整单商品金额，不能只用选中的那几件
    shares = allocate_coupon(all_items, int(order["coupon_cents"]))
    evaluation = evaluate_return(order, reason, now, selected)

    billable = [
        item for item in selected
        if not (reason == NO_REASON and item["kind"] != "normal")
    ]
    excluded = [item["name"] for item in selected if item not in billable]

    # 窗口外、还没签收、还没发货：只解释，不给金额。
    # 给一个数字等于暗示「能退这么多」，模型很可能直接照抄去回复用户。
    if evaluation["expired"] or evaluation["days_since_signed"] is None or order["status"] == "pending_ship":
        return {
            "eligible": False,
            "reason": reason,
            "reason_label": REASONS[reason],
            "refund_cents": 0,
            "refund_yuan": yuan(0),
            "goods_paid_cents": 0,
            "deduction_cents": 0,
            "lines": [],
            "notes": [evaluation["message"]],
            "excluded": excluded,
            "missing_skus": missing,
            "policy": evaluation["policy"],
            "evaluation": evaluation,
            "message": evaluation["message"],
        }

    lines: list[dict] = []
    goods_paid = 0
    for item in billable:
        paid = item_paid_cents(item, shares)
        goods_paid += paid
        lines.append({"label": f"{item['name']} × {item['qty']} 实付", "amount_cents": paid})

    deduction = 0
    if reason == NO_REASON and billable and int(order["shipping_cents"]) == 0:
        deduction = FREE_SHIPPING_DEDUCTION_CENTS
        lines.append({"label": "扣回发货运费（订单包邮）", "amount_cents": -deduction})
    refund_cents = max(goods_paid - deduction, 0)

    notes: list[str] = []
    if excluded:
        notes.append(f"以下商品不适用 7 天无理由退货，未计入退款：{'、'.join(excluded)}。")
    if reason == NO_REASON and int(order["shipping_cents"]) > 0:
        notes.append(f"该订单不是包邮，已支付的运费 {yuan(int(order['shipping_cents']))} 元不予退还，也不额外扣费。")
    if reason == QUALITY and int(order["shipping_cents"]) > 0:
        notes.append("质量问题退货的退货运费由平台承担（通过平台安排上门取件实现），退款金额里不再重复补贴运费。")
    if reason == NO_REASON and order["coupon_cents"]:
        notes.append("部分退货时已使用的优惠券不返还。")
    if missing:
        notes.append(f"订单里没有这些商品：{'、'.join(missing)}。")

    return {
        "eligible": bool(billable),
        "reason": reason,
        "reason_label": REASONS[reason],
        "refund_cents": refund_cents,
        "refund_yuan": yuan(refund_cents),
        "goods_paid_cents": goods_paid,
        "deduction_cents": deduction,
        "lines": lines,
        "notes": notes,
        "excluded": excluded,
        "missing_skus": missing,
        "policy": evaluation["policy"],
        "evaluation": evaluation,
        "message": evaluation["message"],
    }


def evaluate_exchange(
    order: dict,
    stock_by_sku: dict[str, dict | None],
    now: datetime | None = None,
) -> dict:
    """换货可行性：期限、商品类型、库存三个条件都要过。"""
    now = now or now_utc()
    days = calendar_days_since(order.get("signed_at"), now)

    if order["status"] == "pending_ship":
        return {"applicable": False, "days_since_signed": days, "days_left": 0,
                "items": [], "message": "订单还没发货，未发货的订单走「取消订单」。"}
    if days is None:
        return {"applicable": False, "days_since_signed": None, "days_left": 0,
                "items": [], "message": "这个订单还没有签收记录，签收之后才能申请换货。"}
    if days > EXCHANGE_WINDOW_DAYS:
        return {"applicable": False, "days_since_signed": days, "days_left": 0, "items": [],
                "message": (f"已经超过 {EXCHANGE_WINDOW_DAYS} 个自然日的换货期限"
                            f"（现在是签收后第 {days} 天），只能申请厂家保修维修。")}

    rows: list[dict] = []
    for item in order["items"]:
        stock = stock_by_sku.get(item["sku"])
        blocker = None
        if item["kind"] != "normal":
            blocker = f"{EXCLUDED_KINDS.get(item['kind'], item['kind'])}不支持换货"
        elif stock is None:
            blocker = "查不到这个商品的库存记录"
        elif not stock["supports_exchange"]:
            blocker = "这个商品不支持换货"
        elif int(stock["qty"]) <= 0:
            blocker = "这个商品当前没有库存，只能选择退货"
        rows.append({
            "sku": item["sku"],
            "name": item["name"],
            "qty": item["qty"],
            "stock_qty": int(stock["qty"]) if stock else None,
            "can_exchange": blocker is None,
            "blocker": blocker,
        })

    ok = [row for row in rows if row["can_exchange"]]
    if ok:
        names = "、".join(row["name"] for row in ok)
        message = f"{names}有库存，可以申请换货，换货运费由平台承担，期限还剩 {max(EXCHANGE_WINDOW_DAYS - days, 0)} 天。"
    else:
        message = "这个订单里的商品都不满足换货条件：" + "；".join(
            f"{row['name']}——{row['blocker']}" for row in rows
        )

    return {
        "applicable": bool(ok),
        "days_since_signed": days,
        "days_left": max(EXCHANGE_WINDOW_DAYS - days, 0),
        "items": rows,
        "message": message,
    }


def late_ship_compensation(order: dict, now: datetime | None = None) -> dict:
    """超时未发货赔付。"""
    now = now or now_utc()
    if order["status"] != "pending_ship":
        return {"applicable": False, "amount_cents": 0, "message": "订单已经发货，不涉及超时未发货赔付。"}
    created = parse_time(order["created_at"])
    if created is None:
        return {"applicable": False, "amount_cents": 0, "message": "订单缺少下单时间，判断不了是否超时。"}

    promised = int(order["promised_ship_hours"])
    overdue = (now - created).total_seconds() / 3600.0 - promised
    if overdue <= LATE_SHIP_GRACE_HOURS:
        return {
            "applicable": False,
            "amount_cents": 0,
            "overdue_hours": round(overdue, 1),
            "message": (f"承诺 {promised} 小时内发货，目前超时 {overdue:.0f} 小时，"
                        f"还没到 48 小时的赔付门槛。"),
        }

    paid = int(order["paid_cents"])
    raw = round(paid * LATE_SHIP_RATE)
    amount = min(max(raw, LATE_SHIP_MIN_CENTS), LATE_SHIP_MAX_CENTS)
    return {
        "applicable": True,
        "amount_cents": amount,
        "amount_yuan": yuan(amount),
        "overdue_hours": round(overdue, 1),
        "message": (f"已超过承诺发货时间 {overdue:.0f} 小时，按实付金额 {yuan(paid)} 元的 5% "
                    f"（最低 5 元、最高 50 元）计算，应赔付 {yuan(amount)} 元，"
                    f"以优惠券或账户余额形式发放。"),
    }
