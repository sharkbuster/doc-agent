"""给大模型用的工具。

每个工具做三件事：查数据、按规则判断、把结论说成一段人话。

两条约束：

1. **业务错误不抛异常**。订单号不存在、订单还没签收这类都是正常业务分支，
   抛出去模型只看到一句 ToolException，用户只看到 500。
   正确做法是把它当成正常返回值交回模型，让它自己组织语言。
2. **能算的绝不让模型算**。退款金额、剩余天数、是否超期全部由 policy 算完，
   工具只负责转述。让大模型做算术是这类系统最常见的错误来源。

返回值统一是 JSON 字符串：模型能读，前端也能拿去渲染工具轨迹。
"""
from __future__ import annotations

import json
import re
import sqlite3

from langchain_core.tools import tool

from .. import policy, runtime
from ..store import queries

_ORDER_ID_NOISE = re.compile(r"[\s\-_]+")

_REASON_ALIASES = {
    "no_reason": policy.NO_REASON,
    "无理由": policy.NO_REASON,
    "七天无理由": policy.NO_REASON,
    "7天无理由": policy.NO_REASON,
    "无理由退货": policy.NO_REASON,
    "quality": policy.QUALITY,
    "质量问题": policy.QUALITY,
    "质量": policy.QUALITY,
}

REASON_HINT = "no_reason 表示 7 天无理由退货，quality 表示质量问题退换货"


def normalize_order_id(raw: str) -> str:
    """模型经常把订单号写成小写、带空格或连字符，这里统一。"""
    return _ORDER_ID_NOISE.sub("", str(raw)).upper()


def normalize_reason(raw: str) -> str:
    key = str(raw).strip().lower().replace(" ", "")
    if key in _REASON_ALIASES:
        return _REASON_ALIASES[key]
    raise ValueError(f"无法识别的退货原因 {raw!r}，只能是：{REASON_HINT}")


def payload(ok: bool, summary: str, **data) -> str:
    return json.dumps({"ok": ok, "summary": summary, **data}, ensure_ascii=False, default=str)


def _fetch_order(order_id: str) -> dict | None:
    return queries.get_order(runtime.get_connection(), normalize_order_id(order_id))


def _missing(order_id: str) -> str:
    # 故意不列出库里还有哪些订单：这是客服机器人，把别人的订单号说给用户听是事故
    return payload(
        False,
        f"没有查到订单号 {normalize_order_id(order_id)}。请让用户核对「我的订单」里的订单号后重新提供。",
        order_id=normalize_order_id(order_id),
    )


STATUS_LABEL = {
    "pending_ship": "待发货",
    "shipped": "已发货",
    "signed": "已签收",
    "closed": "已完成",
}


def _order_headline(order: dict) -> str:
    parts = [
        f"订单 {order['order_id']}",
        STATUS_LABEL.get(order["status"], order["status"]),
        f"下单 {order['created_at'][:10]}",
        f"收货城市 {order['receiver_city']}",
    ]
    if order.get("signed_at"):
        parts.append(f"签收 {order['signed_at'][:10]}")
    return "，".join(parts)


def _order_items_text(order: dict) -> str:
    shown = [
        {
            "sku": item["sku"],
            "name": item["name"],
            "qty": item["qty"],
            "unit_price_yuan": policy.yuan(int(item["unit_price_cents"])),
            "kind": item["kind"],
        }
        for item in order["items"]
    ]
    return json.dumps(shown, ensure_ascii=False)


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单详情：状态、下单与签收时间、商品清单、各项金额。

    order_id：订单号，形如 SO20261001001。
    需要判断「签收多久了」「买了什么」「实付多少钱」时先调用这个工具。
    """
    try:
        order = _fetch_order(order_id)
    except sqlite3.Error as exc:
        return payload(False, f"查询订单时数据库出错：{exc}")
    if order is None:
        return _missing(order_id)

    goods = "、".join(f"{i['name']} × {i['qty']}" for i in order["items"])
    summary = (
        f"{_order_headline(order)}。商品：{goods}。"
        f"商品金额 {policy.yuan(int(order['goods_cents']))} 元，"
        f"运费 {policy.yuan(int(order['shipping_cents']))} 元，"
        f"优惠券抵扣 {policy.yuan(int(order['coupon_cents']))} 元，"
        f"实付 {policy.yuan(int(order['paid_cents']))} 元。"
    )
    return payload(
        True,
        summary,
        order_id=order["order_id"],
        status=order["status"],
        status_label=STATUS_LABEL.get(order["status"], order["status"]),
        created_at=order["created_at"],
        signed_at=order["signed_at"],
        receiver_city=order["receiver_city"],
        paid_cents=int(order["paid_cents"]),
        paid_yuan=policy.yuan(int(order["paid_cents"])),
        shipping_cents=int(order["shipping_cents"]),
        coupon_cents=int(order["coupon_cents"]),
        promised_ship_hours=int(order["promised_ship_hours"]),
        items=json.loads(_order_items_text(order)),
    )


@tool
def query_logistics(order_id: str) -> str:
    """按订单号查询物流：承运商、运单号、最新物流动态。

    order_id：订单号，形如 SO20261001001。
    用户问「货到哪了」「为什么还没发货」「物流是不是停了」时用这个。
    """
    try:
        order = _fetch_order(order_id)
        if order is None:
            return _missing(order_id)
        record = queries.get_logistics(runtime.get_connection(), order["order_id"])
    except sqlite3.Error as exc:
        return payload(False, f"查询物流时数据库出错：{exc}")

    if record is None:
        return payload(
            False,
            f"订单 {order['order_id']} 目前没有物流记录，说明还没有发货"
            f"（订单状态：{STATUS_LABEL.get(order['status'], order['status'])}）。",
            order_id=order["order_id"],
            status=order["status"],
        )
    summary = (
        f"订单 {order['order_id']} 的物流：{record['carrier']}，运单号 {record['tracking_no']}，"
        f"最新动态「{record['last_event']}」（{record['last_event_at'][:16]}）"
        f"{'，已签收' if record['delivered'] else '，尚未签收'}。"
    )
    return payload(
        True,
        summary,
        order_id=order["order_id"],
        carrier=record["carrier"],
        tracking_no=record["tracking_no"],
        last_event=record["last_event"],
        last_event_at=record["last_event_at"],
        delivered=bool(record["delivered"]),
    )


@tool
def check_return_eligibility(order_id: str, reason: str) -> str:
    """判断一个订单现在还能不能退货，依据的是哪一条政策，期限还剩几天。

    order_id：订单号，形如 SO20261001001。
    reason：no_reason 表示 7 天无理由退货，quality 表示质量问题退换货。
    这个工具会按政策算出结论，不要自己推算日期或期限。
    """
    try:
        reason_code = normalize_reason(reason)
    except ValueError as exc:
        return payload(False, str(exc))
    try:
        order = _fetch_order(order_id)
    except sqlite3.Error as exc:
        return payload(False, f"查询订单时数据库出错：{exc}")
    if order is None:
        return _missing(order_id)

    result = policy.evaluate_return(order, reason_code, runtime.now())
    return payload(
        True,
        result["message"],
        order_id=order["order_id"],
        reason=reason_code,
        reason_label=result["reason_label"],
        eligible=result["eligible"],
        expired=result["expired"],
        window_days=result["window_days"],
        days_since_signed=result["days_since_signed"],
        days_left=result["days_left"],
        policy=result["policy"],
        blocked_items=result["blocked_items"],
    )


@tool
def calc_refund(order_id: str, reason: str, skus: list[str] | None = None) -> str:
    """计算退货能退多少钱，给出逐项明细（商品实付、优惠券分摊、扣回的运费）。

    order_id：订单号，形如 SO20261001001。
    reason：no_reason 表示 7 天无理由退货，quality 表示质量问题退换货。
    skus：只想退其中某几件商品时传商品编码列表；整单退就留空。
    金额一律由这个工具算，不要自己加减。
    """
    try:
        reason_code = normalize_reason(reason)
    except ValueError as exc:
        return payload(False, str(exc))
    try:
        order = _fetch_order(order_id)
    except sqlite3.Error as exc:
        return payload(False, f"查询订单时数据库出错：{exc}")
    if order is None:
        return _missing(order_id)

    result = policy.refund_breakdown(order, reason_code, runtime.now(), skus=skus)
    lines = [
        {"label": line["label"], "amount_cents": line["amount_cents"],
         "amount_yuan": policy.yuan(line["amount_cents"])}
        for line in result["lines"]
    ]
    return payload(
        True,
        f"{result['message']} 应退金额 {result['refund_yuan']} 元。"
        + ("".join(f" {note}" for note in result["notes"])),
        order_id=order["order_id"],
        reason=reason_code,
        reason_label=result["reason_label"],
        eligible=result["eligible"],
        refund_cents=result["refund_cents"],
        refund_yuan=result["refund_yuan"],
        goods_paid_cents=result["goods_paid_cents"],
        deduction_cents=result["deduction_cents"],
        deduction_yuan=policy.yuan(result["deduction_cents"]),
        lines=lines,
        notes=result["notes"],
        excluded=result["excluded"],
        missing_skus=result["missing_skus"],
        policy=result["policy"],
    )


@tool
def check_exchange(order_id: str) -> str:
    """判断一个订单能不能换货：期限是否还在、商品类型是否允许、有没有库存。

    order_id：订单号，形如 SO20261001001。
    用户想「换一个」「换货」「换同款」时用这个；只是想退钱用 calc_refund。
    """
    try:
        order = _fetch_order(order_id)
    except sqlite3.Error as exc:
        return payload(False, f"查询订单时数据库出错：{exc}")
    if order is None:
        return _missing(order_id)

    conn = runtime.get_connection()
    stock = {item["sku"]: queries.get_stock(conn, item["sku"]) for item in order["items"]}
    result = policy.evaluate_exchange(order, stock, runtime.now())
    return payload(
        True,
        result["message"],
        order_id=order["order_id"],
        applicable=result["applicable"],
        days_since_signed=result["days_since_signed"],
        days_left=result["days_left"],
        items=result["items"],
    )


@tool
def calc_late_ship_compensation(order_id: str) -> str:
    """计算超时未发货的赔付金额。

    order_id：订单号，形如 SO20261001001。
    用户抱怨「买了很久还没发货」「说好 48 小时发货」时用这个。
    """
    try:
        order = _fetch_order(order_id)
    except sqlite3.Error as exc:
        return payload(False, f"查询订单时数据库出错：{exc}")
    if order is None:
        return _missing(order_id)

    result = policy.late_ship_compensation(order, runtime.now())
    return payload(
        True,
        result["message"],
        order_id=order["order_id"],
        applicable=result["applicable"],
        amount_cents=result.get("amount_cents", 0),
        amount_yuan=result.get("amount_yuan", "0.00"),
        overdue_hours=result.get("overdue_hours"),
    )


TOOLS = [
    query_order,
    query_logistics,
    check_return_eligibility,
    calc_refund,
    check_exchange,
    calc_late_ship_compensation,
]

TOOL_MAP = {item.name: item for item in TOOLS}
