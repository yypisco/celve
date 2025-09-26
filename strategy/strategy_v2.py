# strategy_v2.py
import math
import pandas as pd
import numpy as np

LOT_SIZE = 100
COMMISSION_RATE = 0.0003
COMMISSION_MIN  = 5.0
STAMP_DUTY_SELL = 0.0005
SLIPPAGE = 0.0

BUY_LIM_A = 0.993   # -0.7%
BUY_LIM_B = 0.986   # -1.4%
STOP_LOSS_PCT = 0.08
TP1_PCT = 0.10      # +10% 卖 1/3
TP2_PCT = 0.20      # +20% 卖 1/3
TRAIL_PCT = 0.08    # 余仓追踪 8% 清仓

REBUY_TP_A = 0.95   # 止盈后接回：卖价*95%（数量=卖出量/2）
REBUY_TP_B = 0.925  # 止盈后接回：卖价*92.5%（数量=卖出量/2）
REBUY_SL_A = 0.993  # 止损后接回：前收*99.3%（现金一半）
REBUY_SL_B = 0.986  # 止损后接回：前收*98.6%（现金一半）

def _lot_floor(q):
    return (q // 100) * 100

def _commission(amount):
    return max(COMMISSION_MIN, amount * COMMISSION_RATE)

def _stamp(amount):
    return amount * STAMP_DUTY_SELL

def backtest(bars: pd.DataFrame, initial_cash: float = 100_000.0):
    """
    第二策略实现：止盈1/3@10%、1/3@20%、余仓追踪8%；止盈后接回两档(各=卖出量一半，长期有效)；
    止损后次日按前收-0.7%、-1.4%各挂半仓资金买入，长期有效；三触发入场同V1。
    返回 curve, trades, label
    """
    cash = initial_cash
    shares = 0
    position_cost = 0.0
    highest_since_entry = None

    # 次日生效清单
    tp_rebuys_next = []  # [{'px':..., 'qty':...}]
    sl_rebuy_flag_next = False  # 次日根据前收生成

    # 已激活的长期有效接回
    tp_rebuys_active = []  # 固定手数
    sl_rebuys_active = []  # 现金一半（下单时计算）

    curve, trades = [], []
    prev_close, prev_high = None, None

    for i, r in bars.iterrows():
        d, o, h, l, c = r["date"], r["open"], r["high"], r["low"], r["close"]

        # 开盘前：激活前日生成的接回
        if tp_rebuys_next:
            tp_rebuys_active.extend(tp_rebuys_next)
            tp_rebuys_next = []
        if sl_rebuy_flag_next and prev_close is not None:
            sl_rebuys_active.extend([
                dict(px=prev_close * REBUY_SL_A),
                dict(px=prev_close * REBUY_SL_B),
            ])
            sl_rebuy_flag_next = False

        # 记录净值
        curve.append(dict(date=d, equity=cash + shares * c, cash=cash, shares=shares, close=c))

        if prev_close is None or prev_high is None:
            prev_close, prev_high = c, h
            continue

        # —— 1) 风控：止损/追踪 —— 
        if shares > 0:
            highest_since_entry = max(highest_since_entry or o, h)
            stop_loss_px = position_cost * (1 - STOP_LOSS_PCT)
            trail_px     = (highest_since_entry) * (1 - TRAIL_PCT)

            triggered = []
            if o <= stop_loss_px: triggered.append(("STOP_LOSS", max(stop_loss_px, o)))
            if o <= trail_px:     triggered.append(("TRAIL_STOP",  max(trail_px, o)))
            if triggered:
                reason, px = max(triggered, key=lambda x: x[1])
                qty = shares
                amt = px * qty
                fee = _commission(amt); tax = _stamp(amt)
                cash += amt - fee - tax
                shares = 0
                trades.append(dict(date=d, side="SELL", price=px, qty=qty, fee=fee, tax=tax, reason=reason))
                position_cost = 0.0
                highest_since_entry = None
                sl_rebuy_flag_next = True
            else:
                for reason, px in sorted([("STOP_LOSS", stop_loss_px), ("TRAIL_STOP", trail_px)], key=lambda x: x[1], reverse=True):
                    if shares > 0 and l <= px <= h:
                        qty = shares
                        amt = px * qty
                        fee = _commission(amt); tax = _stamp(amt)
                        cash += amt - fee - tax
                        shares = 0
                        trades.append(dict(date=d, side="SELL", price=px, qty=qty, fee=fee, tax=tax, reason=reason))
                        position_cost = 0.0
                        highest_since_entry = None
                        sl_rebuy_flag_next = True
                        break

        # —— 2) 止盈：1/3 @ +10%、1/3 @ +20%，每次止盈后生成次日 TP 接回两档（各=卖出量一半，长期有效）——
        if shares > 0:
            tp1 = position_cost * (1 + TP1_PCT)
            tp2 = position_cost * (1 + TP2_PCT)

            def third_qty():
                return _lot_floor(max(0, shares // 3))

            # 开盘越过
            if shares > 0 and o >= tp2:
                q = third_qty()
                if q > 0:
                    amt = tp2*q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp2, qty=q, fee=fee, tax=tax, reason="TP20"))
                    half = _lot_floor(q//2)
                    if half > 0:
                        tp_rebuys_next += [dict(px=tp2*REBUY_TP_A, qty=half), dict(px=tp2*REBUY_TP_B, qty=half)]
            if shares > 0 and o >= tp1:
                q = third_qty()
                if q > 0:
                    amt = tp1*q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp1, qty=q, fee=fee, tax=tax, reason="TP10"))
                    half = _lot_floor(q//2)
                    if half > 0:
                        tp_rebuys_next += [dict(px=tp1*REBUY_TP_A, qty=half), dict(px=tp1*REBUY_TP_B, qty=half)]

            # 盘中越过
            if shares > 0 and h >= tp2:
                q = third_qty()
                if q > 0:
                    amt = tp2*q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp2, qty=q, fee=fee, tax=tax, reason="TP20"))
                    half = _lot_floor(q//2)
                    if half > 0:
                        tp_rebuys_next += [dict(px=tp2*REBUY_TP_A, qty=half), dict(px=tp2*REBUY_TP_B, qty=half)]
            if shares > 0 and h >= tp1:
                q = third_qty()
                if q > 0:
                    amt = tp1*q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp1, qty=q, fee=fee, tax=tax, reason="TP10"))
                    half = _lot_floor(q//2)
                    if half > 0:
                        tp_rebuys_next += [dict(px=tp1*REBUY_TP_A, qty=half), dict(px=tp1*REBUY_TP_B, qty=half)]

        # —— 3) 入场（仅空仓）：先尝试已激活接回，再常规三触发 —— 
        if shares == 0:
            filled_today = False

            # 3.1 TP 固定手数接回（低价优先，**非 OCO，长期保留**；但一天同价多单只成交一次）
            if tp_rebuys_active:
                for od in sorted(tp_rebuys_active, key=lambda x: x["px"]):
                    px = od["px"]
                    if (o <= px) or (l <= px <= h):
                        price = px + SLIPPAGE
                        q = od["qty"]
                        q = _lot_floor(q)
                        if q > 0:
                            amt = price*q
                            fee = _commission(amt)
                            total = amt + fee
                            if total <= cash:
                                cash -= total
                                shares += q
                                position_cost = price
                                highest_since_entry = price
                                trades.append(dict(date=d, side="BUY", price=price, qty=q, fee=fee, reason="TP_REBUY"))
                                od["__filled__"] = True
                                filled_today = True
                                break
                tp_rebuys_active = [x for x in tp_rebuys_active if not x.get("__filled__", False)]

            # 3.2 SL 半仓接回（低价优先；每单用“当前现金的一半”预算，成交后移除）
            if not filled_today and sl_rebuys_active:
                for od in sorted(sl_rebuys_active, key=lambda x: x["px"]):
                    px = od["px"]
                    if (o <= px) or (l <= px <= h):
                        price = px + SLIPPAGE
                        budget = cash * 0.5
                        q = _lot_floor(budget / price)
                        if q > 0:
                            amt = price*q
                            fee = _commission(amt)
                            total = amt + fee
                            if total <= cash:
                                cash -= total
                                shares += q
                                position_cost = price
                                highest_since_entry = price
                                trades.append(dict(date=d, side="BUY", price=price, qty=q, fee=fee, reason="SL_REBUY"))
                                od["__filled__"] = True
                                filled_today = True
                                break
                sl_rebuys_active = [x for x in sl_rebuys_active if not x.get("__filled__", False)]

            # 3.3 常规三触发
            if not filled_today:
                buy_b = prev_close * BUY_LIM_B
                buy_a = prev_close * BUY_LIM_A
                breakout_px = prev_high
                cands = []
                if l <= buy_b <= h: cands.append(("BUY_B", buy_b))
                if l <= buy_a <= h: cands.append(("BUY_A", buy_a))
                if h >= breakout_px:
                    px = max(breakout_px, o)
                    if px <= h:
                        cands.append(("BREAKOUT", px))
                if cands:
                    reason, px = min(cands, key=lambda x: x[1])
                    price = px + SLIPPAGE
                    q = _lot_floor(cash / price)
                    if q > 0:
                        amt = price*q
                        fee = _commission(amt)
                        total = amt + fee
                        if total <= cash:
                            cash -= total
                            shares += q
                            position_cost = price
                            highest_since_entry = price
                            trades.append(dict(date=d, side="BUY", price=price, qty=q, fee=fee, reason=reason))

        prev_close, prev_high = c, h

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    label = "StrategyV2(1/3+1/3止盈/余仓追踪/止盈与止损接回)"
    return curve_df, trades_df, label
