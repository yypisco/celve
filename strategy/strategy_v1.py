# strategy_v1.py
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
TP1_PCT = 0.10      # +10% 卖 50%
TP2_PCT = 0.20      # +20% 卖 50%
TRAIL_PCT = 0.08    # 追踪止盈回撤 8% 清仓
REBUY_A = 0.95      # 卖出后次日 OCO：卖价*95%
REBUY_B = 0.925     # 卖出后次日 OCO：卖价*92.5%

def _lot_floor(q):
    return (q // LOT_SIZE) * LOT_SIZE

def _commission(amount):
    return max(COMMISSION_MIN, amount * COMMISSION_RATE)

def _stamp(amount):
    return amount * STAMP_DUTY_SELL

def backtest(bars: pd.DataFrame, initial_cash: float = 100_000.0):
    """
    bars: DataFrame[date, open, high, low, close] 升序
    返回: curve(DataFrame: date,equity,cash,shares,close), trades(DataFrame), label(str)
    """
    cash = initial_cash
    shares = 0
    position_cost = 0.0
    highest_since_entry = None

    # OCO 接回（次日生效），结构 None or {'px1':..., 'px2':...}
    oco_next = None
    oco_active = None

    curve, trades = [], []
    prev_close, prev_high = None, None

    for i, r in bars.iterrows():
        d, o, h, l, c = r["date"], r["open"], r["high"], r["low"], r["close"]

        # 激活次日接回
        if oco_next is not None:
            oco_active = oco_next
            oco_next = None

        # 记录净值（收盘估值）
        curve.append(dict(date=d, equity=cash + shares * c, cash=cash, shares=shares, close=c))

        # 首日无法计算前收/前高
        if prev_close is None or prev_high is None:
            prev_close, prev_high = c, h
            continue

        # —— 1) 开盘风控（止损/追踪）——
        if shares > 0:
            highest_since_entry = max(highest_since_entry or o, h)
            stop_loss_px = position_cost * (1 - STOP_LOSS_PCT)
            trail_px = (highest_since_entry) * (1 - TRAIL_PCT)

            triggered = []
            if o <= stop_loss_px: triggered.append(("STOP_LOSS", max(stop_loss_px, o)))
            if o <= trail_px:     triggered.append(("TRAIL_STOP",  max(trail_px, o)))
            if triggered:
                reason, px = max(triggered, key=lambda x: x[1])  # 更高价优先
                qty = shares
                # 卖出
                amt = px * qty
                fee = _commission(amt)
                tax = _stamp(amt)
                cash += amt - fee - tax
                shares = 0
                trades.append(dict(date=d, side="SELL", price=px, qty=qty, fee=fee, tax=tax, reason=reason))
                position_cost = 0.0
                highest_since_entry = None
                # 生成 OCO 次日接回
                oco_next = {"px1": px * REBUY_A, "px2": px * REBUY_B}
            else:
                # 盘中风控（价位更高者优先）
                for reason, px in sorted([("STOP_LOSS", stop_loss_px), ("TRAIL_STOP", trail_px)], key=lambda x: x[1], reverse=True):
                    if shares > 0 and l <= px <= h:
                        qty = shares
                        amt = px * qty
                        fee = _commission(amt)
                        tax = _stamp(amt)
                        cash += amt - fee - tax
                        shares = 0
                        trades.append(dict(date=d, side="SELL", price=px, qty=qty, fee=fee, tax=tax, reason=reason))
                        position_cost = 0.0
                        highest_since_entry = None
                        oco_next = {"px1": px * REBUY_A, "px2": px * REBUY_B}
                        break

        # —— 2) 分批止盈 10%/20%（各 50%）——
        if shares > 0:
            tp1 = position_cost * (1 + TP1_PCT)
            tp2 = position_cost * (1 + TP2_PCT)

            def half_qty():
                return _lot_floor(max(0, shares // 2))

            # 开盘越过
            if shares > 0 and o >= tp2:
                q = half_qty()
                if q > 0:
                    amt = tp2 * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp2, qty=q, fee=fee, tax=tax, reason="TP20"))
                    oco_next = {"px1": tp2*REBUY_A, "px2": tp2*REBUY_B}
                if shares > 0 and o >= tp1:
                    q = half_qty()
                    if q > 0:
                        amt = tp1 * q
                        fee = _commission(amt); tax = _stamp(amt)
                        cash += amt - fee - tax; shares -= q
                        trades.append(dict(date=d, side="SELL", price=tp1, qty=q, fee=fee, tax=tax, reason="TP10"))
                        oco_next = {"px1": tp1*REBUY_A, "px2": tp1*REBUY_B}
            elif shares > 0 and o >= tp1:
                q = half_qty()
                if q > 0:
                    amt = tp1 * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp1, qty=q, fee=fee, tax=tax, reason="TP10"))
                    oco_next = {"px1": tp1*REBUY_A, "px2": tp1*REBUY_B}

            # 盘中越过
            if shares > 0 and h >= tp2:
                q = half_qty()
                if q > 0:
                    amt = tp2 * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp2, qty=q, fee=fee, tax=tax, reason="TP20"))
                    oco_next = {"px1": tp2*REBUY_A, "px2": tp2*REBUY_B}
            if shares > 0 and h >= tp1:
                q = half_qty()
                if q > 0:
                    amt = tp1 * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=tp1, qty=q, fee=fee, tax=tax, reason="TP10"))
                    oco_next = {"px1": tp1*REBUY_A, "px2": tp1*REBUY_B}

        # —— 3) 入场：先试 OCO 接回，再三触发 ——（仅空仓）
        if shares == 0:
            filled = False
            if oco_active is not None:
                for px in sorted([oco_active["px1"], oco_active["px2"]]):
                    if (o <= px) or (l <= px <= h):
                        price = px + SLIPPAGE
                        qty = _lot_floor(cash / price)
                        if qty > 0:
                            amt = price * qty
                            fee = _commission(amt)
                            total = amt + fee
                            if total <= cash:
                                cash -= total
                                shares += qty
                                position_cost = price
                                highest_since_entry = price
                                trades.append(dict(date=d, side="BUY", price=price, qty=qty, fee=fee, reason="REBUY"))
                                filled = True
                                break
                if filled:
                    oco_active = None  # OCO：成交一档取消另一档

            if not filled:
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
                    reason, px = min(cands, key=lambda x: x[1])  # 更便宜优先
                    price = px + SLIPPAGE
                    qty = _lot_floor(cash / price)
                    if qty > 0:
                        amt = price * qty
                        fee = _commission(amt)
                        total = amt + fee
                        if total <= cash:
                            cash -= total
                            shares += qty
                            position_cost = price
                            highest_since_entry = price
                            trades.append(dict(date=d, side="BUY", price=price, qty=qty, fee=fee, reason=reason))

        prev_close, prev_high = c, h

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    label = "StrategyV1(两档买入/分批止盈/92%追踪/OCO接回)"
    return curve_df, trades_df, label
