# strategy_v1.py
import pandas as pd

# —— 第一策略：两档买入（-0.7%/-1.4%/突破前高），止损8%，止盈10%/20%各50%，余仓92%追踪；
# —— 任意卖出后生成“次日生效”OCO接回（卖价*95%、卖价*92.5%，成交一档取消另一档，全仓买）

BUY_LIM_A = 0.993   # -0.7%
BUY_LIM_B = 0.986   # -1.4%
STOP_LOSS_PCT = 0.08
TP1_PCT = 0.10      # +10% 卖 50%
TP2_PCT = 0.20      # +20% 卖 50%
TRAIL_PCT = 0.08    # 回撤 8%（= 92% 追踪）
REBUY_A = 0.95      # OCO：卖价*95%
REBUY_B = 0.925     # OCO：卖价*92.5%

def backtest(
    bars: pd.DataFrame,
    initial_cash: float = 100_000.0,
    *,
    lot_size: int = 100,
    commission_rate: float = 0.0003,
    commission_min: float = 5.0,
    stamp_duty_sell: float = 0.0005,
    slippage: float = 0.0,
):
    """
    bars: DataFrame 必含列 ['date','open','high','low','close']，按日期升序
    返回: (curve_df, trades_df, label)
    """
    def _lot_floor(q): return int(q // lot_size) * lot_size
    def _commission(amount): return max(commission_min, amount * commission_rate)
    def _stamp(amount): return amount * stamp_duty_sell
    def _hit_between(lo, hi, px): return (lo <= px) and (px <= hi)

    cash = initial_cash
    shares = 0
    position_cost = 0.0
    highest_since_entry = None

    oco_next = None   # 次日激活
    oco_active = None # 已激活 OCO

    curve, trades = [], []
    prev_close, prev_high = None, None

    for _, r in bars.iterrows():
        d = pd.to_datetime(r["date"])
        o = float(r["open"]); h = float(r["high"]); l = float(r["low"]); c = float(r["close"])

        # 次日生效 → 激活
        if oco_next is not None:
            oco_active = oco_next
            oco_next = None

        # 记录净值
        curve.append(dict(date=d, equity=cash + shares * c, cash=cash, shares=shares, close=c))

        if prev_close is None or prev_high is None:
            prev_close, prev_high = c, h
            continue

        # —— 1) 风控（止损/追踪）——
        if shares > 0:
            highest_since_entry = max(highest_since_entry or o, h)
            stop_loss_px = position_cost * (1 - STOP_LOSS_PCT)
            trail_px     = (highest_since_entry) * (1 - TRAIL_PCT)

            triggered = []
            if o <= stop_loss_px: triggered.append(("STOP_LOSS", max(stop_loss_px, o)))
            if o <= trail_px:     triggered.append(("TRAIL_STOP",  max(trail_px, o)))
            if triggered:
                reason, px = max(triggered, key=lambda x: x[1])
                price = px - slippage
                qty = shares
                amt = price * qty
                fee = _commission(amt); tax = _stamp(amt)
                cash += amt - fee - tax; shares = 0
                trades.append(dict(date=d, side="SELL", price=price, qty=qty, fee=fee, tax=tax, reason=reason))
                position_cost = 0.0; highest_since_entry = None
                oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}
            else:
                for reason, px in sorted([("STOP_LOSS", stop_loss_px), ("TRAIL_STOP", trail_px)], key=lambda x: x[1], reverse=True):
                    if shares > 0 and _hit_between(l, h, px):
                        price = px - slippage
                        qty = shares
                        amt = price * qty
                        fee = _commission(amt); tax = _stamp(amt)
                        cash += amt - fee - tax; shares = 0
                        trades.append(dict(date=d, side="SELL", price=price, qty=qty, fee=fee, tax=tax, reason=reason))
                        position_cost = 0.0; highest_since_entry = None
                        oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}
                        break

        # —— 2) 分批止盈 10%/20%（各 50%）——
        if shares > 0:
            tp1 = position_cost * (1 + TP1_PCT)
            tp2 = position_cost * (1 + TP2_PCT)

            def half_qty(): return _lot_floor(max(0, shares // 2))

            if shares > 0 and o >= tp2:
                q = half_qty()
                if q > 0:
                    price = tp2 - slippage
                    amt = price * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=price, qty=q, fee=fee, tax=tax, reason="TP20"))
                    oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}
                if shares > 0 and o >= tp1:
                    q = half_qty()
                    if q > 0:
                        price = tp1 - slippage
                        amt = price * q
                        fee = _commission(amt); tax = _stamp(amt)
                        cash += amt - fee - tax; shares -= q
                        trades.append(dict(date=d, side="SELL", price=price, qty=q, fee=fee, tax=tax, reason="TP10"))
                        oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}
            elif shares > 0 and o >= tp1:
                q = half_qty()
                if q > 0:
                    price = tp1 - slippage
                    amt = price * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=price, qty=q, fee=fee, tax=tax, reason="TP10"))
                    oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}

            if shares > 0 and h >= tp2:
                q = half_qty()
                if q > 0:
                    price = tp2 - slippage
                    amt = price * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=price, qty=q, fee=fee, tax=tax, reason="TP20"))
                    oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}
            if shares > 0 and h >= tp1:
                q = half_qty()
                if q > 0:
                    price = tp1 - slippage
                    amt = price * q
                    fee = _commission(amt); tax = _stamp(amt)
                    cash += amt - fee - tax; shares -= q
                    trades.append(dict(date=d, side="SELL", price=price, qty=q, fee=fee, tax=tax, reason="TP10"))
                    oco_next = {"px1": price * REBUY_A, "px2": price * REBUY_B}

        # —— 3) 入场：先 OCO 接回，再三触发（空仓）——
        if shares == 0:
            filled = False
            if oco_active is not None:
                for px in sorted([oco_active["px1"], oco_active["px2"]]):
                    if (o <= px) or _hit_between(l, h, px):
                        price = (px + slippage)
                        qty = _lot_floor(cash / price)
                        if qty > 0:
                            amt = price * qty
                            fee = _commission(amt)
                            total = amt + fee
                            if total <= cash:
                                cash -= total; shares += qty
                                position_cost = price; highest_since_entry = price
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
                if _hit_between(l, h, buy_b): cands.append(("BUY_B", buy_b))
                if _hit_between(l, h, buy_a): cands.append(("BUY_A", buy_a))
                if h >= breakout_px:
                    px = max(breakout_px, o)
                    if px <= h:
                        cands.append(("BREAKOUT", px))
                if cands:
                    reason, px = min(cands, key=lambda x: x[1])
                    price = (px + slippage)
                    qty = _lot_floor(cash / price)
                    if qty > 0:
                        amt = price * qty
                        fee = _commission(amt)
                        total = amt + fee
                        if total <= cash:
                            cash -= total; shares += qty
                            position_cost = price; highest_since_entry = price
                            trades.append(dict(date=d, side="BUY", price=price, qty=qty, fee=fee, reason=reason))

        prev_close, prev_high = c, h

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    label = "StrategyV1(-0.7/-1.4/Breakout; TP10/20 50%; 92%Trail; OCO Rebuy)"
    return curve_df, trades_df, label
