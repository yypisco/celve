
# strategy_v12.py
# V12-Fusion: Merge V1 aggressiveness and V2 restraint with risk-budgeted tranches,
#             breakout + pullback + retest entries, profit split + trailing exits,
#             and strategy-level drawdown gates.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class Params:
    ema_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200
    atr_n: int = 14
    breakout_lookback: int = 20  # High(N) breakout window
    breakout_buffer: float = 0.001  # +0.1%
    retest_tol: float = 0.002  # ±0.2%
    retest_window_days: int = 10  # after breakout

    # pullback depths (percent or ATR-based deeper of the two)
    pb1_min_pct: float = 0.007   # 0.7%
    pb1_atr_mult: float = 0.5
    pb2_min_pct: float = 0.014   # 1.4%
    pb2_atr_mult: float = 1.0

    # stop / trailing
    min_stop_pct: float = 0.035   # 3.5%
    stop_atr_mult: float = 3.0    # becomes 2.5× when DD>=8%
    chandelier_atr_mult: float = 3.0

    # profit split thresholds
    tp1: float = 0.10
    tp2: float = 0.20

    # tranche risk-budget (as % of equity) by drawdown regime
    tr_risk_normal: float = 0.0030
    tr_risk_warn: float = 0.0020
    tr_risk_limit: float = 0.0010

    # volatility contraction threshold
    vol_contract_ratio: float = 0.04  # atr/price > 4% halves tranche size

    # drawdown gates
    dd_warn: float = 0.06   # >=6%: disable new breakouts, keep PB & reentry2
    dd_freeze: float = 0.08 # >=8%: freeze all new buys; tighten trailing to 2.5×ATR
    dd_stop: float = 0.10   # >=10%: liquidate all, cool-down N days
    cooldown_days: int = 10

    # execution & rounding
    lot_size: int = 1


class V12State:
    def __init__(self):
        self.equity_peak: float = 0.0
        self.cooldown_left: int = 0
        self.profit_stage: int = 0  # 0,1,2
        self.breakout_ref_price: Optional[float] = None
        self.retest_deadline: Optional[pd.Timestamp] = None
        # pending orders: dicts {type, price, qty_frac, note}
        self.pending: List[Dict] = []


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    # df columns: open, high, low, close (index = date)
    high = df['high']; low = df['low']; close = df['close']
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def backtest(
    bars: pd.DataFrame,
    initial_cash: float,
    lot_size: int = 1,
    commission_rate: float = 0.0,
    commission_min: float = 0.0,
    stamp_duty_sell: float = 0.0,
    slippage: float = 0.0,
):
    """
    Parameters
    ----------
    bars : DataFrame with ['date','open','high','low','close'] ascending by date
    returns
    -------
    curve_df(date,equity), trades_df(date,side,price,qty,fee,reason), label(str)
    """
    P = Params(lot_size=lot_size)
    df = bars[['date','open','high','low','close']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open','high','low','close']).sort_values('date').reset_index(drop=True)
    df = df[df['close'] > 0].reset_index(drop=True)

    if len(df) < 30:
        # too short, just flat
        curve = pd.DataFrame({'date': df['date'], 'equity': initial_cash})
        trades = pd.DataFrame(columns=['date','side','price','qty','fee','reason'])
        return curve, trades, "StrategyV12-Fusion"

    df.set_index('date', inplace=True)
    price = df['close']
    opens = df['open']

    ema20 = _ema(price, P.ema_fast)
    sma50 = _sma(price, P.sma_mid)
    sma200 = _sma(price, P.sma_slow)
    atr = _atr(df, P.atr_n)
    highN = df['high'].rolling(P.breakout_lookback, min_periods=1).max()

    # trend filter
    trend_up = (price > sma50) & (sma50 > sma200)

    # sim state
    st = V12State()
    cash = float(initial_cash)
    shares = 0
    avg_cost = 0.0  # weighted average entry cost of current position

    equity_curve: List[Dict] = []
    trades: List[Dict] = []

    # helpers
    def equity_val(dt):
        return cash + shares * price.loc[dt]

    def drawdown(dt):
        nonlocal st
        eq = equity_val(dt)
        st.equity_peak = max(st.equity_peak, eq)
        if st.equity_peak <= 0:
            return 0.0
        return max(0.0, 1.0 - eq / st.equity_peak)

    def tranche_risk(dd):
        if dd >= P.dd_freeze:
            return P.tr_risk_limit
        elif dd >= P.dd_warn:
            return P.tr_risk_warn
        else:
            return P.tr_risk_normal

    def current_stop_mult(dd):
        return 2.5 if dd >= P.dd_freeze else P.stop_atr_mult

    def position_value_for_tranche(dt, entry_price):
        dd = drawdown(dt)
        r_tr = tranche_risk(dd)
        ap = price.loc[dt]
        a = atr.loc[dt]
        stop_dist = max(P.min_stop_pct, 1.2 * (a / ap))
        # risk per share ≈ entry_price * stop_dist
        risk_per_share = entry_price * stop_dist
        if risk_per_share <= 0:
            return 0.0
        val = (r_tr * equity_val(dt)) / (risk_per_share / entry_price)
        # volatility contraction
        if (a / ap) > P.vol_contract_ratio:
            val *= 0.5
        return max(0.0, val)

    def round_qty(value, px):
        q = int((value / px) // P.lot_size * P.lot_size)
        return max(0, q)

    def fees(side, gross):
        fee_comm = max(gross * commission_rate, commission_min) if commission_rate > 0 else 0.0
        fee_tax = gross * stamp_duty_sell if (side == "SELL" and stamp_duty_sell > 0) else 0.0
        return float(fee_comm + fee_tax)

    def place_trade(dt, side, px, qty, reason):
        nonlocal cash, shares, avg_cost
        if qty <= 0:
            return
        trade_price = float(px * (1.0 + slippage if side == "BUY" else 1.0 - slippage))
        gross = trade_price * qty
        fee = fees(side, gross)
        if side == "BUY":
            total = gross + fee
            if total > cash:
                # downsize to available cash
                qty2 = int(((cash - fee) / trade_price) // P.lot_size * P.lot_size)
                if qty2 <= 0:
                    return
                gross = trade_price * qty2
                fee = fees(side, gross)
                total = gross + fee
                qty = qty2
            cash -= total
            new_cost = (avg_cost * shares + trade_price * qty) / (shares + qty) if (shares + qty) > 0 else 0.0
            shares += qty
            avg_cost = new_cost
        else:
            qty = min(qty, shares)
            if qty <= 0:
                return
            proceed = gross - fee
            cash += proceed
            # update avg_cost for remaining shares
            if shares - qty > 0:
                # keep avg_cost unchanged for remaining
                pass
            else:
                avg_cost = 0.0
                st.profit_stage = 0
            shares -= qty
        trades.append(dict(date=dt, side=side, price=trade_price, qty=int(qty), fee=float(fee), reason=reason))

    # track stop target per day (re-evaluated with chandelier)
    hard_stop_price: Optional[float] = None

    # pending orders are re-built each day based on rules (GTC semantics through re-creation)
    for i, dt in enumerate(df.index):
        # record equity at CLOSE of previous day (for peak/DD)
        _ = drawdown(dt)

        # 1) SELL side: evaluate stops/profit using yesterday's close -> execute TODAY open
        if i > 0:
            prev_dt = df.index[i - 1]
            prev_close = price.loc[prev_dt]
            a = atr.loc[prev_dt]

            # update hard stop initial when we newly entered
            # (handled at entry time)

            # trailing stop (Chandelier)
            if shares > 0:
                # peak since entry using closes
                # For simplicity, use rolling max since last time shares changed; approximated by cummax
                # We'll approximate using cumulative max while in position
                # Maintain via variable? Simpler: use running max of close when in position.
                # For approximation, use max price up to prev_dt.
                peak = price.loc[:prev_dt].max()
                trail_mult = current_stop_mult(drawdown(prev_dt))
                trail_stop = peak - trail_mult * a
                cur_stop = max(hard_stop_price or -np.inf, trail_stop) if hard_stop_price is not None else trail_stop

                if prev_close <= cur_stop:
                    # sell ALL at today's open
                    open_px = opens.loc[dt]
                    place_trade(dt, "SELL", open_px, shares, reason="Stop/Trail Exit")
                    hard_stop_price = None  # reset after exit

            # profit splits
            if shares > 0 and avg_cost > 0:
                gain = prev_close / avg_cost - 1.0
                if st.profit_stage == 0 and gain >= P.tp1:
                    # sell 1/3 at today's open
                    open_px = opens.loc[dt]
                    qty = max(P.lot_size, int(shares / 3 // P.lot_size * P.lot_size))
                    place_trade(dt, "SELL", open_px, qty, reason="TP +10% (1/3)")
                    st.profit_stage = 1
                    # schedule re-entries (RE1/RE2) as GTC limits
                    re1 = max(0.025, atr.loc[prev_dt] / prev_close)  # 2.5% or 1×ATR deeper
                    re2 = max(0.050, 2 * atr.loc[prev_dt] / prev_close)
                    st.pending.append(dict(kind="RE1", price=open_px * (1 - re1), qty=int(qty/2), note="Rebuy -2.5%/1ATR"))
                    st.pending.append(dict(kind="RE2", price=open_px * (1 - re2), qty=qty - int(qty/2), note="Rebuy -5%/2ATR"))
                elif st.profit_stage == 1 and gain >= P.tp2:
                    open_px = opens.loc[dt]
                    qty = max(P.lot_size, int(shares / 2 // P.lot_size * P.lot_size))  # roughly another 1/3 of original
                    place_trade(dt, "SELL", open_px, qty, reason="TP +20% (2/3)")
                    st.profit_stage = 2
                    re1 = max(0.025, atr.loc[prev_dt] / prev_close)
                    re2 = max(0.050, 2 * atr.loc[prev_dt] / prev_close)
                    st.pending.append(dict(kind="RE1", price=open_px * (1 - re1), qty=int(qty/2), note="Rebuy -2.5%/1ATR"))
                    st.pending.append(dict(kind="RE2", price=open_px * (1 - re2), qty=qty - int(qty/2), note="Rebuy -5%/2ATR"))

        # 2) Compute drawdown & gates for TODAY's decisions
        dd_today = drawdown(dt)

        # Hard stop gate: if >=10%, liquidate & cooldown
        if dd_today >= P.dd_stop and shares > 0:
            open_px = opens.loc[dt]
            place_trade(dt, "SELL", open_px, shares, reason="DD>=10% Liquidate")
            st.cooldown_left = P.cooldown_days
            st.pending.clear()
            hard_stop_price = None

        # Handle cooldown decrement
        if st.cooldown_left > 0:
            st.cooldown_left -= 1

        # 3) BUY side: execute pending orders that meet open trigger (respect gates)
        # Apply freeze rules
        allow_new_buys = (st.cooldown_left == 0) and (dd_today < P.dd_freeze)

        # If in freeze, ignore all pending BUYs
        if not allow_new_buys:
            # clear all BUY-type pendings; keep none
            st.pending = [po for po in st.pending if po.get("kind","").startswith("CANCEL_ONLY")]
        else:
            open_px = opens.loc[dt]

            # If dd>=dd_warn, disable breakout A execution and RE1; keep PB & RE2
            allow_breakout = dd_today < P.dd_warn
            new_pending: List[Dict] = []
            executed_indices = set()

            for idx, po in enumerate(st.pending):
                kind = po['kind']
                price_trg = float(po['price'])
                qty = int(po['qty'])

                if qty <= 0:
                    continue

                # gating filters by kind
                if kind == "A" and not allow_breakout:
                    new_pending.append(po); continue
                if kind == "RE1" and dd_today >= P.dd_warn:
                    # skip RE1 when warn; keep RE2
                    continue

                # execution rules at OPEN only
                do_exec = False
                if kind in ("B1","B2","RE1","RE2","C"):
                    if open_px <= price_trg:
                        do_exec = True
                elif kind == "A":
                    if open_px >= price_trg:
                        do_exec = True

                if do_exec:
                    place_trade(dt, "BUY", open_px, qty, reason=f"Entry {kind}")
                    executed_indices.add(idx)
                    # set initial hard stop after entry
                    if shares > 0:
                        ap = price.loc[dt]
                        a = atr.loc[dt]
                        stop_dist = max(P.min_stop_pct, current_stop_mult(dd_today) * (a / ap))
                        hard_stop_price = open_px * (1 - stop_dist)
                    # special: after A fill, create C retest order (0.5 tranche) with deadline
                    if kind == "A":
                        st.breakout_ref_price = price_trg / (1 + P.breakout_buffer)  # approx High(N)
                        st.retest_deadline = dt + pd.Timedelta(days=P.retest_window_days)
                        # size later in creation step
                else:
                    new_pending.append(po)
            st.pending = new_pending

        # 4) REBUILD/REFRESH PENDING ORDERS for next day (GTC semantics)
        # Cancel outdated C
        if st.retest_deadline is not None and dt >= st.retest_deadline:
            st.retest_deadline = None
            # remove any existing C
            st.pending = [po for po in st.pending if po.get("kind") != "C"]

        # Trend filter must be ON to place A/B
        if trend_up.loc[dt] and st.cooldown_left == 0:
            # A: breakout (only if allowed in warn gate; execution gating handled above)
            bk_price = highN.loc[dt] * (1 + P.breakout_buffer)
            # size for one tranche at bk_price
            val = position_value_for_tranche(dt, bk_price)
            qty = round_qty(val, bk_price)
            if qty > 0:
                # ensure single A pending (replace old with new price/qty)
                st.pending = [po for po in st.pending if po.get("kind") != "A"]
                st.pending.append(dict(kind="A", price=bk_price, qty=qty, note="Breakout Buy Stop"))

            # B1/B2: pullback limits (non-OCO, both valid)
            prev_dt = df.index[max(i-1, 0)]
            prev_close = price.loc[prev_dt]
            ap = price.loc[dt]
            a = atr.loc[dt]
            depth1 = max(P.pb1_min_pct, P.pb1_atr_mult * (a / ap))
            depth2 = max(P.pb2_min_pct, P.pb2_atr_mult * (a / ap))
            b1_price = prev_close * (1 - depth1)
            b2_price = prev_close * (1 - depth2)

            # tranche size for each
            val1 = position_value_for_tranche(dt, b1_price)
            val2 = position_value_for_tranche(dt, b2_price)
            qty1 = round_qty(val1, b1_price)
            qty2 = round_qty(val2, b2_price)

            # refresh or upsert B1/B2
            st.pending = [po for po in st.pending if po.get("kind") not in ("B1","B2")]
            if qty1 > 0:
                st.pending.append(dict(kind="B1", price=b1_price, qty=qty1, note="Pullback Shallow"))
            if qty2 > 0:
                st.pending.append(dict(kind="B2", price=b2_price, qty=qty2, note="Pullback Deep"))

        else:
            # not in trend, remove A/B orders
            st.pending = [po for po in st.pending if po.get("kind") not in ("A","B1","B2")]

        # Create/refresh C (retest) after a breakout, valid for window; qty = 0.5 tranche
        if st.retest_deadline is not None and st.breakout_ref_price is not None and st.cooldown_left == 0:
            c_price = st.breakout_ref_price
            # ensure near EMA20 condition: use today EMA approx by ema20.loc[dt]; if price below EMA20, do not create
            if price.loc[dt] >= ema20.loc[dt]:
                valc = 0.5 * position_value_for_tranche(dt, c_price)
                qtyc = round_qty(valc, c_price)
                # upsert C
                st.pending = [po for po in st.pending if po.get("kind") != "C"]
                if qtyc > 0:
                    st.pending.append(dict(kind="C", price=c_price, qty=qtyc, note="Retest after Breakout"))

        # 5) record equity at close
        equity_curve.append(dict(date=dt, equity=float(equity_val(dt))))

    curve_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    label = "StrategyV12-Fusion"
    # ensure dtype/columns
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=['date','side','price','qty','fee','reason'])
    return curve_df.reset_index(drop=True), trades_df.reset_index(drop=True), label
