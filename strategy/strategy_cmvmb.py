
# strategy_cmvmb.py
# CMVMB: Composite Momentum + Volatility Target + Breadth (single-asset compatible)
# Signature & outputs are aligned with strategy_v1/strategy_v2 to plug into main.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    # momentum windows in trading days (3/6/12 months), skip last 1 month to avoid short-term reversal
    mom_windows: Tuple[int, int, int] = (63, 126, 252)
    mom_skip: int = 21
    mom_weights: Tuple[float, float, float] = (0.25, 0.35, 0.40)

    # time-series momentum filters
    ts_use_sma200: bool = True
    ts_use_abs_mom: bool = True
    ts_window_abs_mom: int = 252  # 12 months

    # breadth gating (works even for单标的：用过去N天 True 比例代理广度）
    breadth_ma: int = 200
    breadth_floor: float = 0.30
    breadth_mid: float = 0.50
    risk_floor: float = 0.25
    risk_mid: float = 0.50
    risk_full: float = 1.00

    # portfolio-level volatility targeting (Moreira & Muir style)
    target_annual_vol: float = 0.12
    cov_lookback: int = 63
    max_gross_leverage: float = 1.0

    # rebalancing & costs
    fee_bps: float = 1.0       # one-way commission (bps) approximation
    slippage_bps: float = 1.0  # one-way slippage (bps)
    rebalance_freq: str = 'M'  # month end

    # sanity
    min_history: int = 20


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
    bars : DataFrame with columns ['date','open','high','low','close'] (ascending by date)
    initial_cash : starting cash
    market cost params are accepted for兼容性; we use commission_rate/commission_min/stamp_duty_sell/slippage

    Returns
    -------
    curve_df : DataFrame with columns ['date','equity']
    trades_df: DataFrame with columns ['date','side','price','qty','fee','reason']
    label    : str
    """
    cfg = Cfg()

    df = bars.copy()
    df = df[['date','open','high','low','close']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    px = df.set_index('date')['close']
    op = df.set_index('date')['open']
    rets = px.pct_change().fillna(0.0)

    # --- features ---
    # risk-adjusted composite momentum (one-month skip)
    vol20 = rets.rolling(20).std() * np.sqrt(252.0)
    comp_mom = 0.0
    for w, wgt in zip(cfg.mom_windows, cfg.mom_weights):
        r = (px.shift(cfg.mom_skip) / px.shift(w + cfg.mom_skip) - 1.0)
        s = (r / (vol20.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
        comp_mom = comp_mom + wgt * s

    # time-series filter
    ts_filter = pd.Series(True, index=px.index)
    if cfg.ts_use_sma200:
        sma200 = px.rolling(200, min_periods=1).mean()
        ts_filter &= (px.shift(cfg.mom_skip) > sma200.shift(cfg.mom_skip)).fillna(False)
    if cfg.ts_use_abs_mom:
        abs_m = px.shift(cfg.mom_skip) / px.shift(cfg.ts_window_abs_mom + cfg.mom_skip) - 1.0
        if len(px.dropna()) >= (cfg.ts_window_abs_mom + cfg.mom_skip + 1):
            ts_filter &= (abs_m > 0).fillna(False)
        else:
            # insufficient history: do not block positions by abs momentum
            ts_filter &= True

    # breadth proxy: for单标的=过去N天 ts_filter为True的比例（再平滑）
    breadth = ts_filter.astype(float).rolling(cfg.breadth_ma, min_periods=1).mean().rolling(5, min_periods=1).mean()

    # month-end rebalance dates
    rebal_dates = _month_ends(df['date'])

    # simulate
    cash = float(initial_cash)
    shares = 0
    equity_curve: List[dict] = []
    trades: List[dict] = []

    # helper to estimate current daily port vol (single asset ~ its daily vol * |weight|)
    def est_daily_vol(date):
        hist = rets.loc[:date].tail(cfg.cov_lookback)
        return float(hist.std()) if len(hist) > 2 else 0.0

    prev_date = None
    for d, row in df.iterrows():
        date = row['date']
        close = row['close']
        equity = cash + shares * close
        equity_curve.append(dict(date=date, equity=float(equity)))

        # handle rebalance signals at month-end (execute next trading day at open)
        if date in rebal_dates:
            prev_date = date
            continue

        # if yesterday was a rebalance date => execute today at open
        if prev_date is not None and prev_date in rebal_dates and date > prev_date:
            # compute target exposure on prev_date
            filt = bool(ts_filter.loc[prev_date]) if prev_date in ts_filter.index else False

            # base weight: 1 if in uptrend, else 0
            base_w = 1.0 if filt else 0.0

            # vol targeting
            est_vol_daily = est_daily_vol(prev_date)
            target_vol_daily = cfg.target_annual_vol / math.sqrt(252.0)
            scale = (target_vol_daily / est_vol_daily) if est_vol_daily > 0 else 0.0
            scale = float(np.clip(scale, 0.0, cfg.max_gross_leverage))

            # breadth multiplier
            br = float(breadth.loc[prev_date]) if prev_date in breadth.index else 1.0
            if np.isnan(br):
                risk_mult = cfg.risk_full
            elif br < cfg.breadth_floor:
                risk_mult = cfg.risk_floor
            elif br < cfg.breadth_mid:
                risk_mult = cfg.risk_mid
            else:
                risk_mult = cfg.risk_full

            target_w = base_w * scale * risk_mult

            # translate to shares using today's open (apply slippage & costs)
            price = float(op.loc[date])
            # one-way friction for sizing preview (bps + provided slippage param)
            slip = max(cfg.slippage_bps / 10000.0, 0.0) + max(slippage, 0.0)

            desired_value = equity * target_w
            target_shares = int((desired_value / price) // lot_size * lot_size)

            delta = target_shares - shares
            if delta != 0:
                side = "BUY" if delta > 0 else "SELL"
                q = abs(delta)

                trade_price = price * (1.0 + slip if side == "BUY" else 1.0 - slip)
                gross = trade_price * q

                # commissions
                fee_comm = max(gross * commission_rate, commission_min) if commission_rate > 0 else 0.0
                fee_tax = gross * stamp_duty_sell if side == "SELL" and stamp_duty_sell > 0 else 0.0
                fee = float(fee_comm + fee_tax)

                if side == "BUY":
                    total = gross + fee
                    if total <= cash:
                        cash -= total; shares += q
                        trades.append(dict(date=date, side=side, price=trade_price, qty=q, fee=fee, reason="Monthly Rebalance"))
                else:
                    proceed = gross - fee
                    cash += proceed; shares -= q
                    trades.append(dict(date=date, side=side, price=trade_price, qty=q, fee=fee, reason="Monthly Rebalance"))

            # reset flag so we do not trade again
            prev_date = None

    curve_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    label = "StrategyCMVMB(Momentum/TS/VolTarget/Breadth)"
    return curve_df, trades_df, label



def _month_ends(dates: pd.Series) -> set:
    dates = pd.to_datetime(dates)
    # Use a Series so groupby returns a like-indexed object; take max date per (year, month)
    ser = pd.Series(dates.values, index=dates)
    g = ser.groupby([pd.DatetimeIndex(dates).year, pd.DatetimeIndex(dates).month], group_keys=False).max()
    return set(pd.to_datetime(g.values))
