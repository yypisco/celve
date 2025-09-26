# main.py
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==== 修改这里的股票与区间 ====
SYMBOL = "000063.SZ"      # 中兴通讯 A股 -> SZ；上证用 ".SS"（如 600519.SS）
START  = "2025-01-01"
END    = "2025-09-01"     # 含当日；如想“到9/1不含”，可以设 2025-09-02
INITIAL_CASH = 100_000.0

# ==== 数据获取：优先 AkShare，失败则 yfinance ====
def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    # 先尝试 AkShare
    try:
        import akshare as ak
        code = symbol.replace(".SZ", "").replace(".SS", "")
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=""
        )
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low"})
        df["date"] = pd.to_datetime(df["date"])
        out = df[["date","open","high","low","close"]].sort_values("date").reset_index(drop=True)
        out = out[(out["open"]>0) & (out["high"]>0) & (out["low"]>0) & (out["close"]>0)]
        if len(out) > 0:
            return out
    except Exception as e:
        print("[INFO] AkShare 获取失败，尝试 yfinance ……", e)

    # yfinance 备用
    try:
        import yfinance as yf
        df = yf.download(symbol, start=START, end=END, progress=False)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
        df = df.reset_index().rename(columns={"Date":"date"})
        out = df[["date","open","high","low","close"]].sort_values("date").reset_index(drop=True)
        out = out[(out["open"]>0) & (out["high"]>0) & (out["low"]>0) & (out["close"]>0)]
        return out
    except Exception as e:
        print("[ERROR] yfinance 获取也失败：", e)
        raise

# ==== 导入两个策略 ====
from strategy_v1 import backtest as run_v1
from strategy_v2 import backtest as run_v2

def plot_curves(curves, labels, title="Backtest Equity Curves"):
    plt.figure(figsize=(10,5))
    for df, lb in zip(curves, labels):
        plt.plot(df["date"], df["equity"], label=lb)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Equity (CNY)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print(f"加载数据：{SYMBOL} {START}~{END}")
    bars = load_bars(SYMBOL, START, END)
    if len(bars) == 0:
        print("无数据")
        return

    print("运行 Strategy V1 …")
    curve1, trades1, label1 = run_v1(bars, INITIAL_CASH)

    print("运行 Strategy V2 …")
    curve2, trades2, label2 = run_v2(bars, INITIAL_CASH)

    # 对齐日期，以免长度不同
    merged_dates = pd.Series(sorted(set(curve1["date"]) | set(curve2["date"])))
    curve1 = pd.merge(merged_dates.to_frame(name="date"), curve1, on="date", how="left").ffill()
    curve2 = pd.merge(merged_dates.to_frame(name="date"), curve2, on="date", how="left").ffill()

    plot_curves([curve1, curve2], [label1, label2], title=f"{SYMBOL} Backtest")

    # 打印简单汇总
    for label, df in [(label1, curve1), (label2, curve2)]:
        total_ret = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
        roll_max = df["equity"].cummax()
        mdd = (df["equity"] / roll_max - 1).min()
        print(f"[{label}] 总收益: {total_ret:.2%}, 最大回撤: {mdd:.2%}")

if __name__ == "__main__":
    main()
