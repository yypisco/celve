# main.py  —— 仅导出 XLSX 到 ./output
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 修改这里的股票与区间 ====
SYMBOL = "000063.SZ"      # 示例：中兴通讯（深证）；上证用 ".SS"
START  = "2025-01-01"
END    = "2025-09-01"     # 含当日
INITIAL_CASH = 100_000.0

OUTPUT_DIR = "./output"

# ==== 数据获取：优先 AkShare，失败则 yfinance ====
def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import akshare as ak
        code = symbol.replace(".SZ", "").replace(".SS", "")
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""), adjust=""
        )
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low"})
        df["date"] = pd.to_datetime(df["date"])
        out = df[["date","open","high","low","close"]].sort_values("date").reset_index(drop=True)
        out = out[(out["open"]>0) & (out["high"]>0) & (out["low"]>0) & (out["close"]>0)]
        if len(out) > 0:
            return out
    except Exception as e:
        print("[INFO] AkShare 获取失败，尝试 yfinance ……", e)

    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False)
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

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def get_excel_engine():
    try:
        import openpyxl  # noqa
        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa
            return "xlsxwriter"
        except Exception:
            return None

def save_strategy_xlsx(symbol: str, start: str, end: str, label: str,
                       curve: pd.DataFrame, trades: pd.DataFrame):
    """单策略导出到 XLSX（equity / trades 两个工作表）"""
    ensure_outdir(OUTPUT_DIR)
    engine = get_excel_engine()
    if engine is None:
        raise RuntimeError("未检测到 openpyxl / xlsxwriter，请先安装：pip install openpyxl")

    safe_label = label.replace("/", "_").replace(" ", "")
    xlsx_path = os.path.join(OUTPUT_DIR, f"report_{symbol}_{start}_{end}_{safe_label}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine=engine, datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as w:
        curve.to_excel(w, sheet_name="equity", index=False)
        trades.to_excel(w, sheet_name="trades", index=False)
    print(f"[已导出] {label} -> {xlsx_path}")
    return xlsx_path

def save_comparison_xlsx(symbol: str, start: str, end: str,
                         curves: list, labels: list[str]):
    """把多条净值放在同一个 XLSX 的一个 sheet 里（equity_compare）"""
    ensure_outdir(OUTPUT_DIR)
    engine = get_excel_engine()
    if engine is None:
        raise RuntimeError("未检测到 openpyxl / xlsxwriter，请先安装：pip install openpyxl")

    # 合并日期并前向填充
    all_dates = pd.Series(sorted(set().union(*[set(c["date"]) for c in curves])))
    df = pd.DataFrame({"date": pd.to_datetime(all_dates)})
    for c, lb in zip(curves, labels):
        df = df.merge(c[["date","equity"]].rename(columns={"equity": lb}), on="date", how="left")
    df = df.sort_values("date").ffill()

    xlsx_path = os.path.join(OUTPUT_DIR, f"{symbol}_{start}_{end}_comparison.xlsx")
    with pd.ExcelWriter(xlsx_path, engine=engine, datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as w:
        df.to_excel(w, sheet_name="equity_compare", index=False)
    print(f"[已导出] 对比总表 -> {xlsx_path}")
    return xlsx_path, df

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
        print("无数据"); return

    print("运行 Strategy V1 …")
    curve1, trades1, label1 = run_v1(bars, INITIAL_CASH)
    print("运行 Strategy V2 …")
    curve2, trades2, label2 = run_v2(bars, INITIAL_CASH)

    # 只导出 XLSX
    save_strategy_xlsx(SYMBOL, START, END, label1, curve1, trades1)
    save_strategy_xlsx(SYMBOL, START, END, label2, curve2, trades2)

    # 对比 XLSX
    save_comparison_xlsx(SYMBOL, START, END, [curve1, curve2], [label1, label2])

    # 画图
    plot_curves([curve1, curve2], [label1, label2], title=f"{SYMBOL} Backtest")

    # 简单打印
    for label, df in [(label1, curve1), (label2, curve2)]:
        total_ret = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
        roll_max = df["equity"].cummax()
        mdd = (df["equity"] / roll_max - 1).min()
        print(f"[{label}] 总收益: {total_ret:.2%}, 最大回撤: {mdd:.2%}")

if __name__ == "__main__":
    main()
