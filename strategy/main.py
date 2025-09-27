# main.py —— 自动识别市场；鲁棒 yfinance 取数；仅导出 XLSX 与对比 PNG
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

# ==== 修改这里的股票与区间 ====
SYMBOL = "AAPL"            # 示例：美股 AAPL；A股如 "000063.SZ"/"513130"; 港股如 "3067.HK"
START  = "2025-01-01"
END    = "2025-09-01"
INITIAL_CASH = 100_000.0
OUTPUT_DIR = "./output"

# ==== 数据获取：A股优先 AkShare；否则 yfinance ====
def _normalize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    """把任何 OHLC DataFrame 统一为 date/open/high/low/close 五列"""
    # 扁平列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]) for c in df.columns]
    # 全部小写去空格
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # 重命名常见别名
    ren = {}
    for col in list(df.columns):
        low = col.lower()
        if low.startswith("open") and "open" not in df.columns: ren[col] = "open"
        if low.startswith("high") and "high" not in df.columns: ren[col] = "high"
        if low.startswith("low")  and "low"  not in df.columns: ren[col] = "low"
        if low.startswith("close")and "close"not in df.columns: ren[col] = "close"
    if ren:
        df = df.rename(columns=ren)

    # 取 index->date
    df = df.reset_index()
    # 找到日期列
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "datetime", "time", "index"):
            date_col = c; break
    if date_col is None:
        # 兜底：就取第一列做日期
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})

    # 仅保留五列
    cols = df.columns
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(set(cols)):
        raise RuntimeError(f"无法标准化 OHLC 列，得到的列: {list(cols)}")

    out = df[["date","open","high","low","close"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["open","high","low","close"])
    out[["open","high","low","close"]] = out[["open","high","low","close"]].astype("float64")
    out = out[(out["open"]>0) & (out["high"]>0) & (out["low"]>0) & (out["close"]>0)]
    out = out.sort_values("date").reset_index(drop=True)
    return out

def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    # 1) A股：尝试 AkShare
    try:
        if symbol.upper().endswith((".SZ", ".SS")) or symbol.isdigit():
            import akshare as ak
            code = symbol.replace(".SZ", "").replace(".SS", "")
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""), adjust=""
            )
            df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low"})
            out = _normalize_ohlc_df(df.set_index("date")[["open","high","low","close"]])
            if len(out) > 0:
                return out
    except Exception as e:
        print("[INFO] AkShare 获取失败，转用 yfinance ：", e)

    # 2) 其他/备用：yfinance
    import yfinance as yf
    df = yf.download(
        symbol, start=start, end=end,
        progress=False, auto_adjust=False, actions=False, group_by="ticker", threads=False
    )
    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"yfinance 无法获取 {symbol} {start}~{end} 的数据")
    out = _normalize_ohlc_df(df)
    return out

# ==== 导入策略 ====
from strategy_v1 import backtest as run_v1
from strategy_v2 import backtest as run_v2

def ensure_outdir(path: str): os.makedirs(path, exist_ok=True)

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

def infer_market(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".SZ") or s.endswith(".SS") or s.isdigit():
        return "CN"
    elif s.endswith(".HK"):
        return "HK"
    else:
        return "US"

def save_strategy_xlsx(symbol: str, start: str, end: str, label: str,
                       curve: pd.DataFrame, trades: pd.DataFrame):
    ensure_outdir(OUTPUT_DIR)
    engine = get_excel_engine()
    if engine is None:
        raise RuntimeError("未检测到 openpyxl/xlsxwriter，请先安装：pip install openpyxl")
    safe_label = label.replace("/", "_").replace(" ", "")
    xlsx_path = os.path.join(OUTPUT_DIR, f"report_{symbol}_{start}_{end}_{safe_label}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine=engine, datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as w:
        curve.to_excel(w, sheet_name="equity", index=False)
        trades.to_excel(w, sheet_name="trades", index=False)
    print(f"[已导出] {label} -> {xlsx_path}")
    return xlsx_path

def save_comparison_xlsx(symbol: str, start: str, end: str,
                         curves: list[pd.DataFrame], labels: list[str]):
    ensure_outdir(OUTPUT_DIR)
    engine = get_excel_engine()
    if engine is None:
        raise RuntimeError("未检测到 openpyxl/xlsxwriter，请先安装：pip install openpyxl")
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

def save_comparison_png(symbol: str, start: str, end: str,
                        curves: list[pd.DataFrame], labels: list[str],
                        title: str | None = None, dpi: int = 160):
    ensure_outdir(OUTPUT_DIR)
    if title is None: title = f"{symbol} Backtest"
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for df, lb in zip(curves, labels):
        ax.plot(df["date"], df["equity"], label=lb)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Equity")
    ax.legend(); fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"{symbol}_{start}_{end}_comparison.png")
    fig.savefig(png_path, dpi=dpi); plt.close(fig)
    print(f"[已导出] 对比折线图 -> {png_path}")
    return png_path

def main():
    print(f"加载数据：{SYMBOL} {START}~{END}")
    bars = load_bars(SYMBOL, START, END)
    if len(bars) == 0:
        print("无数据"); return

    # —— 根据市场决定撮合/手续费等参数 —— #
    mkt = infer_market(SYMBOL)
    if mkt == "CN":
        kwargs = dict(lot_size=100, commission_rate=0.0003, commission_min=5.0, stamp_duty_sell=0.0005, slippage=0.0)
    elif mkt == "US":
        kwargs = dict(lot_size=1,   commission_rate=0.0000, commission_min=0.0, stamp_duty_sell=0.0,    slippage=0.0)
    else:  # HK（可按需求补全真实税费；这里先简化）
        kwargs = dict(lot_size=1,   commission_rate=0.0000, commission_min=0.0, stamp_duty_sell=0.0,    slippage=0.0)

    print("运行 Strategy V1 …")
    curve1, trades1, label1 = run_v1(bars, INITIAL_CASH, **kwargs)
    print("运行 Strategy V2 …")
    curve2, trades2, label2 = run_v2(bars, INITIAL_CASH, **kwargs)

    # 导出单策略 XLSX
    save_strategy_xlsx(SYMBOL, START, END, label1, curve1, trades1)
    save_strategy_xlsx(SYMBOL, START, END, label2, curve2, trades2)

    # 对齐日期后导出对比（XLSX + PNG）
    all_dates = pd.Series(sorted(set(curve1["date"]) | set(curve2["date"])))
    curve1a = pd.merge(all_dates.to_frame(name="date"), curve1, on="date", how="left").ffill()
    curve2a = pd.merge(all_dates.to_frame(name="date"), curve2, on="date", how="left").ffill()
    save_comparison_xlsx(SYMBOL, START, END, [curve1a, curve2a], [label1, label2])
    save_comparison_png(SYMBOL, START, END, [curve1a, curve2a], [label1, label2], title=f"{SYMBOL} Backtest")

    # 简要打印
    for label, df in [(label1, curve1), (label2, curve2)]:
        total_ret = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
        roll_max = df["equity"].cummax()
        mdd = (df["equity"] / roll_max - 1).min()
        print(f"[{label}] 总收益: {total_ret:.2%}, 最大回撤: {mdd:.2%}")

if __name__ == "__main__":
    main()
