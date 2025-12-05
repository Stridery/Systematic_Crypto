# label_training.py
# 管理 triple-barrier 参数组合 + 回测 Sharpe，选出最优组合，并把最优组合的带 signal 数据存成 CSV

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from label_generator import TripleBarrierLabelGenerator
from back_trader import SharpeBacktester
from plot_signal_chart import plot_close_with_signal  # ⬅ 新增：导入画图函数


def load_price_df(csv_path: str) -> pd.DataFrame:
    """
    读取原始 K 线 CSV，返回以 datetime 为索引的 DataFrame。
    要求 CSV 至少包含:
      - open_time_ms
      - open, high, low, close, volume
    其他列会原样保留。
    """
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["open_time_ms"], unit="ms")
    df = df.set_index("datetime").sort_index()
    return df


def main():
    # 1. 读入数据（路径你可以自己改 / 从配置里读）
    csv_path = "data/raw/BTCUSDT_1h.csv"
    price_df = load_price_df(csv_path)

    # 2. 初始化 triple-barrier label 生成器 & backtrader 封装
    label_gen = TripleBarrierLabelGenerator(
        price_df=price_df,
        price_cols=["open", "high", "low", "close", "volume"],
        pip_decimals=2,
    )

    backtester = SharpeBacktester(
        initial_cash=1.0,
        commission=0.0,
        periods_per_year=252,  # 日线的话可以用 252
    )

    # 3. 定义参数网格（你可以根据需要改范围）
    # SOL 日线：稍微细一点的网格
    TP_PCT_LIST = [
        0.001, 0.002, 0.0005
    ]  # 止盈 4% ~ 25%，偏细

    SL_PCT_LIST = [
        0.001, 0.002, 0.0005
    ]  # 止损 2% ~ 15%，稍微密一点

    HOLD_BARS_LIST = [
        1
    ]  # 持有 3~40 天，短中长都有

    param_grid = list(itertools.product(TP_PCT_LIST, SL_PCT_LIST, HOLD_BARS_LIST))

    best_result = None  # 用来保存最优组合及其指标

    print(f"[INFO] 总组合数: {len(param_grid)}")

    for tp_pct, sl_pct, hold_bars in tqdm(param_grid, desc="Grid search (triple-barrier + backtest)", leave=True):
        # 3.1 利用当前参数组合生成带 signal 的 DataFrame
        df_with_signal = label_gen.generate_signal_for_params(tp_pct, sl_pct, hold_bars)
        # 3.2 用 backtrader 回测，拿 Sharpe ratio
        metrics = backtester.run(df_with_signal)
        sharpe = metrics["sharpe_ratio"]

        # 可能会出现 Sharpe 为 NaN（比如没有任何仓位变化），这种情况直接跳过
        if np.isnan(sharpe):
            continue

        # 3.3 更新最优解
        if best_result is None or sharpe > best_result["metrics"]["sharpe_ratio"]:
            best_result = {
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "hold_bars": hold_bars,
                "metrics": metrics,
            }

    # 4. 输出最优结果 + 保存对应的带 signal 的 CSV
    if best_result is None:
        print("[RESULT] 没有得到任何有效的 Sharpe ratio（可能所有组合都没有成交）。")
    else:
        m = best_result["metrics"]
        print("\n" + "=" * 80)
        print("[RESULT] 最优 triple-barrier 参数组合：")
        print(f"  TP%        : {best_result['tp_pct']:.4f}")
        print(f"  SL%        : {best_result['sl_pct']:.4f}")
        print(f"  HOLD bars  : {best_result['hold_bars']}")
        print("-" * 80)
        print(f"  Initial    : {m['initial_value']:.6f}")
        print(f"  Final      : {m['final_value']:.6f}")
        print(f"  Total Ret  : {m['total_return']:.2%}")
        print(f"  Sharpe     : {m['sharpe_ratio']:.4f}")
        print(f"  Max DD     : {m['max_drawdown']:.2%}")
        print("=" * 80)

        # 4.1 用最优参数重新生成一次带 signal 的 DataFrame
        best_tp = best_result["tp_pct"]
        best_sl = best_result["sl_pct"]
        best_hold = best_result["hold_bars"]

        df_best = label_gen.generate_signal_for_params(best_tp, best_sl, best_hold)


if __name__ == "__main__":
    main()
