# plot_signal_chart.py

import pandas as pd
import matplotlib.pyplot as plt


def plot_close_with_signal(csv_path: str, out_png: str = None):
    """
    csv_path : 生成的带 signal 的 CSV 路径
    out_png  : None 则直接 plt.show()
               非 None 则保存为 PNG 文件
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    if "close" not in df.columns or "signal" not in df.columns:
        raise ValueError("CSV 必须包含 'close' 和 'signal' 列")

    fig, ax = plt.subplots(figsize=(14, 6))

    # 1. 连续黑色 close 折线（底板）
    ax.plot(df.index, df["close"], color="black", linewidth=0.8)

    # 2. 对 signal 分段着色
    seg_id = (df["signal"] != df["signal"].shift(1)).cumsum()

    for _, seg in df.groupby(seg_id):
        sig = seg["signal"].iloc[0]

        if sig == 1:
            color = "green"
        elif sig == -1:
            color = "red"
        else:
            continue  # signal=0 不额外画，保持黑线

        ax.plot(seg.index, seg["close"], color=color, linewidth=1.5)

    ax.set_title("Close Price — Colored by Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close Price")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    if out_png is None:
        plt.show()
    else:
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[PLOT] 图像已保存到 {out_png}")


# ---------------------------------------------------------
# 手动执行：python plot_signal_chart.py path/to/file.csv
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot crypto close price with signal coloring.")
    parser.add_argument("csv_path", help="路径：带 signal 的 CSV 文件")
    parser.add_argument("--save", help="保存 PNG 的路径（不填则直接显示）", default=None)

    args = parser.parse_args()

    plot_close_with_signal(args.csv_path, args.save)
