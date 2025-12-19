# pipeline/make_signal.py
import pandas as pd
from pathlib import Path


class SignalGenerator:
    """
    信号生成器，用于生成交易信号
    """
    
    def __init__(
        self,
        raw_path: Path = Path("data/intermediate/btc_1h_features.csv"),
        out_dir: Path = Path("data/processed"),
        out_path: Path = None,
        threshold: float = 0.001,
        delta: float = 0.0004,
    ):
        """
        初始化信号生成器
        
        Args:
            raw_path: 输入特征数据路径
            out_dir: 输出目录
            out_path: 输出文件路径，如果为None则使用默认路径
            threshold: 主阈值，未来一根K线相对涨跌幅
            delta: 去噪安全带，贴着阈值的一小圈不要（只用于训练集过滤）
        """
        self.raw_path = raw_path
        self.out_dir = out_dir
        if out_path is None:
            self.out_path = out_dir / "btc_1h_features_signal.csv"
        else:
            self.out_path = out_path
        self.threshold = threshold
        self.delta = delta

    def make_signal(self, df: pd.DataFrame,
                    threshold: float = None,
                    delta: float = None) -> pd.DataFrame:
        """
        基于"下一根K线的 close"生成三分类 signal：
        - ret_next >=  threshold         -> signal =  1
        - ret_next <= -threshold         -> signal = -1
        - 其他                            -> signal =  0

        同时增加去噪标记 is_strong：
        - 明显上涨:   ret_next >= threshold + delta
        - 明显下跌:   ret_next <= -threshold - delta
        - 明显震荡:   |ret_next| <= threshold - delta
        - 上面三种情况 is_strong = 1，其余 is_strong = 0

        注意：这里 **不删任何行**，只增加列：
          - ret_next_1h
          - signal
          - is_strong

        最后一行由于没有 next_close，会被删掉一行（NaN）。
        """
        if threshold is None:
            threshold = self.threshold
        if delta is None:
            delta = self.delta

        # 1. 按时间排序
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
        else:
            raise ValueError("Input df must contain 'datetime' column.")

        # 2. 下一根K线的收盘价 & 未来1小时收益
        df["next_close"] = df["close"].shift(-1)
        df["ret_next_1h"] = df["next_close"] / df["close"] - 1

        # 3. 三分类 signal（不去噪）
        ret_next = df["ret_next_1h"]
        df["signal"] = 0
        df.loc[ret_next >= threshold, "signal"] = 1
        df.loc[ret_next <= -threshold, "signal"] = -1

        # 4. 去噪标记（只打 tag，不删行）
        strong_up = ret_next >= (threshold + delta)
        strong_down = ret_next <= -(threshold + delta)
        strong_flat = ret_next.abs() <= (threshold - delta)

        df["is_strong"] = 0
        df.loc[strong_up | strong_down | strong_flat, "is_strong"] = 1

        # 5. 丢掉没有 next_close 的最后一行
        df = df.dropna(subset=["next_close", "ret_next_1h"]).reset_index(drop=True)

        # 6. 不再需要 next_close 这列
        df = df.drop(columns=["next_close"])

        return df

    def generate_signal(self):
        """
        从特征数据生成信号并保存到文件
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[make_signal] Reading feature data from {self.raw_path} ...")
        df = pd.read_csv(self.raw_path)

        print("[make_signal] input columns:", df.columns.tolist())

        df = self.make_signal(df, self.threshold, self.delta)

        print(f"[make_signal] Save data with signal to {self.out_path} ...")
        df.to_csv(self.out_path, index=False)

        print("[make_signal] head:")
        print(df.head())
        print("[make_signal] tail:")
        print(df.tail())

        print(f"[make_signal] Total samples (no denoise drop): {len(df)}")
        print("[make_signal] signal value counts:")
        print(df["signal"].value_counts().sort_index())
        print("[make_signal] is_strong value counts:")
        print(df["is_strong"].value_counts().sort_index())


# 为了保持向后兼容，提供函数接口
def make_signal(df: pd.DataFrame,
                threshold: float = 0.001,
                delta: float = 0.0004) -> pd.DataFrame:
    """
    向后兼容的函数接口
    """
    generator = SignalGenerator(threshold=threshold, delta=delta)
    return generator.make_signal(df, threshold, delta)


def main():
    """
    向后兼容的函数接口
    """
    generator = SignalGenerator()
    generator.generate_signal()


if __name__ == "__main__":
    main()
