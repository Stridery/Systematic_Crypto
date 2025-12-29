import pandas as pd
import talib
from pathlib import Path


class FeatureGenerator:
    """
    特征生成器，用于从原始数据生成技术指标特征
    """
    
    def __init__(
        self,
        in_path: Path = Path("data/raw/BTCUSDT_1h.csv"),
        out_dir: Path = Path("data/intermediate"),
        out_path: Path = None,
    ):
        """
        初始化特征生成器
        
        Args:
            in_path: 输入数据路径
            out_dir: 输出目录
            out_path: 输出文件路径，如果为None则使用默认路径
        """
        self.in_path = in_path
        self.out_dir = out_dir
        if out_path is None:
            self.out_path = out_dir / "btc_1h_features.csv"
        else:
            self.out_path = out_path

    def add_return_and_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 收益率
        df["ret_1h"] = df["close"].pct_change(1)
        df["ret_3h"] = df["close"].pct_change(3)
        df["ret_6h"] = df["close"].pct_change(6)
        df["ret_12h"] = df["close"].pct_change(12)

        # 滚动波动率（realized vol）
        df["vol_6h"] = df["ret_1h"].rolling(6).std()
        df["vol_24h"] = df["ret_1h"].rolling(24).std()

        # 振幅 / 区间
        df["range_1h"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["range_6h"] = (
            df["high"].rolling(6).max() - df["low"].rolling(6).min()
        ) / df["close"].rolling(6).mean()

        return df

    def add_trend_and_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ===== 1) close / MA，比原始 MA 更稳 =====
        for win in [5, 10, 20]:
            ma_col = f"ma_{win}"
            ma = df[ma_col]
            df[f"close_over_ma_{win}"] = df["close"] / ma

        # ===== 2) BOLL 宽度（波动率 proxy）=====
        df["boll20_width"] = (df["boll20_top"] - df["boll20_bot"]) / df["boll20_mid"]
        df["boll40_width"] = (df["boll40_top"] - df["boll40_bot"]) / df["boll40_mid"]

        # ===== 3) BOLL 位置：价格在带内的位置 =====
        band20_range = (df["boll20_top"] - df["boll20_bot"]).replace(0, pd.NA)
        band40_range = (df["boll40_top"] - df["boll40_bot"]).replace(0, pd.NA)

        df["boll20_pos"] = (df["close"] - df["boll20_mid"]) / band20_range
        df["boll40_pos"] = (df["close"] - df["boll40_mid"]) / band40_range

        # ===== 4) BOLL squeeze：带宽收缩/扩张程度（只做 20 带）=====
        width20_ma = df["boll20_width"].rolling(200).mean()
        df["boll20_squeeze"] = df["boll20_width"] / width20_ma

        # ===== 5) MA 斜率（趋势强度）=====
        df["ma20_slope_4h"] = df["ma_20"] - df["ma_20"].shift(4)
        df["ma10_slope_3h"] = df["ma_10"] - df["ma_10"].shift(3)

        # ===== 6) MA 相对位置 / 间距（多头/空头排列 + 趋势力度）=====
        df["ma5_over_ma20"] = df["ma_5"] / df["ma_20"]
        df["ma10_over_ma20"] = df["ma_10"] / df["ma_20"]

        df["ma_spread_5_20"] = df["ma_5"] - df["ma_20"]
        df["ma_spread_10_20"] = df["ma_10"] - df["ma_20"]

        # ===== 7) MA 金叉 / 死叉 事件（0/1 特征，适合树模型）=====
        # 向上金叉：昨天 5MA 在 20MA 下方，今天上穿
        cross_up = (
            (df["ma_5"].shift(1) < df["ma_20"].shift(1)) &
            (df["ma_5"] >= df["ma_20"])
        )
        df["ma5_cross_ma20"] = cross_up.astype(int)

        # 向下死叉：昨天 5MA 在 20MA 上方，今天下穿
        cross_down = (
            (df["ma_5"].shift(1) > df["ma_20"].shift(1)) &
            (df["ma_5"] <= df["ma_20"])
        )
        df["ma5_crossdown_ma20"] = cross_down.astype(int)

        # ===== 8) 成交量 / OBV 相关（你原来就有的）=====
        df["volume_change_1h"] = df["volume"].pct_change(1)
        df["volume_change_6h"] = df["volume"].pct_change(6)
        df["obv_slope_6h"] = df["obv"] - df["obv"].shift(6)

        return df

    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用 quote_asset_volume / num_trades / taker_* 做微观结构特征
        """
        # 1) 成交额相关（USDT）
        if "quote_asset_volume" in df.columns:
            df["volume_usdt"] = df["quote_asset_volume"].astype(float)
            df["volume_usdt_change_1h"] = df["volume_usdt"].pct_change(1)
            df["volume_usdt_change_6h"] = df["volume_usdt"].pct_change(6)
            df["volume_usdt_over_ma_24h"] = (
                df["volume_usdt"] / df["volume_usdt"].rolling(24).mean()
            )

        # 2) 成交笔数相关
        if "num_trades" in df.columns:
            df["num_trades_change_1h"] = df["num_trades"].pct_change(1)
            df["num_trades_change_6h"] = df["num_trades"].pct_change(6)
            df["num_trades_over_ma_24h"] = (
                df["num_trades"] / df["num_trades"].rolling(24).mean()
            )

        # 3) taker 主动买入：order flow
        if "taker_buy_base" in df.columns and "volume" in df.columns:
            vol = df["volume"].replace(0, pd.NA)
            df["taker_buy_ratio"] = df["taker_buy_base"] / vol
            df["order_flow_imbalance"] = (2 * df["taker_buy_base"] - vol) / vol

        # 4) 按金额衡量的主动买入强度
        if "taker_buy_quote" in df.columns and "quote_asset_volume" in df.columns:
            qvol = df["quote_asset_volume"].replace(0, pd.NA)
            df["taker_buy_quote_ratio"] = df["taker_buy_quote"] / qvol

        return df

    def add_rsi_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) RSI 斜率
        df["rsi_14_slope_1"] = df["rsi_14"] - df["rsi_14"].shift(1)
        df["rsi_14_slope_3"] = df["rsi_14"] - df["rsi_14"].shift(3)

        # 2) 超买 / 超卖 flag
        df["rsi_14_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_14_oversold"] = (df["rsi_14"] < 30).astype(int)

        # 3) 多时间尺度差值
        df["rsi_6_14_diff"] = df["rsi_6"] - df["rsi_14"]
        df["rsi_14_28_diff"] = df["rsi_14"] - df["rsi_28"]

        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            给一些核心特征加 lag，让模型看到"过去几小时的状态"
        """
        lag_cols = [
            "ret_1h",
            "ret_3h",
            "rsi_14",
            "vol_6h",
            "close_over_ma_20",
        ]
        lags = [1, 2, 3]

        for c in lag_cols:
            if c not in df.columns:
                continue
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)

        return df

    def add_obv_features(self, df):
        # ----- OBV slope -----
        df["obv_slope_3"] = df["obv"] - df["obv"].shift(3)
        df["obv_slope_6"] = df["obv"] - df["obv"].shift(6)
        df["obv_slope_12"] = df["obv"] - df["obv"].shift(12)

        # ----- 标准化 OBV（rolling z-score）-----
        df["obv_norm_20"] = (df["obv"] - df["obv"].rolling(20).mean()) / df["obv"].rolling(20).std()

        # ----- 量价相关性 -----
        df["obv_price_corr_6"] = df["obv"].rolling(6).corr(df["close"])
        df["obv_price_corr_12"] = df["obv"].rolling(12).corr(df["close"])

        # ----- OBV 动能 -----
        df["obv_momentum"] = df["obv"].diff(1)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        return df

    def generate_features(self, skip_if_exists: bool = True):
        """
        生成所有特征并保存到文件
        
        Args:
            skip_if_exists: 如果输出文件已存在是否跳过
        """
        # 检查输出文件是否已存在
        if skip_if_exists and self.out_path.exists():
            print(f"[FeatureGenerator] Output file already exists: {self.out_path}")
            print("[FeatureGenerator] Skipping feature generation.")
            return
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.in_path)

        # ===== 1. 用 open_time_ms 转成 datetime =====
        df["datetime"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert(None)
        df = df.sort_values("datetime").reset_index(drop=True)

        # ===== 2. 转成 float =====
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 丢时间戳，保留 microstructure 相关列
        drop_raw_cols = [
            "open_time_ms",
            "close_time_ms",
        ]
        df = df.drop(columns=[c for c in drop_raw_cols if c in df.columns])

        close = df["close"].values
        volume = df["volume"].values

        # ===== 3. 原有技术指标 =====
        # RSI
        df["rsi_6"] = talib.RSI(close, timeperiod=6)
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
        df["rsi_28"] = talib.RSI(close, timeperiod=28)

        # BOLL(20, 40)
        up20, mid20, low20 = talib.BBANDS(close, timeperiod=20)
        up40, mid40, low40 = talib.BBANDS(close, timeperiod=40)
        df["boll20_top"] = up20
        df["boll20_mid"] = mid20
        df["boll20_bot"] = low20
        df["boll40_top"] = up40
        df["boll40_mid"] = mid40
        df["boll40_bot"] = low40

        # MA
        df["ma_5"] = talib.SMA(close, timeperiod=5)
        df["ma_10"] = talib.SMA(close, timeperiod=10)
        df["ma_20"] = talib.SMA(close, timeperiod=20)

        # OBV（用基础币数量 volume）
        df["obv"] = talib.OBV(close, volume)

        # ===== 4. 新增特征：return / vol / trend / microstructure / lag / time =====
        df = self.add_return_and_vol_features(df)
        df = self.add_trend_and_structure_features(df)
        df = self.add_microstructure_features(df)
        df = self.add_rsi_derived_features(df)
        df = self.add_lag_features(df)
        df = self.add_time_features(df)

        # 丢掉有 NaN 的起始部分（指标和 rolling 的前几行）
        df = df.dropna().reset_index(drop=True)

        # ===== 5. 精简掉强共线、重复度高的特征 =====
        # 目前先不精简，后面可以根据 feature importance 再删
        redundant_cols: list[str] = []
        df = df.drop(columns=[c for c in redundant_cols if c in df.columns])

        # ===== 6. 只保留：datetime + OHLC + 其他都当 feature =====
        base_cols = ["datetime", "open", "high", "low"]
        feature_cols = [c for c in df.columns if c not in base_cols]
        cols_keep = base_cols + feature_cols

        df = df[cols_keep]
        df.to_csv(self.out_path, index=False)

        print(f"Saved to {self.out_path}")
        print("Num features:", len(feature_cols))
        print("Columns:", df.columns.tolist())
        print(df.head())


# 为了保持向后兼容，提供函数接口
def main():
    """
    向后兼容的函数接口
    """
    generator = FeatureGenerator()
    generator.generate_features()


if __name__ == "__main__":
    main()
