# label_generator.py
# 单次 triple-barrier 执行：给定 DataFrame + (tp, sl, hold)，生成 signal 列并返回新的 DataFrame

import numpy as np
import pandas as pd
from triple_barrier.trading import TradingParameters, DataSetLabeler
from triple_barrier.trade_labeling import TradeSide


class TripleBarrierLabelGenerator:
    """
    Triple-barrier 打 label 封装：
    - 初始化时接收已经整理好的价数据 DataFrame（index 为 DatetimeIndex）
    - 对每一组 (tp_pct, sl_pct, hold_bars) 运行一次 triple-barrier
    - 返回在原 DataFrame 上附加 'signal' 列的 DataFrame
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        price_cols=None,
        pip_decimals: int = 2,
    ):
        """
        price_df   : 已经 set_index 为 datetime 的 K 线数据（Binance 风格亦可）
                     要求至少包含 open/high/low/close/volume 列
        price_cols : 用于 triple-barrier 的价格列名
        pip_decimals : pip 精度（2 => pip=0.01）
        """
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df.index 必须是 DatetimeIndex，请先处理好时间索引。")

        if price_cols is None:
            price_cols = ["open", "high", "low", "close", "volume"]

        self.original_df = price_df.sort_index().copy()
        self.price_cols = price_cols
        self.pip_decimals = pip_decimals

        # 仅用于 triple-barrier 的价格子表
        self.px = self.original_df[self.price_cols].astype(float).copy()

        # 构造 entry_mark（简单版：每根 bar 都尝试开一笔）
        self.entry_mark = self._build_entry_mark(self.px.index)

    # ========== 内部工具函数 ==========

    @staticmethod
    def _build_entry_mark(index: pd.Index) -> pd.Series:
        """
        简单版：假设“每根 bar 都尝试开一笔”（下一根开盘价入场）。
        后面你可以把这里替换成自己的信号。
        """
        s = index.to_series().notna().astype(int)
        return s.shift(1).fillna(0).astype(int)

    def _pct_to_pips(self, pct: float, ref_price: float) -> int:
        """
        百分比宽度 → 固定 pip 宽度（triple-barrier 需要的是 pips）
        """
        pip_size = 10 ** (-self.pip_decimals)
        width_price = ref_price * pct
        return int(round(width_price / pip_size))

    # ========== triple-barrier 调用 ==========

    def _run_one_side(
        self,
        tp_pct: float,
        sl_pct: float,
        hold_bars: int,
        side: TradeSide,
    ) -> pd.DataFrame:
        """
        对单一方向（BUY 或 SELL）跑 triple-barrier，返回每笔交易明细：
        open_dt, open_price, close_dt, close_price, close_reason, profit (pips)
        """
        ref_price = self.px["close"].median()
        tp_width = self._pct_to_pips(tp_pct, ref_price)
        sl_width = self._pct_to_pips(sl_pct, ref_price)

        params = TradingParameters(
            open_price=self.px["open"],
            high_price=self.px["high"],
            low_price=self.px["low"],
            close_price=self.px["close"],
            entry_mark=self.entry_mark,
            stop_loss_width=sl_width,
            take_profit_width=tp_width,
            pip_decimal_position=self.pip_decimals,
            time_barrier_periods=hold_bars,
            trade_side=side,
            dynamic_exit=None,
        )

        tb = DataSetLabeler(params).compute()
        # tb 的列是：
        # ['open', 'high', 'low', 'close', 'entry', 'close-price',
        #  'close-datetime', 'close-type', 'profit']

        # 映射成后续逻辑用的列名
        tb = tb.rename(
            columns={
                "open": "open_price",
                "close-price": "close_price",
                "close-datetime": "close_dt",
                "close-type": "close_reason",
            }
        )

        # 开仓时间 = index
        tb["open_dt"] = tb.index

        tb = tb[
            ["open_dt", "open_price", "close_dt", "close_price", "close_reason", "profit"]
        ].copy()
        return tb

    def _combine_long_short_and_label(
        self,
        long_df: pd.DataFrame,
        short_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        现在的 label 规则：
        - 只用多头 long 的 close_reason 来打 label：
          * take-profit → +1
          * stop-loss   → -1
          * 其他        → 0

        返回 DataFrame（index=open_dt, 含 label）
        """
        m = long_df.copy()

        def decide_label(reason) -> int:
            r = str(reason).lower()
            if r == "take-profit":
                return 1
            if r == "stop-loss":
                return -1
            return 0

        m["label"] = m["close_reason"].apply(decide_label).astype(int)
        m = m.set_index("open_dt").sort_index()
        return m[["label"]]

    # ========== 对外接口：单次执行 ==========

    def generate_signal_for_params(
        self,
        tp_pct: float,
        sl_pct: float,
        hold_bars: int,
    ) -> pd.DataFrame:
        """
        给定一组 (tp_pct, sl_pct, hold_bars)，执行一次 triple-barrier：
        - 生成 label（index=open_dt）
        - 对齐到原始 DataFrame 的时间索引
        - 在原 DataFrame 上新增 'signal' 列并返回

        返回：带有 signal 列的新 DataFrame（不修改原 self.original_df）
        """
        long_df = self._run_one_side(tp_pct, sl_pct, hold_bars, TradeSide.BUY)
        short_df = self._run_one_side(tp_pct, sl_pct, hold_bars, TradeSide.SELL)
        label_df = self._combine_long_short_and_label(long_df, short_df)

        # 对齐到完整的时间轴上，没有 label 的地方填 0（视为中性 / 无信号）
        out = self.original_df.copy()
        out["signal"] = label_df["label"].reindex(out.index).fillna(0).astype(int)
        return out
