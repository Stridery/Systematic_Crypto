# back_trader.py

import numpy as np
import pandas as pd
import backtrader as bt


class _PandasDataWithSignal(bt.feeds.PandasData):
    """
    内部使用：在标准 PandasData 上加一条 'signal' 线。
    要求传进来的 DataFrame 有列：
      - 'open', 'high', 'low', 'close', 'volume', 'signal'
    index 为 DatetimeIndex。
    """
    # 在父类基础上增加一条 'signal' 线
    lines = ('signal',)

    # 这里只新增 signal 的映射，其余 open/high/low/close/volume 用父类默认列名匹配
    params = (
        ('signal', -1),  # -1 表示按列名 'signal' 自动匹配 DataFrame 里的 'signal' 列
    )


class _ScalingSignalStrategy(bt.Strategy):
    """
    long-short 策略（按“币的个数”来操作）：

    - signal =  1：多头方向信号
    - signal = -1：空头方向信号
    - signal =  0：hold，不操作

    规则：
    - 初始空仓；
    - 每次连续收到相同方向的信号（1 或 -1），目标持仓数量在该方向上 +1；
    - 最大持仓数量为 ±max_units（比如 5 个币）；
    - 方向反转时：
        * 多头 -> 空头：直接把目标仓位改成  -1, -2, ...（从 1 个空单开始累加）
        * 空头 -> 多头：同理；
    - signal = 0：不调仓，保持当前持仓数量不变。
    """

    params = dict(
        max_units=5,   # 最大持仓数量（+5 / -5）
    )

    def __init__(self):
        self.signal_line = self.data.signal

        # 当前方向：1 = 多头，-1 = 空头，0 = 无方向
        self.streak_dir = 0
        # 当前方向下连续信号次数（也等价于目标持仓绝对值，最多 = max_units）
        self.streak_count = 0

        # 记录权益曲线
        self.values = []
        self.dates = []

    def next(self):
        # 记录当前账户价值
        self.values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))

        sig_raw = self.signal_line[0]
        if np.isnan(sig_raw):
            return

        sig = int(round(sig_raw))
        if sig == 0:
            # 0 表示 hold：既不加仓也不减仓
            return

        changed = False

        if self.streak_dir == 0:
            # 之前没有方向，现在第一次出现方向信号
            self.streak_dir = sig
            self.streak_count = 1
            changed = True
        else:
            if sig == self.streak_dir:
                # 方向不变 → 继续“累加手数”（未达到上限前）
                if self.streak_count < self.params.max_units:
                    self.streak_count += 1
                    changed = True
                # 达到 max_units 后就不再加仓
            else:
                # 方向反转 → 目标方向改为 sig，从 1 个单位开始
                self.streak_dir = sig
                self.streak_count = 1
                changed = True

        if changed:
            # 目标持仓数量：±1, ±2, ..., ±max_units
            target_units = self.streak_dir * min(self.streak_count, self.params.max_units)
            # 这里用“数量”而不是百分比，表示买 / 卖多少个币
            self.order_target_size(target=target_units)

    def stop(self):
        # 最后一根再记录一次权益（保证 equity 最后一个点是最终价值）
        self.values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))


class SharpeBacktester:
    """
    供外部 manager 调用的简单封装：

    - 外部准备好 price_df（包含：open/high/low/close/volume/signal）
    - 调用 run(price_df) → 返回一个指标 dict（至少包含 sharpe_ratio / total_return）

    这里我们内部用一个“大本金”跑回测，确保能买得起 1 个币；
    然后把资金曲线归一化为“初始 = 1”，只看百分比收益，与真实本金大小无关。
    """

    def __init__(
        self,
        initial_cash: float = 1.0,
        commission: float = 0.0,
        periods_per_year: int = 252,
        cash_scale: float = 100000.0,
    ):
        """
        initial_cash: 对外展示的初始资金（概念上用 1 即可）
        commission: 手续费比例（默认 0）
        periods_per_year: 年化换算用的周期数（1d bar 通常用 252）
        cash_scale: 实际用于回测的资金放大倍数
                    实际回测本金 = initial_cash * cash_scale
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.periods_per_year = periods_per_year
        self.cash_scale = cash_scale

    @staticmethod
    def _compute_metrics(equity: pd.Series, periods_per_year: int) -> dict:
        """
        根据权益曲线计算 Sharpe 和一些简单指标。
        equity: pd.Series, index=DatetimeIndex, values=账户价值（已经是归一化过的）
        """
        equity = equity.dropna()
        if len(equity) < 2:
            return {
                "initial_value": float(equity.iloc[0]) if len(equity) == 1 else 1.0,
                "final_value": float(equity.iloc[-1]) if len(equity) >= 1 else 1.0,
                "total_return": 0.0,
                "sharpe_ratio": np.nan,
                "max_drawdown": 0.0,
            }

        rets = equity.pct_change().fillna(0.0)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

        # Sharpe（无风险利率假设为 0）
        std = rets.std()
        if std > 0:
            sharpe = float(rets.mean() / std * np.sqrt(periods_per_year))
        else:
            sharpe = np.nan

        # 最大回撤
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min())

        return {
            "initial_value": float(equity.iloc[0]),
            "final_value": float(equity.iloc[-1]),
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    def run(self, price_df: pd.DataFrame) -> dict:
        """
        运行一次回测，返回一个 dict：
          {
            'initial_value': ...,
            'final_value': ...,
            'total_return': ...,
            'sharpe_ratio': ...,
            'max_drawdown': ...
          }

        对外你可以理解为“初始资金 = initial_cash（比如 1），只看收益百分比和 Sharpe”。
        """

        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df.index 必须是 DatetimeIndex（请先 set_index(datetime)）")

        required = {"open", "high", "low", "close", "volume", "signal"}
        missing = required - set(price_df.columns)
        if missing:
            raise ValueError(f"price_df 缺少必要列: {missing}")

        price_df = price_df.sort_index()

        cerebro = bt.Cerebro()

        # 实际用于回测的本金（放大倍数，防止“买不起 1 个币”）
        real_initial_cash = self.initial_cash * self.cash_scale
        cerebro.broker.setcash(real_initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        data = _PandasDataWithSignal(dataname=price_df)
        cerebro.adddata(data)
        cerebro.addstrategy(_ScalingSignalStrategy)

        # 跑一遍策略
        strats = cerebro.run()
        strat = strats[0]

        # 真实资金曲线（比如以 USDT 为单位）
        equity_raw = pd.Series(strat.values, index=pd.DatetimeIndex(strat.dates)).sort_index()

        if equity_raw.empty:
            # 极端情况：没有任何 next 被调用
            equity = pd.Series([self.initial_cash], index=[price_df.index[0]])
        else:
            # 归一化到“初始 = initial_cash”，只看百分比收益
            equity = equity_raw / equity_raw.iloc[0] * self.initial_cash

        metrics = self._compute_metrics(equity, self.periods_per_year)

        # initial_value / final_value 用归一化后的值
        metrics["initial_value"] = float(equity.iloc[0])
        metrics["final_value"] = float(equity.iloc[-1])

        return metrics
