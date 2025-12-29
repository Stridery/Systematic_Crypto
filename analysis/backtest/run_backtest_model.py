# backtest/run_backtest_model.py
from pathlib import Path

import backtrader as bt
import pandas as pd

from ..models.lightgbm_model import LightGBMModel
from ..models.svm_model import SVMModel
from ..models.random_forest_model import RandomForestModel
from ..models.lstm_model import NNModel
from ..models.logistic_model import LogisticModel

from .model_trading_strategy import ModelTradingStrategy


class BacktestRunner:
    """
    封装回测运行器，用于执行模型回测
    """
    
    def __init__(
        self,
        val_path: Path = Path("data/processed/btc_1h_features_signal.csv"),
        model_paths: dict = None,
        coin: str = "btc",
        timeframe: str = "1h",
        output_dir: Path = None,
        lookahead_periods: int = 1,
    ):
        """
        初始化回测运行器
        
        Args:
            val_path: 验证数据路径
            model_paths: 模型路径字典，如果为None则根据coin、timeframe和lookahead_periods自动生成
            coin: 币种，如 "btc", "sol"（小写）
            timeframe: K线时长，如 "1h", "1d", "1m"
            output_dir: 输出目录，用于保存equity curve图片，如果为None则使用默认路径
            lookahead_periods: 信号生成的lookahead周期数，用于生成模型路径和图片文件名
        """
        self.val_path = val_path
        self.coin = coin.lower()
        self.timeframe = timeframe
        self.lookahead_periods = lookahead_periods
        
        # 如果未指定output_dir，使用默认路径：analysis/backtest/curves
        if output_dir is None:
            self.output_dir = Path("analysis/backtest/curves")
        else:
            self.output_dir = output_dir
        
        if model_paths is None:
            # 根据coin、timeframe和lookahead_periods自动生成模型路径
            # 模型文件名格式：{timeframe}_p{lookahead_periods}_{model}.pkl
            model_dir = Path(f"models/{self.coin}")
            self.model_paths = {
                "logistic": model_dir / f"{timeframe}_p{lookahead_periods}_logistic.pkl",
                "lightgbm": model_dir / f"{timeframe}_p{lookahead_periods}_lightgbm.pkl",
                "svm": model_dir / f"{timeframe}_p{lookahead_periods}_svm.pkl",
                "random_forest": model_dir / f"{timeframe}_p{lookahead_periods}_random_forest.pkl",
                "nn": model_dir / f"{timeframe}_p{lookahead_periods}_lstm.pkl",
            }
        else:
            self.model_paths = {k: Path(v) if not isinstance(v, Path) else v 
                               for k, v in model_paths.items()}

    def create_pandasdata_with_features(self, feature_cols):
        """
        动态创建一个带任意 feature 列的 PandasData 子类，
        这样这些列可以在 Strategy 里通过 self.data.<col> 访问。
        """
        lines = tuple(feature_cols)
        params = tuple((c, -1) for c in feature_cols)

        FeatureData = type(
            "FeaturePandasData",
            (bt.feeds.PandasData,),
            {
                "lines": lines,
                "params": params,
            },
        )
        return FeatureData

    def _load_model_for_type(self, model_type: str):
        """
        根据 model_type 加载对应的模型实例，并返回 (model, model_path)。

        要求每个模型类都有:
        - classmethod load(path)
        - 属性 feature_cols
        - 可选属性 train_ratio
        """
        model_type = model_type.lower()
        if model_type not in self.model_paths:
            raise ValueError(f"Unknown model_type={model_type}, available={list(self.model_paths.keys())}")

        model_path = self.model_paths[model_type]

        if model_type == "lightgbm":
            model = LightGBMModel.load(model_path)
        elif model_type == "svm":
            model = SVMModel.load(model_path)
        elif model_type == "random_forest":
            model = RandomForestModel.load(model_path)
        elif model_type == "nn":
            model = NNModel.load(model_path)
        elif model_type == "logistic":
            model = LogisticModel.load(model_path)
        else:
            raise ValueError(f"Unsupported model_type={model_type}")

        return model, model_path

    def run_backtest(self, model_type: str = "lightgbm"):
        # ==== 1) 读完整数据 ====
        df = pd.read_csv(self.val_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # ==== 2) 加载对应模型，拿 feature_cols 和 train_ratio ====
        print(f"[backtest] Coin: {self.coin.upper()}, Timeframe: {self.timeframe}")
        print(f"[backtest] Using model_type = {model_type}")
        model, model_path = self._load_model_for_type(model_type)
        feature_cols = list(model.feature_cols)
        train_ratio = getattr(model, "train_ratio", 0.8)

        print(f"[backtest] Loading model from {model_path} ...")
        print(f"[backtest] Using feature cols ({len(feature_cols)}): {feature_cols}")

        # ==== 3) 切出验证集部分来回测（后 20%） ====
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        df_bt = df.iloc[n_train:].reset_index(drop=True)
        print(f"[backtest] Total rows: {n_total}, using val rows: {len(df_bt)}")

        # ==== 4) 创建带 feature lines 的 PandasData 子类 ====
        FeatureData = self.create_pandasdata_with_features(feature_cols)

        # ==== 5) 构造 Backtrader 数据源 ====
        data_kwargs = dict(
            dataname=df_bt,
            datetime="datetime",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=None,
        )
        # 为每个特征添加映射：参数名=列名
        for c in feature_cols:
            data_kwargs[c] = c

        data = FeatureData(**data_kwargs)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # ==== 6) 加载策略，把模型类型、模型路径字典和特征列顺序传进去 ====
        # 注意：model_paths 里是 Path，这里转成 str 传给策略
        model_paths_str = {k: str(v) for k, v in self.model_paths.items()}

        cerebro.addstrategy(
            ModelTradingStrategy,
            model_type=model_type,
            model_paths=model_paths_str,
            feature_cols=feature_cols,
        )

        # 资金设置
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.0005)  # 示例：万5 手续费

        # ==== 7) 添加各种分析器 ====
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
        results = cerebro.run()
        strat = results[0]
        print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

        # ==== 8) 打印绩效指标 ====

        # SharpeRatio 分析器
        try:
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
            sharpe_ratio = sharpe_analysis.get("sharperatio", None)
            print(f"Sharpe Ratio (SharpeRatio analyzer): {sharpe_ratio}")
        except Exception as e:
            print(f"Sharpe Ratio (SharpeRatio analyzer) error: {e}")

        # DrawDown 分析器
        try:
            dd = strat.analyzers.drawdown.get_analysis()
            max_dd = dd.get("max", {}).get("drawdown", None)
            max_len = dd.get("max", {}).get("len", None)
            print(f"Max Drawdown: {max_dd:.2f}% over {max_len} bars" if max_dd is not None else "Max Drawdown: N/A")
        except Exception as e:
            print(f"Drawdown analyzer error: {e}")

        # Returns 分析器
        try:
            ret = strat.analyzers.returns.get_analysis()
            rnorm100 = ret.get("rnorm100", None)      # 年化收益（百分比）
            vnorm100 = ret.get("vnorm100", None)      # 年化波动（百分比）
            sharpe_ret = ret.get("sharperatio", None)
            if rnorm100 is not None:
                print(f"Annual Return: {rnorm100:.2f}%")
            else:
                print("Annual Return: N/A")
            if vnorm100 is not None:
                print(f"Annual Volatility: {vnorm100:.2f}%")
            else:
                print("Annual Volatility: N/A")
            print(f"Sharpe (Returns analyzer): {sharpe_ret}")
        except Exception as e:
            print(f"Returns analyzer error: {e}")

        # TradeAnalyzer 分析器
        try:
            ta = strat.analyzers.trades.get_analysis()
            total_closed = ta.get("total", {}).get("closed", 0)
            total_open = ta.get("total", {}).get("open", 0)
            total_trades = total_closed + total_open
            won_total = ta.get("won", {}).get("total", 0)
            lost_total = ta.get("lost", {}).get("total", 0)
            pnl_net = ta.get("pnl", {}).get("net", {})
            avg_pnl = pnl_net.get("average", None)

            print(f"Total Trades (closed): {total_closed}")
            print(f"Total Trades (including open): {total_trades}")
            if total_closed > 0:
                win_rate = won_total / total_closed
                print(f"Win Rate: {win_rate:.2%}")
            else:
                print("Win Rate: N/A")
            print(f"Average PnL per closed trade: {avg_pnl}" if avg_pnl is not None else "Average PnL per trade: N/A")
        except Exception as e:
            print(f"TradeAnalyzer error: {e}")

        # ==== 9) 使用backtrader绘制并保存equity curve ====
        try:
            # 创建币种子文件夹：analysis/backtest/curves/{coin}/
            coin_output_dir = self.output_dir / self.coin
            coin_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 图片命名格式：{timeframe}_p{lookahead_periods}_{model}.png
            output_filename = f"{self.timeframe}_p{self.lookahead_periods}_{model_type.lower()}.png"
            output_path = coin_output_dir / output_filename
            
            # 使用backtrader自带的plot功能
            # 设置非交互式后端以避免显示窗口
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            
            # backtrader的plot方法会创建图表
            # 使用style='bar'显示K线图，volume=False不显示成交量
            cerebro.plot(style='bar', barup='green', bardown='red', volume=False)
            
            # 获取当前figure并保存
            fig = plt.gcf()
            if fig is not None:
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"[backtest] Equity curve saved to: {output_path}")
            else:
                print("[backtest] Warning: No figure generated by backtrader plot")
                
        except Exception as e:
            print(f"[backtest] Error plotting equity curve: {e}")
            import traceback
            traceback.print_exc()


# 为了保持向后兼容，提供函数接口
def run_backtest(model_type: str = "lightgbm"):
    """
    向后兼容的函数接口
    """
    runner = BacktestRunner()
    runner.run_backtest(model_type)


if __name__ == "__main__":
    # 这里可以切换不同模型做回测logistic
    runner = BacktestRunner()
    runner.run_backtest("logistic")
    # runner.run_backtest("lightgbm")
    # runner.run_backtest("svm")
    # runner.run_backtest("random_forest")
    # runner.run_backtest("nn")
