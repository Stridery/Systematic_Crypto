import sys
import os
from pathlib import Path
import backtrader as bt
import pandas as pd
import matplotlib

# =========================================
# 关键：AWS/Linux服务器必须设置无头模式，否则报错
# =========================================
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from models.lightgbm_model import LightGBMModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.lstm_model import NNModel
from models.logistic_model import LogisticModel
from models.transformer_model import TransformerNNModel
from backtest.model_trading_strategy import ModelTradingStrategy

# ============================================================
# 1. 定义一个Analyzer：专门用于记录每一天的资金净值
# 这样就不需要修改你的 Strategy 文件了
# ============================================================
class NetValueAnalyzer(bt.Analyzer):
    def start(self):
        self.equity = []
        self.dates = []

    def next(self):
        # 记录当前时间
        self.dates.append(self.datas[0].datetime.datetime(0))
        # 记录当前账户总资产 (现金 + 持仓市值)
        self.equity.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return {
            'dates': self.dates,
            'equity': self.equity
        }

class BacktestRunner:
    def __init__(
        self,
        val_path: Path = Path("data/processed/btc_1h_p1_features_signal.csv"),
        model_paths: dict = None,
        coin: str = "btc",
        timeframe: str = "1h",
        output_dir: Path = None,
        lookahead_periods: int = 1,
    ):
        self.val_path = val_path
        self.coin = coin.lower()
        self.timeframe = timeframe
        self.lookahead_periods = lookahead_periods
        
        # 默认输出目录
        if output_dir is None:
            self.output_dir = Path("analysis/backtest/curves")
        else:
            self.output_dir = output_dir
        
        # 自动推断模型路径
        if model_paths is None:
            model_dir = Path(f"models/{self.coin}")
            self.model_paths = {
                "logistic": model_dir / f"{timeframe}_p{lookahead_periods}_logistic.pkl",
                "lightgbm": model_dir / f"{timeframe}_p{lookahead_periods}_lightgbm.pkl",
                "svm": model_dir / f"{timeframe}_p{lookahead_periods}_svm.pkl",
                "random_forest": model_dir / f"{timeframe}_p{lookahead_periods}_random_forest.pkl",
                "nn": model_dir / f"{timeframe}_p{lookahead_periods}_lstm.pkl",
                "transformer": model_dir / f"{timeframe}_p{lookahead_periods}_transformer.pkl",
            }
        else:
            self.model_paths = {k: Path(v) if not isinstance(v, Path) else v 
                               for k, v in model_paths.items()}

    def create_pandasdata_with_features(self, feature_cols):
        lines = tuple(feature_cols)
        params = tuple((c, -1) for c in feature_cols)
        FeatureData = type("FeaturePandasData", (bt.feeds.PandasData,), {"lines": lines, "params": params})
        return FeatureData

    def _load_model_for_type(self, model_type: str):
        model_type = model_type.lower()
        if model_type not in self.model_paths:
            raise ValueError(f"Unknown model_type={model_type}")
        model_path = self.model_paths[model_type]
        
        print(f"[Info] Loading {model_type} from {model_path}...")
        
        if model_type == "lightgbm": model = LightGBMModel.load(model_path)
        elif model_type == "svm": model = SVMModel.load(model_path)
        elif model_type == "random_forest": model = RandomForestModel.load(model_path)
        elif model_type == "nn": model = NNModel.load(model_path)
        elif model_type == "logistic": model = LogisticModel.load(model_path)
        elif model_type == "transformer": model = TransformerNNModel.load(model_path)
        else: raise ValueError(f"Unsupported model_type={model_type}")
        return model, model_path

    def run_backtest(self, model_type: str = "lightgbm"):
        # 1. 准备数据
        if not os.path.exists(self.val_path):
            print(f"[Error] Data file not found: {self.val_path}")
            return

        df = pd.read_csv(self.val_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # 2. 加载模型以获取特征列
        model, model_path = self._load_model_for_type(model_type)
        feature_cols = list(model.feature_cols)
        train_ratio = getattr(model, "train_ratio", 0.8)

        # 3. 切分数据 (只回测验证集)
        n_train = int(len(df) * train_ratio)
        df_bt = df.iloc[n_train:].reset_index(drop=True)
        print(f"[backtest] Running on {len(df_bt)} bars (Validation Set)")
        
        # 4. 设置 Backtrader 数据源
        FeatureData = self.create_pandasdata_with_features(feature_cols)
        data_kwargs = dict(
            dataname=df_bt, datetime="datetime", open="open", high="high", low="low", close="close", volume="volume", openinterest=None
        )
        for c in feature_cols: data_kwargs[c] = c
        data = FeatureData(**data_kwargs)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # 5. 加载策略
        model_paths_str = {k: str(v) for k, v in self.model_paths.items()}
        cerebro.addstrategy(
            ModelTradingStrategy,
            model_type=model_type,
            model_paths=model_paths_str,
            feature_cols=feature_cols,
        )

        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.0005)

        # 6. 添加分析器 (Analyzers)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        
        # === 核心：添加我们自定义的 NetValueAnalyzer，命名为 'nav' ===
        cerebro.addanalyzer(NetValueAnalyzer, _name='nav')

        print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        print("Final Portfolio Value: %.2f" % final_value)

        # 7. 打印基础指标
        try:
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
            dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown')
            print(f"Sharpe Ratio: {sharpe}")
            print(f"Max Drawdown: {dd:.2f}%")
        except:
            pass

        # 8. 绘制并保存资金曲线 (Equity Curve)
        self.plot_equity_curve(strat, model_type, final_value)

    def plot_equity_curve(self, strat, model_type, final_value):
        try:
            # 创建输出目录
            coin_output_dir = self.output_dir / self.coin
            coin_output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{self.timeframe}_p{self.lookahead_periods}_{model_type.lower()}.png"
            output_path = coin_output_dir / output_filename
            
            # === 从 Analyzer 获取数据 (而不是从 Strategy) ===
            nav_data = strat.analyzers.nav.get_analysis()
            equity_dates = nav_data['dates']
            equity_values = nav_data['equity']
            
            if not equity_values:
                print("[Warning] No equity data recorded.")
                return

            # 开始绘图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制曲线
            ax.plot(equity_dates, equity_values, linewidth=1.5, label='Equity', color='#1f77b4')
            
            # 填充颜色 (以10万初始资金为界，上方绿，下方红)
            initial_cash = 100000.0
            ax.fill_between(equity_dates, equity_values, initial_cash, 
                           where=(pd.Series(equity_values) >= initial_cash),
                           interpolate=True, color='green', alpha=0.1)
            ax.fill_between(equity_dates, equity_values, initial_cash, 
                           where=(pd.Series(equity_values) < initial_cash),
                           interpolate=True, color='red', alpha=0.1)

            # 绘制初始资金基准线
            ax.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.8, linewidth=1)

            # 标题
            ret_pct = ((final_value / initial_cash) - 1) * 100
            ax.set_title(f'Equity Curve: {self.coin.upper()} - {model_type.upper()}\n'
                         f'Return: {ret_pct:.2f}%', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Value (USDT)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 格式化日期轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # 保存
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"[Success] Equity curve saved to: {output_path}")

        except Exception as e:
            print(f"[Error] Failed to plot equity curve: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 你可以在这里修改要回测的模型类型
    runner = BacktestRunner()
    runner.run_backtest("lightgbm")