# backtest/run_backtest_model.py
from pathlib import Path

import backtrader as bt
import pandas as pd

from models.lightgbm_model import LightGBMModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.lstm_model import NNModel
from models.logistic_model import LogisticModel

from .model_trading_strategy import ModelTradingStrategy

VAL_PATH = Path("data/processed/btc_1h_features_signal.csv")

# 不同模型各自的路径（按实际文件名改）
MODEL_PATHS = {
    "logistic": Path("models/btc/1h_logistic.pkl"),
    "lightgbm": Path("models/btc/1h_lightgbm.pkl"),
    "svm": Path("models/btc/1h_svm.pkl"),
    "random_forest": Path("models/btc/1h_random_forest.pkl"),
    "nn": Path("models/btc/1h_lstm.pkl"),
}


def create_pandasdata_with_features(feature_cols):
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


def _load_model_for_type(model_type: str):
    """
    根据 model_type 加载对应的模型实例，并返回 (model, model_path)。

    要求每个模型类都有:
      - classmethod load(path)
      - 属性 feature_cols
      - 可选属性 train_ratio
    """
    model_type = model_type.lower()
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model_type={model_type}, available={list(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[model_type]

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


def run_backtest(model_type: str = "lightgbm"):
    # ==== 1) 读完整数据 ====
    df = pd.read_csv(VAL_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # ==== 2) 加载对应模型，拿 feature_cols 和 train_ratio ====
    print(f"[backtest] Using model_type = {model_type}")
    model, model_path = _load_model_for_type(model_type)
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
    FeatureData = create_pandasdata_with_features(feature_cols)

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
    # 注意：MODEL_PATHS 里是 Path，这里转成 str 传给策略
    model_paths_str = {k: str(v) for k, v in MODEL_PATHS.items()}

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


if __name__ == "__main__":
    # 这里可以切换不同模型做回测logistic
    run_backtest("logistic")
    #run_backtest("lightgbm")
    #run_backtest("svm")
    #run_backtest("random_forest")
    #run_backtest("nn")
