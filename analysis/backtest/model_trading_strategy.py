# backtest/model_trading_strategy.py
import backtrader as bt
import pandas as pd

# 按你的实际文件结构改这里的 import 路径和类名
from models.lightgbm_model import LightGBMModel
from models.svm_model import SVMModel
from models.random_forest_model import RandomForestModel
from models.lstm_model import NNModel
from models.logistic_model import LogisticModel
from models.transformer_model import TransformerNNModel


class ModelTradingStrategy(bt.Strategy):
    params = dict(
        # 使用哪个模型：'lightgbm' | 'svm' | 'random_forest' | 'nn' | 'transformer'
        model_type="lightgbm",

        # 每种模型的路径可以在 run_backtest 里覆盖
        model_paths={
            "lightgbm": "models/btc/1h_lightgbm.pkl",
            "logistic": "models/btc/1h_logistic.pkl",
            "svm": "models/btc/1h_svm.pkl",
            "random_forest": "models/btc/1h_random_forest.pkl",
            "nn": "models/btc/1h_lstm.pkl",   # 假设 nn 用的是 .pt / .pth
            "transformer": "models/btc/1h_transformer.pkl",
        },

        feature_cols=(),   # 由 run_backtest 传入，保持和训练一致的列顺序
        warmup_bars=30,    # 前几根K线不交易，留一点缓冲
    )

    def __init__(self):
        # 1) 加载模型（根据 model_type 自动选择）
        self.model = self._load_model()

        # 2) 特征列顺序，优先用传进来的；如果没传则用模型里保存的
        if self.p.feature_cols:
            self.feature_cols = list(self.p.feature_cols)
        else:
            # 要求每种模型类里都实现 .feature_cols 属性
            self.feature_cols = list(self.model.feature_cols)

        # 当前持仓方向：-1 = 做空, 0 = 空仓, 1 = 做多
        self.current_dir = 0
        # 当前方向已连续加仓次数（1 次 = 20% 仓位），最多 5 次
        self.steps_in_dir = 0
        self.order = None

    # ========= 核心改动：统一的模型加载入口 =========
    def _load_model(self):
        """
        根据 params.model_type 选择对应的模型类，并从对应路径加载。
        要求每个模型类都实现:
          - 类方法: load(path) -> model_instance
          - 方法:   predict(df) -> np.ndarray / list
          - 属性:   feature_cols (list[str])
        """
        model_type = self.p.model_type.lower()
        model_paths = self.p.model_paths

        if model_type not in model_paths:
            raise ValueError(f"Unknown model_type={model_type}, "
                             f"available={list(model_paths.keys())}")

        model_path = model_paths[model_type]

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
        elif model_type == "transformer":
            model = TransformerNNModel.load(model_path)
        else:
            raise ValueError(f"Unsupported model_type={model_type}")

        return model

    # ========= 以下逻辑基本不变，只是从 lgb_model -> 通用 model =========
    def _get_features(self):
        """
        从 data 里按 feature_cols 顺序取当前 bar 的特征值，拼成列表
        特征列在 run_backtest 的 PandasDataEx 里已经映射好了
        """
        values = []
        for col in self.feature_cols:
            line = getattr(self.data, col)
            values.append(float(line[0]))
        return values

    def _target_percent_for_state(self) -> float:
        """
        根据当前方向 & 步数，计算目标仓位：
        每步 20%，最多 5 步 → -1.0 ~ 1.0
        """
        return 0.2 * self.steps_in_dir * self.current_dir

    def next(self):
        # 预热阶段不交易
        if len(self.data) < self.p.warmup_bars:
            return

        # 1) 取当前bar特征，喂给模型
        features = self._get_features()
        feat_df = pd.DataFrame([features], columns=self.feature_cols)
        # 这里约定：所有模型的 predict 都返回 {-1, 0, 1} 的标签
        signal = int(self.model.predict(feat_df)[0])

        # 2) 根据信号调整仓位
        # 0 表示“保持现有仓位”，不加不减
        if signal == 0:
            return

        # 有 signal ∈ {1, -1}
        if self.current_dir == 0:
            # 当前空仓：按新方向开 20% 仓位
            self.current_dir = signal
            self.steps_in_dir = 1
        else:
            if signal == self.current_dir:
                # 连续同方向信号：每次加 20%，最多 5 次
                if self.steps_in_dir < 5:
                    self.steps_in_dir += 1
                # 如果已经 5 次就维持满仓，不再加
            elif signal == -self.current_dir:
                # 方向反转：立即平掉原方向，并反向开 20% 仓位
                self.current_dir = signal
                self.steps_in_dir = 1

        # 3) 计算目标仓位百分比，并下单
        target_pct = self._target_percent_for_state()
        self.order = self.order_target_percent(target=target_pct)

        # 调试可以打开：
        # dt = self.data.datetime.datetime(0)
        # print(f"{dt} signal={signal}, dir={self.current_dir}, "
        #       f"steps={self.steps_in_dir}, target={target_pct:.2f}")
