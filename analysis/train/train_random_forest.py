# train/train_random_forest.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.random_forest_model import RandomForestModel
from util.weight_utils import normalize_weights_robust

RET_NEXT_COL = "ret_next_lookahead"  # 未来收益列，用于计算权重


class RandomForestTrainer:
    """
    随机森林模型训练器
    """
    
    def __init__(
        self,
        data_path: Path = Path("data/processed/btc_1h_features_signal.csv"),
        model_dir: Path = Path("models/btc"),
        model_path: Path = None,
        train_ratio: float = 0.8,
        timeframe: str = "1h",
        lookahead_periods: int = 1,
        log_range: int = 1000
    ):
        """
        初始化训练器
        
        Args:
            data_path: 数据路径
            model_dir: 模型目录
            model_path: 模型保存路径，如果为None则使用默认路径（基于timeframe和lookahead_periods）
            train_ratio: 训练集比例
            timeframe: K线时长，如 "1h", "1d", "1min"，用于生成模型文件名
            lookahead_periods: 信号生成的lookahead周期数，用于生成模型文件名
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.timeframe = timeframe
        self.lookahead_periods = lookahead_periods
        if model_path is None:
            # 模型文件名格式：{timeframe}_p{lookahead_periods}_{model}.pkl
            self.model_path = model_dir / f"{timeframe}_p{lookahead_periods}_random_forest.pkl"
        else:
            self.model_path = model_path
        self.train_ratio = train_ratio

    def load_data(self, train_ratio: float = None, use_strong_only: bool = True):
        if train_ratio is None:
            train_ratio = self.train_ratio
            
        print(f"[train_rf] Reading data from {self.data_path} ...")
        df = pd.read_csv(self.data_path, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # 提取权重（ret_next_lookahead的绝对值）
        if RET_NEXT_COL not in df.columns:
            raise ValueError(f"Column '{RET_NEXT_COL}' not found in data. Please regenerate signal with updated make_signal.py")
        
        ret_next_all = df[RET_NEXT_COL].values
        weights_all = normalize_weights_robust(ret_next_all, min_weight=0.1, max_weight=10.0, log_range=self.log_range)
        print(f"[train_rf] Weight stats: min={weights_all.min():.4f}, max={weights_all.max():.4f}, mean={weights_all.mean():.4f}")

        # 目标列
        y = df["signal"].astype(int)

        # 这些列不能当特征用：
        #  - 时间 & OHLCV
        #  - label
        #  - 用来构造 label 的未来收益 ret_next_lookahead
        #  - is_strong（仅用来做训练样本过滤）
        drop_cols = [
            "datetime",
            "signal",
            "open",
            "high",
            "low",
            "ret_next_lookahead",
            "is_strong",
        ]

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols]

        n_total = len(df)
        n_train = int(n_total * train_ratio)

        print(f"[train_rf] Total samples: {n_total}")
        print(f"[train_rf] Train: {n_train}, Val: {n_total - n_train}")
        print(f"[train_rf] Feature cols ({len(feature_cols)}): {feature_cols}")

        # 先按时间切分 train / val
        X_train = X.iloc[:n_train].reset_index(drop=True)
        X_val = X.iloc[n_train:].reset_index(drop=True)
        y_train = y.iloc[:n_train].reset_index(drop=True)
        y_val = y.iloc[n_train:].reset_index(drop=True)
        
        # 权重也要切分
        weights_train_full = weights_all[:n_train]
        weights_val = weights_all[n_train:]

        if use_strong_only:
            # 只在训练集上做去噪；验证集保持原始分布
            is_strong_train = df["is_strong"].iloc[:n_train].reset_index(drop=True)
            mask = is_strong_train == 1

            print(f"[train_rf] Using strong-only train samples: {mask.sum()} / {len(mask)}")
            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)
            weights_train = weights_train_full[mask.values]
        else:
            weights_train = weights_train_full

        return X_train, X_val, y_train, y_val, feature_cols, weights_train

    def train(self):
        """
        训练模型
        """
        X_train, X_val, y_train, y_val, feature_cols, weights_train = self.load_data(self.train_ratio)

        # 初始化模型
        rf_model = RandomForestModel()
        rf_model.feature_cols = feature_cols
        rf_model.train_ratio = self.train_ratio

        print("[train_rf] Training RandomForest model ...")
        rf_model.fit(X_train, y_train, sample_weight=weights_train)

        # ===== 在训练集上评估 =====
        print("\n[train_rf] Evaluating on TRAIN set ...")
        train_report, train_cm = rf_model.evaluate(X_train, y_train)

        print("\n[train_rf] Classification report (train):")
        print(train_report)

        print("\n[train_rf] Confusion matrix (train):")
        print(train_cm)

        # ===== 在验证集上评估（原来就有的逻辑）=====
        print("\n[val_rf] Evaluating on VALIDATION set ...")
        val_report, val_cm = rf_model.evaluate(X_val, y_val)

        print("\n[val_rf] Classification report (val):")
        print(val_report)

        print("\n[val_rf] Confusion matrix (val):")
        print(val_cm)

        # 保存模型
        self.model_dir.mkdir(parents=True, exist_ok=True)
        rf_model.save(self.model_path)
        print(f"\n[train_rf] Model saved to {self.model_path}")


# 为了保持向后兼容，提供函数接口
def load_data(train_ratio: float = 0.8, use_strong_only: bool = True):
    """
    向后兼容的函数接口
    注意：现在返回6个值，包括权重
    """
    trainer = RandomForestTrainer(train_ratio=train_ratio)
    return trainer.load_data(train_ratio, use_strong_only)


def main():
    """
    向后兼容的函数接口
    """
    trainer = RandomForestTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
