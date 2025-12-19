# train/train_lightgbm.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.lightgbm_model import LightGBMModel


class LightGBMTrainer:
    """
    LightGBM 模型训练器
    """
    
    def __init__(
        self,
        data_path: Path = Path("data/processed/btc_1h_features_signal.csv"),
        model_dir: Path = Path("models/btc"),
        model_path: Path = None,
        train_ratio: float = 0.8,
    ):
        """
        初始化训练器
        
        Args:
            data_path: 数据路径
            model_dir: 模型目录
            model_path: 模型保存路径，如果为None则使用默认路径
            train_ratio: 训练集比例
        """
        self.data_path = data_path
        self.model_dir = model_dir
        if model_path is None:
            self.model_path = model_dir / "1h_lightgbm.pkl"
        else:
            self.model_path = model_path
        self.train_ratio = train_ratio

    def load_data(self, train_ratio: float = None, use_strong_only: bool = True):
        if train_ratio is None:
            train_ratio = self.train_ratio
            
        print(f"[train_lgb] Reading data from {self.data_path} ...")
        df = pd.read_csv(self.data_path, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # 目标列
        y = df["signal"].astype(int)

        # 这些列不能当特征用：
        #  - 时间 & OHLCV
        #  - label
        #  - 用来构造 label 的未来收益 ret_next_1h
        #  - is_strong（仅用来做训练样本过滤）
        drop_cols = [
            "datetime",
            "signal",
            "open",
            "high",
            "low",
            "ret_next_1h",
            "is_strong",
        ]

        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols]

        n_total = len(df)
        n_train = int(n_total * train_ratio)

        print(f"[train_lgb] Total samples: {n_total}")
        print(f"[train_lgb] Train: {n_train}, Val: {n_total - n_train}")
        print(f"[train_lgb] Feature cols ({len(feature_cols)}): {feature_cols}")

        # 先按时间切分 train / val
        X_train = X.iloc[:n_train].reset_index(drop=True)
        X_val = X.iloc[n_train:].reset_index(drop=True)
        y_train = y.iloc[:n_train].reset_index(drop=True)
        y_val = y.iloc[n_train:].reset_index(drop=True)

        if use_strong_only:
            # 只在训练集上做去噪；验证集保持原始分布
            is_strong_train = df["is_strong"].iloc[:n_train].reset_index(drop=True)
            mask = is_strong_train == 1

            print(f"[train_lgb] Using strong-only train samples: {mask.sum()} / {len(mask)}")
            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        return X_train, X_val, y_train, y_val, feature_cols

    def train(self):
        """
        训练模型
        """
        X_train, X_val, y_train, y_val, feature_cols = self.load_data(self.train_ratio)

        # 初始化模型
        lgb_model = LightGBMModel()
        lgb_model.feature_cols = feature_cols
        lgb_model.train_ratio = self.train_ratio

        print("[train_lgb] Training LightGBM model ...")
        # 使用验证集做 early stopping
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            lgbm_params=None,  # 使用默认参数；你也可以在这里传一份 dict 进去
        )

        # ===== 在训练集上评估 =====
        print("\n[train_lgb] Evaluating on TRAIN set ...")
        train_report, train_cm = lgb_model.evaluate(X_train, y_train)

        print("\n[train_lgb] Classification report (train):")
        print(train_report)

        print("\n[train_lgb] Confusion matrix (train):")
        print(train_cm)

        # ===== 在验证集上评估 =====
        print("\n[val_lgb] Evaluating on VALIDATION set ...")
        val_report, val_cm = lgb_model.evaluate(X_val, y_val)

        print("\n[val_lgb] Classification report (val):")
        print(val_report)

        print("\n[val_lgb] Confusion matrix (val):")
        print(val_cm)

        # 保存模型
        self.model_dir.mkdir(parents=True, exist_ok=True)
        lgb_model.save(self.model_path)
        print(f"\n[train_lgb] Model saved to {self.model_path}")


# 为了保持向后兼容，提供函数接口
def load_data(train_ratio: float = 0.8, use_strong_only: bool = True):
    """
    向后兼容的函数接口
    """
    trainer = LightGBMTrainer(train_ratio=train_ratio)
    return trainer.load_data(train_ratio, use_strong_only)


def main():
    """
    向后兼容的函数接口
    """
    trainer = LightGBMTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
