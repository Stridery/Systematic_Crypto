# train/train_random_forest.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.random_forest_model import RandomForestModel


# ===== 参数区 =====
DATA_PATH = Path("data/processed/btc_1h_features_signal.csv")

MODEL_DIR = Path("models/btc")
MODEL_PATH = MODEL_DIR / "1h_random_forest.pkl"

TRAIN_RATIO = 0.8
# ==================


def load_data(train_ratio: float = TRAIN_RATIO, use_strong_only: bool = True):
    print(f"[train_lgb] Reading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
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


def main():
    X_train, X_val, y_train, y_val, feature_cols = load_data(TRAIN_RATIO)

    # 初始化模型
    rf_model = RandomForestModel()
    rf_model.feature_cols = feature_cols
    rf_model.train_ratio = TRAIN_RATIO

    print("[train_rf] Training RandomForest model ...")
    rf_model.fit(X_train, y_train)

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
    rf_model.save(MODEL_PATH)
    print(f"\n[train_rf] Model saved to {MODEL_PATH}")



if __name__ == "__main__":
    main()
