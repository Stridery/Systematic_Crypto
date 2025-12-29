# models/lightgbm_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import pickle
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class LightGBMModel:
    """
    封装一个三分类 LightGBM 模型，用于预测 signal ∈ {-1, 0, 1}
    - 不做特征缩放（树模型不需要）
    - 内部保存 feature_cols，方便之后加载模型做预测
    """
    model: Optional[LGBMClassifier] = None
    feature_cols: Optional[List[str]] = None
    train_ratio: float = 0.8

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        lgbm_params: Optional[Dict[str, Any]] = None,
        sample_weight=None,
    ):
        """
        训练 LightGBM 模型
        lgbm_params 允许你自己传入超参数；如果不传就用一套默认值
        eval_set: [(X_val, y_val)]
        这里不再传 early_stopping_rounds，兼容老版本 lightgbm
        """
        if lgbm_params is None:
            lgbm_params = {
                "objective": "multiclass",
                "num_class": 3,
                "boosting_type": "gbdt",

                # --- 1) 整体训练节奏：小步多走，配合 early stopping ---
                "n_estimators": 500,       # 提高上限，用 early_stopping 控制实际棵树数
                "learning_rate": 0.03,     # 步子再小一点，减轻过拟合

                # --- 2) 单棵树别太复杂 ---
                "num_leaves": 16,          # 从 31 -> 16，树更“粗糙”，不容易记住噪声
                "max_depth": 4,            # 从 6 -> 4，强限制树深度

                # --- 3) 每个叶子至少要有足够样本 ---
                "min_child_samples": 200,  # 从 100 -> 200，叶子上的样本更“多”，减少细碎拟合

                # --- 4) Bagging + 特征采样，增加随机性 ---
                "subsample": 0.6,          # 行采样比例 0.7 -> 0.6
                "subsample_freq": 5,       # 每 5 轮做一次 bagging
                "colsample_bytree": 0.6,   # 列采样比例 0.7 -> 0.6

                # --- 5) 正则化再狠一点 ---
                "reg_lambda": 10.0,        # L2 从 5 -> 10
                "reg_alpha": 1.0,          # L1 从 0 -> 1，鼓励一些叶子权重变小
                "min_split_gain": 0.1,     # 从 0.05 -> 0.1，只有“收益很大”才分裂

                "n_jobs": -1,
                "random_state": 42,

                "class_weight": {
                    -1: 1.0,
                    0: 1.0,
                    1: 1.0,
                },
            }

        self.model = LGBMClassifier(**lgbm_params)

        if eval_set is not None:
            fit_params = {
                "eval_set": eval_set,              # [(X_train, y_train), (X_val, y_val)] 或 [(X_val, y_val)]
                "eval_metric": "multi_logloss",    # 多分类常用
                "callbacks": [
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50),  # 每 50 轮打印一次日志，可按需调
                ],
            }
            if sample_weight is not None:
                fit_params["sample_weight"] = sample_weight
            self.model.fit(X_train, y_train, **fit_params)
        else:
            if sample_weight is not None:
                self.model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first or load from file.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first or load from file.")
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("This LightGBM model does not support predict_proba.")
        return self.model.predict_proba(X)

    def evaluate(self, X_val, y_val, digits: int = 4):
        """
        在验证集上评估，返回 (report_str, confusion_matrix_array)
        """
        y_pred = self.predict(X_val)
        report = classification_report(y_val, y_pred, digits=digits)
        cm = confusion_matrix(y_val, y_pred)
        return report, cm

    def save(self, path: str | Path):
        """
        把模型和 feature_cols 一起存下来
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or load one first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "train_ratio": self.train_ratio,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMModel":
        """
        从 pkl 文件加载一个 LightGBMModel 实例
        """
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        inst = cls()
        inst.model = payload["model"]
        inst.feature_cols = payload.get("feature_cols")
        inst.train_ratio = payload.get("train_ratio", 0.8)
        return inst
