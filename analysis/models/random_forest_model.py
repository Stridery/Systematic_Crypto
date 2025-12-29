# models/random_forest_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class RandomForestModel:
    """
    封装一个三分类 RandomForest 模型，用于预测 signal ∈ {-1, 0, 1}
    - 不做特征缩放（树模型不需要）
    - 内部保存 feature_cols，方便之后加载模型做预测
    """
    model: Optional[RandomForestClassifier] = None
    feature_cols: Optional[List[str]] = None
    train_ratio: float = 0.8

    def fit(
        self,
        X_train,
        y_train,
        rf_params: Optional[Dict[str, Any]] = None,
        sample_weight=None,
    ):
        """
        训练随机森林模型
        rf_params 允许你自己传入超参数；如果不传就用一套默认值
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            rf_params: 随机森林超参数
            sample_weight: 样本权重（可选）
        """
        if rf_params is None:
            rf_params = {
                "n_estimators": 300,        # 300 棵树够用，先不动
                "max_depth": 7,             # 从 10 降到 7，单棵树别长太深
                "min_samples_leaf": 50,     # 每个叶子至少 50 个样本（原来是 10）
                "min_samples_split": 100,   # 至少 100 个样本才考虑再分裂（原来 20）
                "max_features": "sqrt",     # 保持不变，已经是比较稳的设置
                "max_samples": 0.6,         # 每棵树只用 60% 的样本，加大随机性（原来 0.8）

                "n_jobs": -1,
                "class_weight": None,       # 类别还比较均衡，先不用 balanced
                "random_state": 42,
            }

        self.model = RandomForestClassifier(**rf_params)
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
            raise RuntimeError("This RandomForest model does not support predict_proba.")
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
    def load(cls, path: str | Path) -> "RandomForestModel":
        """
        从 pkl 文件加载一个 RandomForestModel 实例
        """
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        inst = cls()
        inst.model = payload["model"]
        inst.feature_cols = payload.get("feature_cols")
        inst.train_ratio = payload.get("train_ratio", 0.8)
        return inst
