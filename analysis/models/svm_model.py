# models/svm_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class SVMModel:
    """
    封装一个多分类 SVM 模型，用于预测 signal ∈ {-1, 0, 1}
    - 内部使用 Pipeline: StandardScaler + SVC
    - 支持 predict_proba（SVC 需要 probability=True）
    - 保存时会一起存 feature_cols 和 train_ratio
    """
    model: Optional[Pipeline] = None
    feature_cols: Optional[List[str]] = None
    train_ratio: float = 0.8

    def fit(
        self,
        X,
        y,
        svm_params: Optional[Dict[str, Any]] = None,
    ):
        """
        训练 SVM 模型
        svm_params 允许传入超参数；如果不传，就用一套默认值
        """
        if svm_params is None:
            svm_params = {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "class_weight": None,  # 三分类稍微不均衡时更稳一点
                "probability": True,         # 为了支持 predict_proba
                "random_state": 42,
            }

        svc = SVC(**svm_params)

        # 用 Pipeline 先标准化，再喂给 SVM
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", svc),
            ]
        )

        self.model.fit(X, y)

    def predict(self, X):
        """
        预测标签（-1 / 0 / 1）
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first or load from file.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        预测每一类的概率，形状 [n_samples, n_classes]
        类别顺序为 self.model.named_steps['svc'].classes_
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first or load from file.")
        svc = self.model.named_steps["svc"]
        if not hasattr(svc, "predict_proba"):
            raise RuntimeError("Underlying SVC does not support predict_proba (probability=False?).")
        return self.model.predict_proba(X)

    def evaluate(self, X, y, digits: int = 4):
        """
        在给定数据集上评估，返回 (report_str, confusion_matrix_array)
        """
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, digits=digits)
        cm = confusion_matrix(y, y_pred)
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

        print(f"[SVMModel] Saved model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVMModel":
        """
        从 pkl 文件加载一个 SVMModel 实例
        """
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        inst = cls()
        inst.model = payload["model"]
        inst.feature_cols = payload.get("feature_cols")
        inst.train_ratio = payload.get("train_ratio", 0.8)
        return inst
