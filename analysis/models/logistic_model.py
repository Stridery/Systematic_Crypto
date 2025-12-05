# models/logistic_model.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticModel:
    """
    封装多分类逻辑回归，用于 -1 / 0 / 1 三分类。

    属性:
      - pipeline: sklearn Pipeline(scaler + logistic)
      - feature_cols: 训练时使用的特征列名
      - train_ratio: 训练集比例（方便回测切分）
    """

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ):
        if pipeline is None:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            multi_class="multinomial",
                            max_iter=1000,
                            class_weight=None,
                        ),
                    ),
                ]
            )
        self.pipeline = pipeline
        self.feature_cols = feature_cols or []
        self.train_ratio = float(train_ratio)

    # =============== 训练 & 预测 ===============

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        训练逻辑回归模型。
        这里假设 X 的列顺序已经按 feature_cols 排好。
        """
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回预测的类别（-1 / 0 / 1），前提是训练时 y 就是这些取值。
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回 (N, num_classes) 的概率分布。
        """
        return self.pipeline.predict_proba(X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        digits: int = 4,
    ) -> Tuple[str, np.ndarray]:
        """
        返回 (classification_report_str, confusion_matrix_array)
        """
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred, digits=digits)
        cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
        return report, cm

    # =============== 保存 & 加载 ===============

    def save(self, path: str | Path) -> None:
        """
        把整个 LogisticModel 实例 pickle 下来。
        """
        path = Path(path)
        obj = {
            "pipeline": self.pipeline,
            "feature_cols": self.feature_cols,
            "train_ratio": self.train_ratio,
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticModel":
        """
        从 .pkl 文件恢复 LogisticModel 实例。
        """
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)

        model = cls(
            pipeline=obj["pipeline"],
            feature_cols=obj.get("feature_cols", []),
            train_ratio=obj.get("train_ratio", 0.8),
        )
        return model
