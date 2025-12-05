# models/nn_model.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    简单 LSTM 多分类模型（这里 seq_len=1，只在特征维度上做 gating）

    输入:  x: (batch_size, feature_dim)
    输出:  logits: (batch_size, num_classes)

    说明:
    - 你已经有大量 lag 特征，短期时间信息已经编码在 feature 里，
      所以这里先用一个 seq_len=1 的 LSTM，后面如果要显式用 window 再升级。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feature_dim) -> (batch, seq_len=1, feature_dim)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim)
        last = out[:, -1, :]           # (batch, hidden_dim)
        logits = self.fc(last)         # (batch, num_classes)
        return logits


class NNModel:
    """
    封装好的 NN 模型接口，用来在训练后 dump 成 .pkl，
    回测或推理时直接 load + predict 即可。

    属性:
      - model: PyTorch 模型 (LSTMClassifier)
      - feature_cols: 使用的特征列名列表
      - train_ratio: 训练集比例（方便回测切 train/val 边界）
      - scaler: 标准化器 (如 sklearn StandardScaler)
      - label_mapping: 内部类别索引 -> 原始标签 (如 {0: -1, 1: 0, 2: 1})

    方法:
      - save(path): 保存整个 NNModel 为 .pkl
      - @classmethod load(path): 从 .pkl 载入 NNModel
      - predict_proba(df): (N, num_classes) 概率
      - predict(df): (N,) 原始标签 {-1,0,1}
    """
    def __init__(
        self,
        model: nn.Module,
        feature_cols: Sequence[str],
        train_ratio: float = 0.8,
        scaler: Optional[object] = None,
        label_mapping: Optional[Dict[int, int]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.train_ratio = float(train_ratio)
        self.scaler = scaler
        self.device = device or torch.device("cpu")

        # 内部 idx -> 原始标签
        # 如 {0: -1, 1: 0, 2: 1}
        if label_mapping is None:
            label_mapping = {0: -1, 1: 0, 2: 1}
        self.label_mapping = label_mapping

        self.model.to(self.device)
        self.model.eval()

    # ---------- I/O ----------
    def save(self, path: str | Path) -> None:
        """
        把整个 NNModel 实例 pickle 下来，后续 load 即可 predict。
        """
        import pickle

        path = Path(path)
        # 为了安全，存的时候先把模型搬到 CPU
        self.model.to("cpu")
        self.device = torch.device("cpu")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "NNModel":
        """
        从 .pkl 文件载入已训练好的 NNModel。
        """
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # 确保 device 设置好
        if not hasattr(obj, "device") or obj.device is None:
            obj.device = torch.device("cpu")

        obj.model.to(obj.device)
        obj.model.eval()
        return obj

    # ---------- 内部辅助 ----------
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    # ---------- 推理接口 ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        返回每一行样本属于各类别的概率，形状 (N, num_classes)
        """
        X = self._prepare_X(df)
        x_tensor = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)             # (N, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        输出原始标签空间里的预测：例如 {-1, 0, 1}
        """
        probs = self.predict_proba(df)
        cls_idx = probs.argmax(axis=1)   # 0/1/2
        mapping = self.label_mapping
        preds = np.array([mapping[int(i)] for i in cls_idx], dtype=int)
        return preds
