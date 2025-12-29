# train/train_lstm.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..models.lstm_model import LSTMClassifier, NNModel


LABEL_COL = "signal"        # 分类标签：-1 / 0 / 1
IS_STRONG_COL = "is_strong"  # 和 LightGBM 一样，用它做强样本筛选


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMTrainer:
    """
    LSTM 模型训练器
    """
    
    def __init__(
        self,
        data_path: Path = Path("data/processed/btc_1h_features_signal.csv"),
        model_dir: Path = Path("models/btc"),
        model_pkl_path: Path = None,
        train_ratio: float = 0.8,
        timeframe: str = "1h",
        lookahead_periods: int = 1,
    ):
        """
        初始化训练器
        
        Args:
            data_path: 数据路径
            model_dir: 模型目录
            model_pkl_path: 模型保存路径，如果为None则使用默认路径（基于timeframe和lookahead_periods）
            train_ratio: 训练集比例
            timeframe: K线时长，如 "1h", "1d", "1min"，用于生成模型文件名
            lookahead_periods: 信号生成的lookahead周期数，用于生成模型文件名
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.timeframe = timeframe
        self.lookahead_periods = lookahead_periods
        if model_pkl_path is None:
            # 模型文件名格式：{timeframe}_p{lookahead_periods}_{model}.pkl
            self.model_pkl_path = model_dir / f"{timeframe}_p{lookahead_periods}_lstm.pkl"
        else:
            self.model_pkl_path = model_pkl_path
        self.train_ratio = train_ratio

    def evaluate_split(
        self,
        model: nn.Module,
        loader: DataLoader,
        idx_to_label: dict[int, int],
        split_name: str,
        device: torch.device,
    ):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # 映射回原始标签 -1/0/1
        y_true_lbl = np.array([idx_to_label[int(i)] for i in all_targets])
        y_pred_lbl = np.array([idx_to_label[int(i)] for i in all_preds])

        print(f"\n[{split_name}] Classification report (original labels -1/0/1):")
        print(classification_report(y_true_lbl, y_pred_lbl, digits=4))

        cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=[-1, 0, 1])
        print(f"[{split_name}] Confusion matrix (rows=true, cols=pred, labels=-1/0/1):")
        print(cm)

    def train(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        batch_size: int = 128,
        num_epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        es_patience: int = 8,         # early stopping 容忍的 epoch 数
    ):
        # ---- 1. 读数据 & 按时间排序（和 LightGBM 一样）----
        print(f"[train_lstm] Reading data from {self.data_path} ...")
        df = pd.read_csv(self.data_path, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # ---- 2. 构造特征列（照抄 train_lightgbm 的 drop_cols）----
        drop_cols = [
            "datetime",
            "signal",
            "open",
            "high",
            "low",
            "ret_next_1h",
            "is_strong",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_cols: List[str] = [c for c in df.columns if c not in drop_cols]
        print(f"[train_lstm] Feature cols ({len(feature_cols)}): {feature_cols}")

        # 标签（-1/0/1）
        y_all_raw = df[LABEL_COL].astype(int).values

        # ---- 3. 按时间切分 train / val（和 LightGBM 一样）----
        n_total = len(df)
        n_train = int(n_total * self.train_ratio)
        print(f"[train_lstm] Total samples: {n_total}")
        print(f"[train_lstm] Train: {n_train}, Val: {n_total - n_train}")

        X_all = df[feature_cols]
        X_train_full = X_all.iloc[:n_train].reset_index(drop=True)
        X_val = X_all.iloc[n_train:].reset_index(drop=True)

        y_train_full_raw = y_all_raw[:n_train]
        y_val_raw = y_all_raw[n_train:]

        # ---- 4. 训练集只用 is_strong == 1 的样本（和 LightGBM 一样）----
        if IS_STRONG_COL in df.columns:
            is_strong_train = df[IS_STRONG_COL].iloc[:n_train].reset_index(drop=True)
            mask = is_strong_train == 1
            strong_count = int(mask.sum())
            print(f"[train_lstm] Using strong-only train samples: {strong_count} / {len(mask)}")

            if strong_count == 0:
                print("[train_lstm] WARNING: is_strong==1 has 0 samples in train range, using ALL train rows")
                X_train = X_train_full
                y_train_raw = y_train_full_raw
            else:
                X_train = X_train_full[mask].reset_index(drop=True)
                y_train_raw = y_train_full_raw[mask.values]
        else:
            print("[train_lstm] WARNING: no is_strong column, using ALL train rows")
            X_train = X_train_full
            y_train_raw = y_train_full_raw

        # ---- 5. 标签映射 (-1/0/1 -> 0/1/2) ----
        label_to_idx = {-1: 0, 0: 1, 1: 2}
        idx_to_label = {v: k for k, v in label_to_idx.items()}

        y_train = np.array([label_to_idx[int(v)] for v in y_train_raw], dtype=np.int64)
        y_val = np.array([label_to_idx[int(v)] for v in y_val_raw], dtype=np.int64)

        # ---- 6. 标准化（只用 train 拟合）----
        scaler = StandardScaler()
        scaler.fit(X_train.values)

        X_train_scaled = scaler.transform(X_train.values)
        X_val_scaled = scaler.transform(X_val.values)

        # ---- 7. DataLoader / 模型 / 优化器 ----
        train_ds = TabularDataset(X_train_scaled, y_train)
        val_ds = TabularDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train_scaled.shape[1]
        num_classes = len(np.unique(y_train))

        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        ).to(device)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train,
        )
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 根据 val_loss 动态调 lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,     # lr *= 0.5
            patience=3,     # 连续 3 个 epoch 没下降就降 lr
            min_lr=1e-5,
        )

        print(f"[train_lstm] input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")
        print(f"[train_lstm] class_weights={class_weights}")
        print(f"[train_lstm] initial lr={lr}, es_patience={es_patience}")

        # ---- 8. 训练循环（动态 lr + early stopping）----
        best_val_loss = float("inf")
        best_state_dict = None
        bad_epochs = 0

        for epoch in range(1, num_epochs + 1):
            # ---------- train ----------
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

            avg_train_loss = total_loss / len(train_ds)

            # ---------- validate ----------
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)

                    preds = logits.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())

            avg_val_loss = val_loss / len(val_ds)
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            val_acc = (all_preds == all_targets).mean()

            # 当前 lr
            prev_lr = optimizer.param_groups[0]["lr"]

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={prev_lr:.6f}"
            )

            # ---------- 调 lr ----------
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"[train_lstm] LR reduced: {prev_lr:.6f} -> {new_lr:.6f}")

            # ---------- early stopping ----------
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                best_state_dict = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= es_patience:
                    print(f"[train_lstm] Early stopping triggered at epoch {epoch}")
                    break

        # ---- 9. 用最优参数，做最终评估 ----
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        print("\n========== Final evaluation ==========")

        # 评估时的 DataLoader（train 用不 shuffle 的）
        train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        self.evaluate_split(model, train_loader_eval, idx_to_label, "TRAIN", device)
        self.evaluate_split(model, val_loader, idx_to_label, "VAL", device)

        # ---- 10. 打包成 NNModel 并保存 .pkl ----
        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_cpu = model.to("cpu")
        nn_model = NNModel(
            model=model_cpu,
            feature_cols=feature_cols,
            train_ratio=self.train_ratio,
            scaler=scaler,
            label_mapping=idx_to_label,
            device=torch.device("cpu"),
        )
        nn_model.save(self.model_pkl_path)

        print(f"\n[train_lstm] LSTM NNModel saved to {self.model_pkl_path}")
        print(f"[train_lstm] feature_dim={input_dim}, classes={num_classes}")


# 为了保持向后兼容，提供函数接口
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    idx_to_label: dict[int, int],
    split_name: str,
    device: torch.device,
):
    """
    向后兼容的函数接口
    """
    trainer = LSTMTrainer()
    trainer.evaluate_split(model, loader, idx_to_label, split_name, device)


def train_lstm_model(
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 128,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    es_patience: int = 8,         # early stopping 容忍的 epoch 数
):
    """
    向后兼容的函数接口
    """
    trainer = LSTMTrainer(train_ratio=train_ratio)
    trainer.train(hidden_dim, num_layers, batch_size, num_epochs, lr, weight_decay, es_patience)


if __name__ == "__main__":
    trainer = LSTMTrainer()
    trainer.train()
