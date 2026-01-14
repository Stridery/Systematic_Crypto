# util/weight_utils.py
"""
权重工具函数，用于归一化价格变动权重
"""
import numpy as np


def normalize_weights_robust(price_changes, min_weight=0.1, max_weight=10.0, log_range=1000):
    """
    稳健的权重归一化方法（方法5）
    1. 使用分位数避免异常值
    2. 对数变换放大差异
    3. 线性缩放到目标范围
    
    Args:
        price_changes: 价格变动百分比数组（如 ret_next_lookahead）
        min_weight: 最小权重
        max_weight: 最大权重
    
    Returns:
        归一化后的权重数组
    """
    abs_changes = np.abs(price_changes)
    abs_changes = np.nan_to_num(abs_changes, nan=0.0)
    
    # 如果所有值都相同或为0，返回均匀权重
    if abs_changes.max() == abs_changes.min() or abs_changes.max() == 0:
        return np.ones_like(abs_changes)
    
    # 使用分位数裁剪异常值
    lower = np.quantile(abs_changes, 0.01)  # 下1%分位数
    upper = np.quantile(abs_changes, 0.99)  # 上99%分位数
    clipped = np.clip(abs_changes, lower, upper)
    
    # 对数变换（放大差异）
    log_changes = np.log1p(clipped * log_range)  # 乘以1000放大
    
    # 线性缩放到目标范围
    if log_changes.max() == log_changes.min():
        return np.ones_like(abs_changes)
    
    normalized = (log_changes - log_changes.min()) / (log_changes.max() - log_changes.min())
    weights = min_weight + normalized * (max_weight - min_weight)
    
    return weights

