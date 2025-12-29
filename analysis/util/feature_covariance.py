import pandas as pd
import numpy as np
from pathlib import Path


class FeatureCovarianceAnalyzer:
    """
    特征协方差分析器，用于分析特征之间的相关性
    """
    
    def __init__(
        self,
        data_path: Path = Path("data/intermediate/btc_1h_features.csv"),
        threshold: float = 0.95,
    ):
        """
        初始化特征协方差分析器
        
        Args:
            data_path: 数据路径
            threshold: 高相关性阈值
        """
        self.data_path = data_path
        self.threshold = threshold

    def analyze(self):
        """
        分析特征相关性
        """
        df = pd.read_csv(self.data_path)

    # 删掉不属于特征的列
    drop_cols = ["datetime", "signal", "open", "high", "low", "close"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]

    print(f"Loaded {len(feature_cols)} features.")

    # === 相关性矩阵 ===
    corr = X.corr()

    # 保存完整矩阵
    #corr.to_csv("feature_correlation_matrix.csv", float_format="%.4f")
    #print("Saved feature_correlation_matrix.csv")

    # === 协方差矩阵 ===
    #cov = X.cov()
    #cov.to_csv("feature_covariance_matrix.csv", float_format="%.4f")
    #print("Saved feature_covariance_matrix.csv")

    # === 找高相关特征对 ===
    high_corr_pairs = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            fi = feature_cols[i]
            fj = feature_cols[j]
                if abs(corr.iloc[i, j]) > self.threshold:
                high_corr_pairs.append((fi, fj, corr.iloc[i, j]))

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: -abs(x[2]))

    # 输出前 50 个最高相关的特征对
        print(f"\n=== Highly correlated feature pairs (|corr| > {self.threshold}) ===")
    for fi, fj, val in high_corr_pairs[:50]:
        print(f"{fi}  <-->  {fj}   corr={val:.4f}")


# 为了保持向后兼容，提供函数接口
def main():
    """
    向后兼容的函数接口
    """
    analyzer = FeatureCovarianceAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
