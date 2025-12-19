# workflow/example_usage.py
"""
AnalysisManager 使用示例
"""
from pathlib import Path
from workflow.analysis_manager import AnalysisManager


def example_full_pipeline():
    """示例：执行完整流程"""
    # 创建管理器实例
    manager = AnalysisManager(
        raw_data_path=Path("data/raw/BTCUSDT_1h.csv"),
        intermediate_dir=Path("data/intermediate"),
        processed_dir=Path("data/processed"),
        model_dir=Path("models/btc"),
        train_ratio=0.8,
    )
    
    # 执行完整流程（训练lightgbm模型）
    manager.run_full_pipeline(
        model_type="lightgbm",
        skip_feature_gen=True,  # 如果中间文件已存在则跳过
        skip_signal_gen=True,   # 如果处理后文件已存在则跳过
        skip_training=False,    # 强制重新训练
        run_backtest=True,      # 执行回测
    )


def example_custom_pipeline():
    """示例：执行自定义流程"""
    manager = AnalysisManager()
    
    # 只执行特征生成和信号生成
    manager.run_custom_pipeline(
        steps=["feature", "signal"],
    )
    
    # 训练多个模型
    for model_type in ["lightgbm", "logistic", "svm"]:
        manager.run_model_training(model_type, skip_if_exists=True)
    
    # 回测所有模型
    for model_type in ["lightgbm", "logistic", "svm"]:
        manager.run_backtest(model_type)


def example_step_by_step():
    """示例：逐步执行"""
    manager = AnalysisManager()
    
    # Step 1: 生成特征
    manager.run_feature_generation(skip_if_exists=True)
    
    # Step 2: 生成信号
    manager.run_signal_generation(skip_if_exists=True)
    
    # Step 3: 训练模型
    manager.run_model_training("lightgbm", skip_if_exists=False)
    
    # Step 4: 执行回测
    manager.run_backtest("lightgbm")


def example_with_custom_params():
    """示例：使用自定义参数"""
    manager = AnalysisManager(
        raw_data_path=Path("data/raw/BTCUSDT_1h.csv"),
        intermediate_dir=Path("data/intermediate"),
        processed_dir=Path("data/processed"),
        feature_threshold=0.002,  # 自定义阈值
        feature_delta=0.0005,     # 自定义delta
        model_dir=Path("models/btc"),
        train_ratio=0.75,         # 自定义训练集比例
    )
    
    manager.run_full_pipeline(
        model_type="random_forest",
        skip_feature_gen=False,  # 强制重新生成特征
        skip_signal_gen=False,   # 强制重新生成信号
        skip_training=False,     # 强制重新训练
        run_backtest=True,
    )


if __name__ == "__main__":
    # 选择要运行的示例
    example_full_pipeline()
    # example_custom_pipeline()
    # example_step_by_step()
    # example_with_custom_params()

