# workflow/example_usage.py
"""
AnalysisManager 使用示例
"""
from pathlib import Path
from workflow.analysis_manager import AnalysisManager


def example_full_pipeline():
    """示例：执行完整流程"""
    # 创建管理器实例（使用coin和timeframe自动生成路径）
    manager = AnalysisManager(
        coin="btc",           # 币种
        timeframe="1h",       # K线时长
        train_ratio=0.8,
    )
    
    # 执行完整流程（训练lightgbm模型）
    manager.run_full_pipeline(
        model_type="lightgbm",
        skip_feature_gen=True,  # 如果中间文件已存在则跳过
        skip_signal_gen=True,   # 如果处理后文件已存在则跳过
        skip_training=False,    # 强制重新训练
        run_backtest=True,      # 执行回测
        lookahead_periods=1,    # 使用初始化时的值，或传入具体数字如5
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
        coin="sol",                    # 使用SOL币种
        timeframe="1d",               # 使用1天K线
        signal_threshold=0.002,        # 自定义阈值
        signal_delta=0.0005,           # 自定义delta
        signal_lookahead_periods=5,   # 比较第5根K线（而不是下一根）
        train_ratio=0.75,              # 自定义训练集比例
    )
    
    manager.run_full_pipeline(
        model_type="random_forest",
        skip_feature_gen=False,  # 强制重新生成特征
        skip_signal_gen=False,   # 强制重新生成信号
        skip_training=False,     # 强制重新训练
        run_backtest=True,
        lookahead_periods=5,     # 比较第5根K线
    )


def example_different_coins():
    """示例：分析不同的币种"""
    coins = ["btc", "sol", "eth"]
    timeframes = ["1h", "1d"]
    
    for coin in coins:
        for timeframe in timeframes:
            print(f"\n{'='*60}")
            print(f"Processing {coin.upper()} with {timeframe} timeframe")
            print(f"{'='*60}\n")
            
            manager = AnalysisManager(
                coin=coin,
                timeframe=timeframe,
            )
            
            manager.run_full_pipeline(
                model_type="lightgbm",
                skip_feature_gen=True,
                skip_signal_gen=True,
                skip_training=True,
                run_backtest=False,  # 可以最后统一回测
            )


def example_with_dynamic_lookahead():
    """示例：动态修改lookahead_periods"""
    manager = AnalysisManager()
    
    # 先生成特征
    manager.run_feature_generation(skip_if_exists=True)
    
    # 使用不同的lookahead_periods生成多个信号文件
    for periods in [1, 3, 5, 10]:
        print(f"\n生成 lookahead_periods={periods} 的信号...")
        # 临时修改输出文件名
        original_path = manager.processed_path
        manager.processed_path = manager.processed_dir / f"btc_1h_features_signal_{periods}periods.csv"
        manager.signal_generator.out_path = manager.processed_path
        
        # 生成信号
        manager.run_signal_generation(
            skip_if_exists=False,
            lookahead_periods=periods,
        )
        
        # 恢复原始路径
        manager.processed_path = original_path
        manager.signal_generator.out_path = original_path


if __name__ == "__main__":
    # 选择要运行的示例
    example_full_pipeline()
    # example_custom_pipeline()
    # example_step_by_step()
    # example_with_custom_params()


