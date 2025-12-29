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


def different_coins_lookahead():
    """示例：分析不同的币种和不同的lookahead_periods"""
    coins = ["btc", "sol", "eth"]
    timeframes = ["1h"]
    lookahead_periods_list = [1, 2, 5, 10, 20, 50]
    model_type = "lightgbm"
    
    for coin in coins:
        for timeframe in timeframes:
            print(f"\n{'='*80}")
            print(f"Processing {coin.upper()} with {timeframe} timeframe")
            print(f"{'='*80}\n")
            
            # 第一个 lookahead_periods 需要生成特征（特征文件对所有 lookahead_periods 都相同）
            first_iteration = True
            
            for lookahead_periods in lookahead_periods_list:
                print(f"\n{'-'*80}")
                print(f"Processing {coin.upper()} {timeframe} with lookahead_periods={lookahead_periods}")
                print(f"{'-'*80}\n")

                # 定义映射关系：key 是 lookahead_periods，value 是 (threshold, delta) 的元组
                config_map = {
                    1:  (0.002, 0.0005),
                    2:  (0.004, 0.001),
                    5:  (0.008, 0.0002),
                    10: (0.015, 0.0004),
                    20: (0.03, 0.0008),
                    50: (0.05, 0.001),
                }

                # 使用 .get() 方法获取值
                # 如果 lookahead_periods 不在字典里，可以返回 None 或者设定的默认值
                result = config_map.get(lookahead_periods)

                if result:
                    signal_threshold, signal_delta = result
                else:
                    # 处理默认情况 (default case)
                    print(f"未知的周期: {lookahead_periods}")
                    signal_threshold, signal_delta = 0.0, 0.0 # 举例
                
                # 为每个 lookahead_periods 创建新的 AnalysisManager
                # 因为不同的 lookahead_periods 会生成不同的 processed 文件和模型文件
                manager = AnalysisManager(
                    coin=coin,
                    timeframe=timeframe,
                    signal_lookahead_periods=lookahead_periods,
                    signal_threshold=signal_threshold,
                    signal_delta=signal_delta,
                )
                
                # 执行完整流程
                manager.run_full_pipeline(
                    model_type=model_type,
                    # 特征文件对所有 lookahead_periods 都相同，所以第一个生成后，后续都跳过
                    skip_feature_gen=True,
                    # 每个 lookahead_periods 都会生成不同的信号文件，所以第一个不跳过，后续可以跳过（如果已存在）
                    skip_signal_gen=True,
                    # 每个 lookahead_periods 都会训练不同的模型，所以第一个不跳过，后续可以跳过（如果已存在）
                    skip_training=True,
                    # 是否执行回测
                    run_backtest=True,
                    # 传入 lookahead_periods（虽然已经在初始化时设置了，但这里显式传入更清晰）
                    lookahead_periods=lookahead_periods,
                    # 是否更新数据（第一个时更新，后续可以跳过）
                    update_data=True,
                )
                
                print(f"\n✓ Completed {coin.upper()} {timeframe} lookahead_periods={lookahead_periods}\n")
            
            print(f"\n{'='*80}")
            print(f"✓ All lookahead_periods completed for {coin.upper()} {timeframe}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    # 选择要运行的示例
    # example_full_pipeline()
    # example_custom_pipeline()
    # example_step_by_step()
    # example_with_custom_params()
    different_coins_lookahead()


