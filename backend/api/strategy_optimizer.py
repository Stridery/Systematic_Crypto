"""
Strategy parameter optimization module using grid search
"""
import pandas as pd
from typing import List, Dict, Tuple
from .strategy import calculate_btc_ma_signals, calculate_performance_metrics


def generate_param_combinations(short_min: int, short_max: int, long_min: int, long_max: int) -> List[Tuple[int, int]]:
    """
    Generate all valid parameter combinations for grid search.
    
    Args:
        short_min: Minimum short MA window
        short_max: Maximum short MA window
        long_min: Minimum long MA window
        long_max: Maximum long MA window
    
    Returns:
        List of (short_window, long_window) tuples where long > short
    """
    combinations = []
    for short in range(short_min, short_max + 1):
        for long in range(long_min, long_max + 1):
            if long > short:  # Ensure long > short
                combinations.append((short, long))
    return combinations


def print_results_table(all_results: List[Dict]):
    """
    Print all results in a formatted table.
    
    Args:
        all_results: List of result dictionaries with metrics
    """
    # print("\n" + "="*140)
    # print(f"{'Short MA':<10} {'Long MA':<10} {'Monthly Return':<15} {'Win Rate':<12} {'Max Drawdown':<15} {'Sharpe Ratio':<13} {'Total Trades':<13} {'Avg Return':<12} {'Volatility':<12}")
    # print("="*140)
    
    # for result in all_results:
    #     print(f"{result['short_window']:<10} {result['long_window']:<10} "
    #           f"{result['monthly_return']:<15.2%} {result['win_rate']:<12.2%} "
    #           f"{result['max_drawdown']:<15.2%} {result['sharpe_ratio']:<13.2f} "
    #           f"{result['total_trades']:<13} {result['avg_return']:<12.4%} {result['volatility']:<12.4%}")
    
    # print("="*140)


def run_grid_search(
    data: pd.DataFrame,
    short_min: int,
    short_max: int,
    long_min: int,
    long_max: int,
    start_date=None,
    end_date=None
) -> Dict:
    """
    Run grid search to find optimal parameters.
    
    Args:
        data: Price data DataFrame
        short_min: Minimum short MA window
        short_max: Maximum short MA window
        long_min: Minimum long MA window
        long_max: Maximum long MA window
        start_date: Start date for trading (str 'YYYY-MM-DD' or datetime). If None, uses default logic.
        end_date: End date for trading (str 'YYYY-MM-DD' or datetime). If None, uses last date in data.
    
    Returns:
        Dictionary containing:
        - all_results: List of all parameter combinations and their metrics
        - best_result: The best parameter combination based on sharpe ratio
        - best_params: Best parameters dict
        - results: Results DataFrame for the best parameters
    """
    # Generate all parameter combinations
    combinations = generate_param_combinations(short_min, short_max, long_min, long_max)
    # print(f"\n=== Starting Grid Search: Testing {len(combinations)} parameter combinations ===")
    if start_date and end_date:
        # print(f"Date range: {start_date} to {end_date}")
        pass
    
    all_results = []
    
    # Run strategy for each combination
    for short, long in combinations:
        #print(f"Testing: Short MA={short}, Long MA={long}")
        results = calculate_btc_ma_signals(
            data, 
            short_window=short, 
            long_window=long,
            start_date=start_date,
            end_date=end_date,
            verbose=False,
            calculate_daily_equity=False  # Grid search时不计算每日净值，减轻计算量
        )
        
        # Calculate metrics
        df_with_signals = results.copy()
        df_with_signals.set_index('timestamp', inplace=True)
        metrics = calculate_performance_metrics(df_with_signals)
        
        all_results.append({
            'short_window': short,
            'long_window': long,
            'monthly_return': metrics['annual_return']/12,
            'win_rate': metrics['win_rate'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'profit_factor': metrics.get('profit_factor', 0.0),
            'total_trades': metrics['total_trades'],
            'avg_return': metrics['avg_return'],
            'volatility': metrics['volatility'],
            'results': results
        })
    
    # Print results table (保留注释，不打印每个参数对的详细结果表格)
    # print_results_table(all_results)
    
    # Find best parameters based on sharpe ratio (highest sharpe ratio)
    best_result = max(all_results, key=lambda x: x['sharpe_ratio'])
    
    # print(f"\n=== Best Parameters Found ===")
    # print(f"Short MA: {best_result['short_window']}")
    # print(f"Long MA: {best_result['long_window']}")
    # print(f"Monthly Return: {best_result['monthly_return']:.2%}")
    # print(f"Win Rate: {best_result['win_rate']:.2%}")
    # print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    # print("=============================\n")
    
    # Prepare response with best parameters
    best_params = {
        'short_window': best_result['short_window'],
        'long_window': best_result['long_window'],
        'monthlyReturn': f"{best_result['monthly_return']:.2%}"
    }
    
    # Prepare metrics dict
    profit_factor_value = best_result.get('profit_factor', 0.0)
    metrics_dict = {
        'annual_return': best_result['monthly_return'] * 12,
        'win_rate': best_result['win_rate'],
        'max_drawdown': best_result['max_drawdown'],
        'sharpe_ratio': best_result['sharpe_ratio'],
        'profit_factor': profit_factor_value,
        'total_trades': best_result['total_trades'],
        'avg_return': best_result['avg_return'],
        'volatility': best_result['volatility']
    }
    
    return {
        'all_results': all_results,
        'best_result': best_result,
        'best_params': best_params,
        'metrics': metrics_dict,
        'results': best_result['results']
    }


def run_training_validation(
    data: pd.DataFrame,
    short_min: int,
    short_max: int,
    long_min: int,
    long_max: int,
    train_start_date=None,
    train_end_date=None,
    validation_start_date=None,
    validation_end_date=None
) -> Dict:
    """
    Run training-validation workflow:
    1. Run grid search on training period to find best parameters
    2. Run validation on validation period with best parameters
    
    Args:
        data: Price data DataFrame
        short_min: Minimum short MA window
        short_max: Maximum short MA window
        long_min: Minimum long MA window
        long_max: Maximum long MA window
        train_start_date: Training period start date (str 'YYYY-MM-DD' or datetime)
        train_end_date: Training period end date (str 'YYYY-MM-DD' or datetime)
        validation_start_date: Validation period start date (str 'YYYY-MM-DD' or datetime)
        validation_end_date: Validation period end date (str 'YYYY-MM-DD' or datetime)
    
    Returns:
        Dictionary containing:
        - training_metrics: Training performance metrics
        - validation_metrics: Validation performance metrics (if validation dates provided)
        - best_params: Best parameters found during training
        - training_results: Training results DataFrame
        - validation_results: Validation results DataFrame (if validation dates provided)
    """
    # Step 1: Run grid search on training period
    # print(f"\n=== Training Phase ===")
    # print(f"Training Period: {train_start_date} to {train_end_date}")
    
    optimization_result = run_grid_search(
        data,
        short_min,
        short_max,
        long_min,
        long_max,
        start_date=train_start_date,
        end_date=train_end_date
    )
    
    # Extract training results
    training_results = optimization_result['results']
    training_metrics = optimization_result['metrics']
    best_params = optimization_result['best_params']
    
    # 用最佳参数在training period上再运行一次，输出详细的交易记录和每日净值（用于绘图）
    # print(f"\n=== Training Period Trade Details (Best Parameters) ===")
    # print(f"Using Best Parameters: Short MA={best_params['short_window']}, Long MA={best_params['long_window']}")
    training_results_with_details = calculate_btc_ma_signals(
        data,
        short_window=best_params['short_window'],
        long_window=best_params['long_window'],
        start_date=train_start_date,
        end_date=train_end_date,
        verbose=True,
        calculate_daily_equity=True  # 计算每日净值用于绘图
    )
    # 更新training_results为带详细输出的结果
    training_results = training_results_with_details
    
    # Step 2: Run validation if validation dates are provided
    validation_metrics = None
    validation_results = None
    
    if validation_start_date and validation_end_date:
        # print(f"\n=== Validation Phase ===")
        # print(f"Validation Period: {validation_start_date} to {validation_end_date}")
        # print(f"Using Best Parameters: Short MA={best_params['short_window']}, Long MA={best_params['long_window']}")
        
        # Run strategy on validation period with best parameters
        validation_results = calculate_btc_ma_signals(
            data,
            short_window=best_params['short_window'],
            long_window=best_params['long_window'],
            start_date=validation_start_date,
            end_date=validation_end_date,
            verbose=True,
            calculate_daily_equity=True  # 计算每日净值用于绘图
        )
        
        # Calculate validation metrics
        validation_df = pd.DataFrame(validation_results)
        validation_df.set_index('timestamp', inplace=True)
        validation_metrics = calculate_performance_metrics(validation_df, verbose=True)
        
        # print(f"\n=== Validation Performance Metrics ===")
        # print(f"Annual Return: {validation_metrics['annual_return']:.2%}")
        # print(f"Sharpe Ratio: {validation_metrics['sharpe_ratio']:.2f}")
        # print(f"Max Drawdown: {validation_metrics['max_drawdown']:.2%}")
        # print(f"Win Rate: {validation_metrics['win_rate']:.2%}")
        # print(f"Total Trades: {validation_metrics['total_trades']}")
    
    return {
        'training_metrics': training_metrics,
        'validation_metrics': validation_metrics,
        'best_params': best_params,
        'training_results': training_results,
        'validation_results': validation_results
    }

