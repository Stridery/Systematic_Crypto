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
    print("\n" + "="*140)
    print(f"{'Short MA':<10} {'Long MA':<10} {'Monthly Return':<15} {'Win Rate':<12} {'Max Drawdown':<15} {'Sharpe Ratio':<13} {'Total Trades':<13} {'Avg Return':<12} {'Volatility':<12}")
    print("="*140)
    
    for result in all_results:
        print(f"{result['short_window']:<10} {result['long_window']:<10} "
              f"{result['monthly_return']:<15.2%} {result['win_rate']:<12.2%} "
              f"{result['max_drawdown']:<15.2%} {result['sharpe_ratio']:<13.2f} "
              f"{result['total_trades']:<13} {result['avg_return']:<12.4%} {result['volatility']:<12.4%}")
    
    print("="*140)


def run_grid_search(
    data: pd.DataFrame,
    short_min: int,
    short_max: int,
    long_min: int,
    long_max: int
) -> Dict:
    """
    Run grid search to find optimal parameters.
    
    Args:
        data: Price data DataFrame
        short_min: Minimum short MA window
        short_max: Maximum short MA window
        long_min: Minimum long MA window
        long_max: Maximum long MA window
    
    Returns:
        Dictionary containing:
        - all_results: List of all parameter combinations and their metrics
        - best_result: The best parameter combination based on monthly return
        - best_params: Best parameters dict
        - results: Results DataFrame for the best parameters
    """
    # Generate all parameter combinations
    combinations = generate_param_combinations(short_min, short_max, long_min, long_max)
    print(f"\n=== Starting Grid Search: Testing {len(combinations)} parameter combinations ===")
    
    all_results = []
    
    # Run strategy for each combination
    for short, long in combinations:
        print(f"Testing: Short MA={short}, Long MA={long}")
        results = calculate_btc_ma_signals(data, short_window=short, long_window=long, verbose=False)
        
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
            'total_trades': metrics['total_trades'],
            'avg_return': metrics['avg_return'],
            'volatility': metrics['volatility'],
            'results': results
        })
    
    # Print results table
    print_results_table(all_results)
    
    # Find best parameters based on monthly return
    best_result = max(all_results, key=lambda x: x['monthly_return'])
    
    print(f"\n=== Best Parameters Found ===")
    print(f"Short MA: {best_result['short_window']}")
    print(f"Long MA: {best_result['long_window']}")
    print(f"Monthly Return: {best_result['monthly_return']:.2%}")
    print(f"Win Rate: {best_result['win_rate']:.2%}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print("=============================\n")
    
    # Prepare response with best parameters
    best_params = {
        'short_window': best_result['short_window'],
        'long_window': best_result['long_window'],
        'monthlyReturn': f"{best_result['monthly_return']:.2%}"
    }
    
    # Prepare metrics dict
    metrics_dict = {
        'annual_return': best_result['monthly_return'] * 12,
        'win_rate': best_result['win_rate'],
        'max_drawdown': best_result['max_drawdown'],
        'sharpe_ratio': best_result['sharpe_ratio'],
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

