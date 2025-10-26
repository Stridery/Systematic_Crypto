from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

router = APIRouter()

# Database configuration
DB_NAME = "trading_db"
DB_USER = "zhenghaoyou"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432

def calculate_performance_metrics(data: pd.DataFrame) -> dict:
    """Calculate detailed performance metrics for the strategy"""
    # 计算年化收益率
    total_days = (data.index[-1] - data.index[0]).days
    total_return = data['cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (365 / total_days) - 1

    # 计算夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    returns_std = data['returns'].std() * np.sqrt(365)  # 加密货币全年无休，使用365天年化波动率
    sharpe_ratio = (annual_return - risk_free_rate) / returns_std if returns_std != 0 else 0

    # 计算最大回撤
    cumulative_max = data['cumulative_returns'].expanding().max()
    drawdowns = (data['cumulative_returns'] - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    # 计算交易统计
    # 找出交易信号变化点（开仓/平仓点）
    trades = data['signal'].diff().fillna(0)
    trade_points = trades != 0
    
    # 提取每笔交易的收益
    trade_returns = []
    position_start = None
    current_signal = 0
    cumulative_return = 1.0
    
    for i in range(len(data)):
        if trade_points.iloc[i]:
            if current_signal != 0:  # 平仓
                # 计算这笔交易的收益
                end_price = data['close'].iloc[i]
                start_price = data['close'].iloc[position_start]
                trade_return = (end_price - start_price) / start_price * current_signal
                trade_returns.append(trade_return)
                current_signal = 0
            else:  # 开仓
                position_start = i
                current_signal = data['signal'].iloc[i]
    
    # 计算交易统计
    total_trades = len(trade_returns)
    winning_trades = sum(1 for r in trade_returns if r > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 计算平均收益率和波动率（基于交易）
    avg_return = np.mean(trade_returns) if trade_returns else 0
    volatility = np.std(trade_returns) if trade_returns else 0

    metrics = {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'volatility': volatility
    }
    return metrics

def calculate_btc_ma_signals(data: pd.DataFrame, short_window=20, long_window=50, verbose=True) -> pd.DataFrame:
    """Calculate moving average crossover signals for BTC price data"""
    # 确保数据类型正确
    data = data.copy()  # 创建副本避免修改原始数据
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data.set_index('timestamp', inplace=True)
    
    if verbose:
        print(f"Data types after conversion: {data.dtypes}")
        print(f"Sample data after conversion:\n{data.head()}")
    
    # 使用收盘价计算移动平均线
    data['short_ma'] = data['close'].rolling(window=short_window).mean()
    data['long_ma'] = data['close'].rolling(window=long_window).mean()
    
    if verbose:
        print(f"Moving averages calculated. NaN count - short_ma: {data['short_ma'].isna().sum()}, long_ma: {data['long_ma'].isna().sum()}")
    
    # 生成交易信号
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # 买入信号
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # 卖出信号
    
    # 计算每日收益
    data['returns'] = data['signal'].shift(1) * data['close'].pct_change()
    
    # 计算累计收益
    data['cumulative_returns'] = (1 + data['returns']).cumprod()
    
    if verbose:
        # 计算并打印性能指标
        metrics = calculate_performance_metrics(data)
        print("\n=== 策略性能指标 ===")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"平均收益率: {metrics['avg_return']:.4%}")
        print(f"波动率: {metrics['volatility']:.4%}")
        print("==================\n")
    
    data.reset_index(inplace=True)
    return data

def calculate_ma_signals(data, short_window=10, long_window=30):
    """Calculate moving average crossover signals for volatility data"""
    # Calculate moving averages
    data['short_ma'] = data['price'].rolling(window=short_window).mean()
    data['long_ma'] = data['price'].rolling(window=long_window).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # Buy signal
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # Sell signal
    
    # Calculate daily returns (simplified)
    data['returns'] = data['signal'].shift(1) * data['price'].pct_change()
    
    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod()
    
    return data


@router.post("/strategy/run")
async def run_strategy(params: Dict[str, Any] = None):
    """Run the moving average crossover strategy with parameter grid search"""
    print("Starting strategy execution with params:", params)
    conn = None
    try:
        # Connect to database
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=3
            )
        except psycopg2.OperationalError as e:
            print(f"Database connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to database. Please ensure PostgreSQL is running and the database exists."
            )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 获取参数中的trading_pair，默认为BTCUSD
        trading_pair = params.get('trading_pair', 'BTCUSD') if params else 'BTCUSD'
        
        print(f"Fetching price data for trading pair: {trading_pair}")
        
        # 获取指定交易对的价格数据
        cur.execute("""
            SELECT timestamp, close
            FROM price_data
            WHERE trading_pair = %s
            ORDER BY timestamp ASC;
        """, (trading_pair,))
        rows = cur.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail=f"No price data found for trading pair: {trading_pair}")

        # 转换为DataFrame
        df = pd.DataFrame(rows)
        
        try:
            # 从params中获取参数范围
            if params:
                short_min = int(params.get('short_min', 20))
                short_max = int(params.get('short_max', 20))
                long_min = int(params.get('long_min', 50))
                long_max = int(params.get('long_max', 50))
            else:
                # 默认值，用于向后兼容
                short_min = short_max = 20
                long_min = long_max = 50
            
            # Check if we're doing grid search
            is_grid_search = (short_min != short_max) or (long_min != long_max)
            
            # Print trading pair info
            print(f"Trading Pair: {trading_pair}")
            
            if is_grid_search:
                # Import here to avoid circular dependency
                from .strategy_optimizer import run_grid_search
                
                # Use grid search optimizer
                optimization_result = run_grid_search(df, short_min, short_max, long_min, long_max)
                
                # Extract results
                results = optimization_result['results']
                metrics = optimization_result['metrics']
                best_params = optimization_result['best_params']
            else:
                # Single parameter execution (backward compatibility)
                short_window = short_min
                long_window = long_min
                
                print(f"Using parameters: short_window={short_window}, long_window={long_window}")
                results = calculate_btc_ma_signals(df, short_window=short_window, long_window=long_window, verbose=True)
                
                # Calculate metrics
                df_with_signals = pd.DataFrame(results)
                df_with_signals.set_index('timestamp', inplace=True)
                metrics = calculate_performance_metrics(df_with_signals)
                best_params = None
            
            # 将性能指标添加到响应中
            response_metrics = {
                'winRate': f"{metrics['win_rate']:.2%}",
                'monthlyReturn': f"{metrics['annual_return']/12:.2%}",
                'maxDrawdown': f"{metrics['max_drawdown']:.2%}",
                'sharpeRatio': f"{metrics['sharpe_ratio']:.2f}",
                'totalTrades': metrics['total_trades'],
                'avgReturn': f"{metrics['avg_return']:.4%}",
                'volatility': f"{metrics['volatility']:.4%}"
            }
            
        except Exception as e:
            print(f"Error in strategy calculation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Strategy calculation failed: {str(e)}")
        
        # Store results in equity_curves table
        cur.execute("""
            DELETE FROM equity_curves 
            WHERE strategy_key = 'test_strategy';
        """)

        # Insert new results
        base_equity = 100000
        for i, row in results.iterrows():
            equity = base_equity * row['cumulative_returns']
            month = row['timestamp'].strftime('%Y-%m')
            equity_value = float(equity) if not pd.isna(equity) else 100000.0
            
            cur.execute("""
                INSERT INTO equity_curves 
                (strategy_key, strategy_name, strategy_color, month, equity, snapshot_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, ('test_strategy', 'Test MA Strategy', '#8B5CF6', month, equity_value, row['timestamp']))

        conn.commit()
        
        response = {
            "status": "success", 
            "message": "Strategy executed successfully",
            "metrics": response_metrics
        }
        
        if best_params:
            response["best_params"] = best_params
        
        return response

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if conn:
            conn.close()

@router.get("/strategy/results")
async def get_strategy_results():
    """Get strategy results"""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT 
                strategy_key, 
                strategy_name, 
                strategy_color, 
                month, 
                CAST(equity AS float) as equity,
                snapshot_time
            FROM equity_curves
            WHERE strategy_key = 'test_strategy'
            ORDER BY snapshot_time ASC;
        """)
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No strategy results found")

        # Convert the results to a list of dictionaries with proper type handling
        formatted_rows = []
        for row in rows:
            formatted_row = {
                'strategy_key': str(row['strategy_key']),
                'strategy_name': str(row['strategy_name']),
                'strategy_color': str(row['strategy_color']),
                'month': str(row['month']),
                'equity': float(row['equity']),
                'snapshot_time': row['snapshot_time'].isoformat() if row['snapshot_time'] else None
            }
            formatted_rows.append(formatted_row)

        return formatted_rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if conn:
            conn.close()
