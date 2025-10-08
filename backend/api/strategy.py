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

def calculate_btc_ma_signals(data: pd.DataFrame, short_window=20, long_window=50) -> pd.DataFrame:
    """Calculate moving average crossover signals for BTC price data"""
    # 确保数据类型正确
    data = data.copy()  # 创建副本避免修改原始数据
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    
    print(f"Data types after conversion: {data.dtypes}")
    print(f"Sample data after conversion:\n{data.head()}")
    
    # 使用收盘价计算移动平均线
    data['short_ma'] = data['close'].rolling(window=short_window).mean()
    data['long_ma'] = data['close'].rolling(window=long_window).mean()
    
    print(f"Moving averages calculated. NaN count - short_ma: {data['short_ma'].isna().sum()}, long_ma: {data['long_ma'].isna().sum()}")
    
    # 生成交易信号
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # 买入信号
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # 卖出信号
    
    # 计算每日收益
    data['returns'] = data['signal'].shift(1) * data['close'].pct_change()
    
    # 计算累计收益
    data['cumulative_returns'] = (1 + data['returns']).cumprod()
    
    print(f"Strategy calculation completed. Final cumulative return: {data['cumulative_returns'].iloc[-1]}")
    
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
    """Run the moving average crossover strategy"""
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
                connect_timeout=3  # Add timeout
            )
        except psycopg2.OperationalError as e:
            print(f"Database connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to database. Please ensure PostgreSQL is running and the database exists."
            )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 获取BTC价格数据
        cur.execute("""
            SELECT timestamp, close
            FROM price_data
            WHERE trading_pair = 'BTCUSD'
            ORDER BY timestamp ASC;
        """)
        rows = cur.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No BTC price data found")

        # 转换为DataFrame并运行策略
        df = pd.DataFrame(rows)
        
        # 添加详细日志
        print(f"DataFrame columns: {df.columns}")
        print(f"First few rows: {df.head()}")
        
        try:
            results = calculate_btc_ma_signals(df)
            print(f"Strategy calculation successful. Results shape: {results.shape}")
        except Exception as e:
            print(f"Error in strategy calculation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Strategy calculation failed: {str(e)}")
        
        # Store results in equity_curves table
        # First, clear existing test strategy data
        cur.execute("""
            DELETE FROM equity_curves 
            WHERE strategy_key = 'test_strategy';
        """)

        # Insert new results
        base_equity = 100000  # Starting equity
        for i, row in results.iterrows():
            equity = base_equity * row['cumulative_returns']
            month = row['timestamp'].strftime('%B')
            
            # Ensure equity is a valid float
            equity_value = float(equity) if not pd.isna(equity) else 100000.0
            
            cur.execute("""
                INSERT INTO equity_curves 
                (strategy_key, strategy_name, strategy_color, month, equity, snapshot_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, ('test_strategy', 'Test MA Strategy', '#8B5CF6', month, equity_value, row['timestamp']))

        conn.commit()
        return {"status": "success", "message": "Strategy executed successfully"}

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
