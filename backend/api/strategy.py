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

def calculate_ma_signals(data, short_window=10, long_window=30):
    """Calculate moving average crossover signals"""
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

        # Get market data (using volatility as price for simulation)
        cur.execute("""
            SELECT timestamp, volatility as price
            FROM market_metrics
            ORDER BY timestamp ASC;
        """)
        rows = cur.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No market data found")

        # Convert to pandas DataFrame
        df = pd.DataFrame(rows)
        
        # Run strategy
        results = calculate_ma_signals(df)
        
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
