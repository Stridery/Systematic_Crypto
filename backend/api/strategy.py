from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

router = APIRouter()

# Database configuration
DB_NAME = "trading_db"
DB_USER = "zhenghaoyou"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432


class MAStrategy:
    """
    Moving Average Crossover Strategy Class
    
    This class encapsulates the logic for:
    - Calculating moving average crossover signals
    - Computing performance metrics
    - Printing trade details
    """
    
    def __init__(self, base_equity: float = 100000.0, risk_free_rate: float = 0.02):
        """
        Initialize the MA Strategy
        
        Args:
            base_equity: Base equity value for calculations
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.base_equity = base_equity
        self.risk_free_rate = risk_free_rate
    
    def print_trade_details(self, data: pd.DataFrame, verbose: bool = True) -> None:
        """
        Print detailed information for each trade
        
        Args:
            data: DataFrame with trading signals and results
            verbose: Whether to print details
        """
        if not verbose:
            return
        
        # 找出所有交易信号点（signal != 0的点）
        signal_points = data['signal'] != 0
        
        # 提取每笔交易的详细信息
        trade_count = 0
        position_start = None
        has_position = False  # 是否有持仓
        
        for i in range(len(data)):
            if signal_points.iloc[i]:
                signal_value = data['signal'].iloc[i]
                
                if signal_value == 1:  # 买入信号（开仓）
                    if has_position:
                        # 如果已经有持仓，先平仓再开仓（理论上不应该发生，但为了安全）
                        trade_count += 1
                        buy_date = data.index[position_start]
                        sell_date = data.index[i]
                        buy_price = data['open'].iloc[position_start]  # 买入使用open价格
                        sell_price = data['open'].iloc[i]  # 卖出使用open价格
                        
                        # 计算买入时的净值
                        if position_start == 0:
                            buy_equity = self.base_equity
                        else:
                            buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
                            if pd.isna(buy_cumulative_return):
                                buy_equity = self.base_equity
                            else:
                                buy_equity = self.base_equity * buy_cumulative_return
                        
                        # 计算卖出时的净值
                        sell_cumulative_return = data['cumulative_returns'].iloc[i]
                        if pd.isna(sell_cumulative_return):
                            sell_equity = self.base_equity
                        else:
                            sell_equity = self.base_equity * sell_cumulative_return
                        
                        trade_return_pct = (sell_price - buy_price) / buy_price
                        trade_profit = sell_equity - buy_equity
                        
                        # 获取买入和卖出日期的MA值
                        buy_short_ma = data['short_ma'].iloc[position_start]
                        buy_long_ma = data['long_ma'].iloc[position_start]
                        sell_short_ma = data['short_ma'].iloc[i]
                        sell_long_ma = data['long_ma'].iloc[i]
                    
                    # 开新仓
                    position_start = i
                    has_position = True
                    
                elif signal_value == -1:  # 卖出信号（平仓）
                    if has_position and position_start is not None:
                        trade_count += 1
                        # 获取交易信息
                        buy_date = data.index[position_start]
                        sell_date = data.index[i]
                        buy_price = data['open'].iloc[position_start]  # 买入使用open价格
                        sell_price = data['open'].iloc[i]  # 卖出使用open价格
                        
                        # 计算买入时的净值（如果是第一天，使用初始净值）
                        if position_start == 0:
                            buy_equity = self.base_equity
                        else:
                            buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
                            if pd.isna(buy_cumulative_return):
                                # 如果cumulative_returns是NaN，使用前一天的或者初始净值
                                if position_start > 0:
                                    prev_cumulative_return = data['cumulative_returns'].iloc[position_start - 1]
                                    if pd.isna(prev_cumulative_return):
                                        buy_equity = self.base_equity
                                    else:
                                        buy_equity = self.base_equity * prev_cumulative_return
                                else:
                                    buy_equity = self.base_equity
                            else:
                                buy_equity = self.base_equity * buy_cumulative_return
                        
                        # 计算卖出时的净值
                        sell_cumulative_return = data['cumulative_returns'].iloc[i]
                        if pd.isna(sell_cumulative_return):
                            sell_equity = self.base_equity
                        else:
                            sell_equity = self.base_equity * sell_cumulative_return
                        
                        # 计算交易收益
                        trade_return_pct = (sell_price - buy_price) / buy_price
                        trade_profit = sell_equity - buy_equity
                        
                        # 获取买入和卖出日期的MA值
                        buy_short_ma = data['short_ma'].iloc[position_start]
                        buy_long_ma = data['long_ma'].iloc[position_start]
                        sell_short_ma = data['short_ma'].iloc[i]
                        sell_long_ma = data['long_ma'].iloc[i]
                        
                        # 平仓后重置
                        has_position = False
                        position_start = None
        
        # 如果最后还有持仓，显示当前持仓信息
        if has_position and position_start is not None:
            buy_date = data.index[position_start]
            buy_price = data['open'].iloc[position_start]  # 买入使用open价格
            
            # 计算买入时的净值（如果是第一天，使用初始净值）
            if position_start == 0:
                buy_equity = self.base_equity
            else:
                buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
                if pd.isna(buy_cumulative_return):
                    if position_start > 0:
                        prev_cumulative_return = data['cumulative_returns'].iloc[position_start - 1]
                        if pd.isna(prev_cumulative_return):
                            buy_equity = self.base_equity
                        else:
                            buy_equity = self.base_equity * prev_cumulative_return
                    else:
                        buy_equity = self.base_equity
                else:
                    buy_equity = self.base_equity * buy_cumulative_return
            
            last_date = data.index[-1]
            last_price = data['close'].iloc[-1]
            last_cumulative_return = data['cumulative_returns'].iloc[-1]
            if pd.isna(last_cumulative_return):
                last_equity = self.base_equity
            else:
                last_equity = self.base_equity * last_cumulative_return
            
            # 获取买入日和当前日的MA值
            buy_short_ma = data['short_ma'].iloc[position_start]
            buy_long_ma = data['long_ma'].iloc[position_start]
            last_short_ma = data['short_ma'].iloc[-1]
            last_long_ma = data['long_ma'].iloc[-1]
    
    def calculate_performance_metrics(self, data: pd.DataFrame, verbose: bool = False) -> dict:
        """
        Calculate detailed performance metrics for the strategy
        
        Args:
            data: DataFrame with trading signals and results (indexed by timestamp)
            verbose: Whether to print debug information
            
        Returns:
            Dictionary containing performance metrics
        """
        # 确保cumulative_returns是权益曲线（从1.0开始）
        equity_curve = data['cumulative_returns']
        
        # 计算年化收益率
        total_days = (data.index[-1] - data.index[0]).days
        if total_days == 0:
            total_days = 1  # 避免除零错误
        total_return = equity_curve.iloc[-1] - 1  # 总收益率
        annual_return = (1 + total_return) ** (365 / total_days) - 1

        # 计算夏普比率
        rf_daily = (1 + self.risk_free_rate) ** (1/365) - 1  # 日频无风险利率
        
        # 计算equity curve的日收益率
        daily_returns = equity_curve.pct_change().fillna(0)
        
        # 确保第一天的收益率为0
        if len(daily_returns) > 0:
            daily_returns.iloc[0] = 0
        
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        
        if std_daily == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = ((mean_daily - rf_daily) / std_daily) * np.sqrt(365)

        # 计算最大回撤
        cumulative_max = equity_curve.cummax()
        drawdowns = equity_curve / cumulative_max - 1
        max_drawdown = drawdowns.min()

        # 计算交易统计
        trade_returns = []
        position_start = None
        has_position = False
        
        # 找出所有交易信号点（signal != 0的点）
        signal_points = data['signal'] != 0
        
        for i in range(len(data)):
            if signal_points.iloc[i]:
                signal_value = data['signal'].iloc[i]
                
                if signal_value == 1:  # 买入信号（开仓）
                    if has_position:
                        # 如果已经有持仓，先平仓再开仓（理论上不应该发生）
                        if position_start is not None and position_start < i:
                            trade_period_returns = data['returns'].iloc[position_start:i+1]
                            trade_return = (1 + trade_period_returns).prod() - 1
                            trade_returns.append(trade_return)
                    
                    # 开新仓
                    position_start = i
                    has_position = True
                    
                elif signal_value == -1:  # 卖出信号（平仓）
                    if has_position and position_start is not None:
                        # 计算这笔交易的累计收益
                        trade_period_returns = data['returns'].iloc[position_start:i+1]
                        trade_return = (1 + trade_period_returns).prod() - 1
                        trade_returns.append(trade_return)
                        
                        # 平仓后重置
                        has_position = False
                        position_start = None
        
        # 计算交易统计
        total_trades = len(trade_returns)
        winning_trades = sum(1 for r in trade_returns if r > 0)
        losing_trades = sum(1 for r in trade_returns if r < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均收益率和波动率（基于交易）
        avg_return = np.mean(trade_returns) if trade_returns else 0
        volatility = np.std(trade_returns) if trade_returns else 0
        
        # 计算Profit Factor（盈利因子）
        total_profit = sum(r for r in trade_returns if r > 0)  # 总盈利
        total_loss = abs(sum(r for r in trade_returns if r < 0))  # 总亏损（取绝对值）
        
        if total_loss == 0:
            if total_profit > 0:
                profit_factor = float('inf')  # 只有盈利，没有亏损
            else:
                profit_factor = 0.0  # 没有交易或没有盈利
        else:
            profit_factor = total_profit / total_loss

        metrics = {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'volatility': volatility,
            'profit_factor': profit_factor
        }
        return metrics
    
    def calculate_signals(self, data: pd.DataFrame, short_window: int = 20, long_window: int = 50, 
                          start_date: Optional[Any] = None, end_date: Optional[Any] = None, 
                          verbose: bool = True, calculate_daily_equity: bool = False) -> pd.DataFrame:
        """
        Calculate moving average crossover signals for price data
        
        Args:
            data: Price data DataFrame with 'timestamp', 'open', 'close' columns
            short_window: Short moving average window
            long_window: Long moving average window
            start_date: Start date for trading (str 'YYYY-MM-DD' or datetime). If None, uses default logic.
            end_date: End date for trading (str 'YYYY-MM-DD' or datetime). If None, uses last date in data.
            verbose: Whether to print debug information
            calculate_daily_equity: Whether to calculate daily equity curve for plotting
            
        Returns:
            DataFrame with signals, returns, and cumulative_returns
        """
        # 确保数据类型正确
        data = data.copy()  # 创建副本避免修改原始数据
        data['open'] = pd.to_numeric(data['open'], errors='coerce')
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        
        # 处理timestamp列：统一处理时区问题
        if 'timestamp' in data.columns:
            timestamp_col = pd.to_datetime(data['timestamp'], utc=True)
            # 移除时区信息，转换为naive datetime
            if isinstance(timestamp_col, pd.Series):
                if timestamp_col.dt.tz is not None:
                    timestamp_col = timestamp_col.dt.tz_convert('UTC').dt.tz_localize(None)
            elif isinstance(timestamp_col, pd.DatetimeIndex):
                if timestamp_col.tz is not None:
                    timestamp_col = timestamp_col.tz_convert('UTC').tz_localize(None)
            data['timestamp'] = timestamp_col
        
        # 设置index
        data.set_index('timestamp', inplace=True)
        
        # 确保index是DatetimeIndex且不带时区
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        
        # 如果index有时区信息，转换为UTC并移除时区
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC').tz_localize(None)
        
        # 处理日期参数（确保不带时区）
        if start_date is not None:
            start_date = pd.to_datetime(start_date, utc=False).normalize()
            if isinstance(start_date, pd.Timestamp) and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            elif isinstance(start_date, datetime) and start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
        else:
            # 默认逻辑：使用最后365天
            end_date_default = data.index[-1]
            start_date = end_date_default - timedelta(days=365)
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date, utc=False).normalize()
            if isinstance(end_date, pd.Timestamp) and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            elif isinstance(end_date, datetime) and end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)
        else:
            # 默认使用数据的最后日期
            end_date = data.index[-1]
        
        # 计算init phase的开始日期：start_date往前推long_window天
        init_start_date = start_date - timedelta(days=long_window)
        
        # 过滤数据：保留从init_start_date到end_date的数据
        mask = (data.index >= init_start_date) & (data.index <= end_date)
        data = data[mask].copy()
        
        if len(data) == 0:
            raise ValueError(f"No data found in the specified date range: {init_start_date} to {end_date}")
        
        # 计算移动平均线（使用所有过滤后的数据，包括init phase）
        data['short_ma'] = data['close'].rolling(window=short_window).mean().shift(1)
        data['long_ma'] = data['close'].rolling(window=long_window).mean().shift(1)
        
        # 找到start_date对应的索引（交易开始日期）
        start_idx = data.index.get_indexer([start_date], method='nearest')[0]
        if start_idx < 0:
            start_idx = 0
        
        # 保存第一天的MA值（在切片之前）
        first_day_short_ma = data['short_ma'].iloc[start_idx] if start_idx < len(data) else None
        first_day_long_ma = data['long_ma'].iloc[start_idx] if start_idx < len(data) else None
        
        # 只保留start_date到end_date的数据用于回测（去掉init phase）
        data = data.iloc[start_idx:].copy()
        
        # 生成交易信号：只在交叉时产生信号
        data['signal'] = 0
        
        # 第一天（开始日期）的信号：如果short_ma > long_ma，signal=1，否则signal=0
        if len(data) > 0 and first_day_short_ma is not None and first_day_long_ma is not None:
            if not pd.isna(first_day_short_ma) and not pd.isna(first_day_long_ma):
                if first_day_short_ma > first_day_long_ma:
                    data.iloc[0, data.columns.get_loc('signal')] = 1
                else:
                    data.iloc[0, data.columns.get_loc('signal')] = 0
        
        # 从第二天开始，检测交叉信号
        if len(data) > 1:
            # 检测上穿：当前short_ma > long_ma，且前一日short_ma <= long_ma
            golden_cross = (data['short_ma'] > data['long_ma']) & (data['short_ma'].shift(1) <= data['long_ma'].shift(1))
            data.loc[golden_cross, 'signal'] = 1  # 买入信号
            
            # 检测下穿：当前short_ma < long_ma，且前一日short_ma >= long_ma
            death_cross = (data['short_ma'] < data['long_ma']) & (data['short_ma'].shift(1) >= data['long_ma'].shift(1))
            data.loc[death_cross, 'signal'] = -1  # 卖出信号
        
        # 计算持仓状态
        data['position'] = data['signal'].replace({1: 1, -1: 0, 0: np.nan}).ffill().fillna(0).astype(int)
        
        # 计算每日收益
        data['returns'] = 0.0
        current_buy_open = None
        current_buy_idx = None
        
        for i in range(len(data)):
            # 如果当天有买入信号，使用open价格买入，当天收益为0
            if data['signal'].iloc[i] == 1:
                current_buy_open = data['open'].iloc[i]
                current_buy_idx = i
                data.iloc[i, data.columns.get_loc('returns')] = 0.0
            
            # 如果当天有卖出信号，使用open价格卖出
            elif data['signal'].iloc[i] == -1:
                if current_buy_open is not None:
                    sell_price = data['open'].iloc[i]
                    trade_return = (sell_price - current_buy_open) / current_buy_open
                    data.iloc[i, data.columns.get_loc('returns')] = trade_return
                    current_buy_open = None
                    current_buy_idx = None
                else:
                    data.iloc[i, data.columns.get_loc('returns')] = 0.0
        
        # 不再强制平仓：如果到end_date还有未完成的交易，直接忽略最后一次交易
        # 性能指标计算时只统计已完成的交易（signal=-1），未完成的交易会被忽略
        
        # 重新计算持仓状态
        data['position'] = data['signal'].replace({1: 1, -1: 0, 0: np.nan}).ffill().fillna(0).astype(int)
        
        # 计算累计收益
        if calculate_daily_equity:
            # 计算每日净值变化（用于绘图）
            data['cumulative_returns'] = 1.0
            current_buy_open = None
            base_equity_multiplier = 1.0
            
            for i in range(len(data)):
                if data['signal'].iloc[i] == 1:  # 买入信号
                    current_buy_open = data['open'].iloc[i]
                    current_close = data['close'].iloc[i]
                    daily_multiplier = current_close / current_buy_open
                    data.iloc[i, data.columns.get_loc('cumulative_returns')] = base_equity_multiplier * daily_multiplier
                
                elif data['signal'].iloc[i] == -1:  # 卖出信号
                    if current_buy_open is not None:
                        sell_price = data['open'].iloc[i]
                        trade_multiplier = sell_price / current_buy_open
                        base_equity_multiplier = base_equity_multiplier * trade_multiplier
                        data.iloc[i, data.columns.get_loc('cumulative_returns')] = base_equity_multiplier
                        current_buy_open = None
                    else:
                        data.iloc[i, data.columns.get_loc('cumulative_returns')] = base_equity_multiplier
                
                else:  # 持仓期间或空仓期间
                    if current_buy_open is not None:  # 持仓期间
                        current_close = data['close'].iloc[i]
                        daily_multiplier = current_close / current_buy_open
                        data.iloc[i, data.columns.get_loc('cumulative_returns')] = base_equity_multiplier * daily_multiplier
                    else:  # 空仓期间
                        data.iloc[i, data.columns.get_loc('cumulative_returns')] = base_equity_multiplier
        else:
            # 简化版：只计算用于性能指标的cumulative_returns
            data['cumulative_returns'] = (1 + data['returns'].fillna(0)).cumprod()
        
        if verbose:
            # 打印每笔交易的详细信息
            self.print_trade_details(data, verbose=verbose)
            
            # 计算并打印性能指标
            metrics = self.calculate_performance_metrics(data, verbose=verbose)
        
        data.reset_index(inplace=True)
        return data


# 为了向后兼容，保留函数接口
def calculate_btc_ma_signals(data: pd.DataFrame, short_window=20, long_window=50, 
                              start_date=None, end_date=None, verbose=True, 
                              calculate_daily_equity=False) -> pd.DataFrame:
    """Backward compatibility wrapper for calculate_signals"""
    strategy = MAStrategy()
    return strategy.calculate_signals(data, short_window, long_window, start_date, 
                                      end_date, verbose, calculate_daily_equity)


def calculate_performance_metrics(data: pd.DataFrame, verbose: bool = False) -> dict:
    """Backward compatibility wrapper for calculate_performance_metrics"""
    strategy = MAStrategy()
    return strategy.calculate_performance_metrics(data, verbose)


def print_trade_details(data: pd.DataFrame, base_equity: float = 100000.0, verbose: bool = True):
    """Backward compatibility wrapper for print_trade_details"""
    strategy = MAStrategy(base_equity=base_equity)
    strategy.print_trade_details(data, verbose)


@router.post("/strategy/run")
async def run_strategy(params: Dict[str, Any] = None):
    """Run the moving average crossover strategy with parameter grid search"""
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
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to database. Please ensure PostgreSQL is running and the database exists."
            )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 获取参数中的trading_pair，默认为BTCUSD
        trading_pair = params.get('trading_pair', 'BTCUSD') if params else 'BTCUSD'
        
        # 获取指定交易对的价格数据
        cur.execute("""
            SELECT timestamp, open, close
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
                
                train_start_date = params.get('train_start_date')
                train_end_date = params.get('train_end_date')
                validation_start_date = params.get('validation_start_date')
                validation_end_date = params.get('validation_end_date')
            else:
                short_min = short_max = 20
                long_min = long_max = 50
                train_start_date = None
                train_end_date = None
                validation_start_date = None
                validation_end_date = None
            
            # Check if we're doing grid search
            is_grid_search = (short_min != short_max) or (long_min != long_max)
            
            # Determine if we have both training and validation dates
            has_training_dates = train_start_date and train_end_date
            has_validation_dates = validation_start_date and validation_end_date
            is_training_validation_mode = has_training_dates and has_validation_dates
            
            # 用于存储equity curve数据的变量
            training_equity_curve_data = None
            validation_equity_curve_data = None
            
            # 创建策略实例
            strategy = MAStrategy()
            
            if is_training_validation_mode:
                # Training-validation mode
                if is_grid_search:
                    # Grid search mode: use optimizer
                    from .strategy_optimizer import StrategyOptimizer
                    
                    print(f"Training Period: {train_start_date} to {train_end_date}")
                    print(f"Validation Period: {validation_start_date} to {validation_end_date}")
                    
                    optimizer = StrategyOptimizer(strategy)
                    workflow_result = optimizer.run_training_validation(
                        df,
                        short_min,
                        short_max,
                        long_min,
                        long_max,
                        train_start_date=train_start_date,
                        train_end_date=train_end_date,
                        validation_start_date=validation_start_date,
                        validation_end_date=validation_end_date
                    )
                    
                    results = workflow_result['training_results']
                    metrics = workflow_result['training_metrics']
                    best_params = workflow_result['best_params']
                    validation_metrics = workflow_result['validation_metrics']
                    training_equity_curve_data = workflow_result.get('training_results')
                    validation_equity_curve_data = workflow_result.get('validation_results')
                else:
                    # Single parameter mode
                    short_window = short_min
                    long_window = long_min
                    
                    # Run strategy on training period
                    training_results = strategy.calculate_signals(
                        df, 
                        short_window=short_window, 
                        long_window=long_window,
                        start_date=train_start_date,
                        end_date=train_end_date,
                        verbose=True,
                        calculate_daily_equity=True
                    )
                    
                    # Calculate training metrics
                    training_df = pd.DataFrame(training_results)
                    training_df.set_index('timestamp', inplace=True)
                    metrics = strategy.calculate_performance_metrics(training_df)
                    
                    # Run strategy on validation period
                    validation_results = strategy.calculate_signals(
                        df,
                        short_window=short_window,
                        long_window=long_window,
                        start_date=validation_start_date,
                        end_date=validation_end_date,
                        verbose=True,
                        calculate_daily_equity=True
                    )
                    
                    # Calculate validation metrics
                    validation_df = pd.DataFrame(validation_results)
                    validation_df.set_index('timestamp', inplace=True)
                    validation_metrics = strategy.calculate_performance_metrics(validation_df, verbose=True)
                    
                    results = training_results
                    best_params = None
                    training_equity_curve_data = training_results
                    validation_equity_curve_data = validation_results
                
            elif is_grid_search:
                # Grid search mode (training only, no validation)
                from .strategy_optimizer import StrategyOptimizer
                
                if train_start_date and train_end_date:
                    print(f"Training Period: {train_start_date} to {train_end_date}")
                
                optimizer = StrategyOptimizer(strategy)
                optimization_result = optimizer.run_grid_search(
                    df, 
                    short_min, 
                    short_max, 
                    long_min, 
                    long_max,
                    start_date=train_start_date,
                    end_date=train_end_date
                )
                
                results = optimization_result['results']
                metrics = optimization_result['metrics']
                best_params = optimization_result['best_params']
                validation_metrics = None
                
            else:
                # Single parameter execution
                short_window = short_min
                long_window = long_min
                
                results = strategy.calculate_signals(
                    df, 
                    short_window=short_window, 
                    long_window=long_window,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    verbose=True,
                    calculate_daily_equity=True
                )
                
                # Calculate metrics
                df_with_signals = pd.DataFrame(results)
                df_with_signals.set_index('timestamp', inplace=True)
                metrics = strategy.calculate_performance_metrics(df_with_signals)
                best_params = None
                validation_metrics = None
            
            # 格式化性能指标
            profit_factor_value = metrics.get('profit_factor', 0.0)
            if profit_factor_value == float('inf'):
                profit_factor_str = "∞"
            else:
                profit_factor_str = f"{profit_factor_value:.2f}"
            
            response_metrics = {
                'winRate': f"{metrics['win_rate']:.2%}",
                'monthlyReturn': f"{metrics['annual_return']/12:.2%}",
                'maxDrawdown': f"{metrics['max_drawdown']:.2%}",
                'sharpeRatio': f"{metrics['sharpe_ratio']:.2f}",
                'profitFactor': profit_factor_str,
                'totalTrades': metrics['total_trades'],
                'avgReturn': f"{metrics['avg_return']:.4%}",
                'volatility': f"{metrics['volatility']:.4%}"
            }
            
            # 如果有validation metrics，格式化并添加到响应中
            validation_response_metrics = None
            if validation_metrics:
                validation_profit_factor_value = validation_metrics.get('profit_factor', 0.0)
                if validation_profit_factor_value == float('inf'):
                    validation_profit_factor_str = "∞"
                else:
                    validation_profit_factor_str = f"{validation_profit_factor_value:.2f}"
                
                validation_response_metrics = {
                    'winRate': f"{validation_metrics['win_rate']:.2%}",
                    'monthlyReturn': f"{validation_metrics['annual_return']/12:.2%}",
                    'maxDrawdown': f"{validation_metrics['max_drawdown']:.2%}",
                    'sharpeRatio': f"{validation_metrics['sharpe_ratio']:.2f}",
                    'profitFactor': validation_profit_factor_str,
                    'totalTrades': validation_metrics['total_trades'],
                    'avgReturn': f"{validation_metrics['avg_return']:.4%}",
                    'volatility': f"{validation_metrics['volatility']:.4%}"
                }
            
        except Exception as e:
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
        
        # 构建响应
        response = {
            "status": "success", 
            "message": "Strategy executed successfully",
            "metrics": response_metrics
        }
        
        # 如果有training和validation metrics，使用training-validation模式
        if validation_response_metrics:
            response["training_metrics"] = response_metrics
            response["validation_metrics"] = validation_response_metrics
            del response["metrics"]
            
            # 添加training和validation的equity curve数据
            training_equity_curve = []
            validation_equity_curve = []
            
            if is_training_validation_mode and training_equity_curve_data is not None:
                training_df = pd.DataFrame(training_equity_curve_data)
                if 'timestamp' in training_df.columns and 'cumulative_returns' in training_df.columns:
                    base_equity = 100000
                    for _, row in training_df.iterrows():
                        equity = base_equity * row['cumulative_returns']
                        timestamp_str = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                        training_equity_curve.append({
                            'timestamp': timestamp_str,
                            'equity': float(equity) if not pd.isna(equity) else base_equity
                        })
            
            if is_training_validation_mode and validation_equity_curve_data is not None:
                validation_df = pd.DataFrame(validation_equity_curve_data)
                if 'timestamp' in validation_df.columns and 'cumulative_returns' in validation_df.columns:
                    base_equity = 100000
                    for _, row in validation_df.iterrows():
                        equity = base_equity * row['cumulative_returns']
                        timestamp_str = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                        validation_equity_curve.append({
                            'timestamp': timestamp_str,
                            'equity': float(equity) if not pd.isna(equity) else base_equity
                        })
            
            if training_equity_curve or validation_equity_curve:
                response["training_equity_curve"] = training_equity_curve
                response["validation_equity_curve"] = validation_equity_curve
        
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
