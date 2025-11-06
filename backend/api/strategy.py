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

def print_trade_details(data: pd.DataFrame, base_equity: float = 100000.0, verbose: bool = True):
    """Print detailed information for each trade"""
    if not verbose:
        return
    
    # 找出所有交易信号点（signal != 0的点）
    signal_points = data['signal'] != 0
    
    # 提取每笔交易的详细信息
    trade_count = 0
    position_start = None
    has_position = False  # 是否有持仓
    
    print("\n" + "="*100)
    print("交易明细 (Trade Details)")
    print("="*100)
    
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
                    sell_price = data['close'].iloc[i]  # 卖出使用close价格
                    
                    # 计算买入时的净值
                    if position_start == 0:
                        buy_equity = base_equity
                    else:
                        buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
                        if pd.isna(buy_cumulative_return):
                            buy_equity = base_equity
                        else:
                            buy_equity = base_equity * buy_cumulative_return
                    
                    # 计算卖出时的净值
                    sell_cumulative_return = data['cumulative_returns'].iloc[i]
                    if pd.isna(sell_cumulative_return):
                        sell_equity = base_equity
                    else:
                        sell_equity = base_equity * sell_cumulative_return
                    
                    trade_return_pct = (sell_price - buy_price) / buy_price
                    trade_profit = sell_equity - buy_equity
                    
                    # 获取买入和卖出日期的MA值
                    buy_short_ma = data['short_ma'].iloc[position_start]
                    buy_long_ma = data['long_ma'].iloc[position_start]
                    sell_short_ma = data['short_ma'].iloc[i]
                    sell_long_ma = data['long_ma'].iloc[i]
                    
                    print(f"\n交易 #{trade_count}")
                    print(f"  买入日期: {buy_date.strftime('%Y-%m-%d')}")
                    print(f"  买入价格: ${buy_price:,.2f}")
                    print(f"  买入日 Short MA: ${buy_short_ma:,.2f}" if not pd.isna(buy_short_ma) else "  买入日 Short MA: N/A")
                    print(f"  买入日 Long MA: ${buy_long_ma:,.2f}" if not pd.isna(buy_long_ma) else "  买入日 Long MA: N/A")
                    print(f"  买入前净值: ${buy_equity:,.2f}")
                    print(f"  卖出日期: {sell_date.strftime('%Y-%m-%d')}")
                    print(f"  卖出价格: ${sell_price:,.2f}")
                    print(f"  卖出日 Short MA: ${sell_short_ma:,.2f}" if not pd.isna(sell_short_ma) else "  卖出日 Short MA: N/A")
                    print(f"  卖出日 Long MA: ${sell_long_ma:,.2f}" if not pd.isna(sell_long_ma) else "  卖出日 Long MA: N/A")
                    print(f"  卖出后净值: ${sell_equity:,.2f}")
                    print(f"  持仓天数: {(sell_date - buy_date).days} 天")
                    print(f"  交易收益: ${trade_profit:,.2f}")
                    print(f"  收益率: {trade_return_pct:.2%}")
                    print(f"  {'盈利' if trade_profit > 0 else '亏损' if trade_profit < 0 else '持平'}")
                    print("-" * 100)
                
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
                    sell_price = data['close'].iloc[i]  # 卖出使用close价格
                    
                    # 计算买入时的净值（如果是第一天，使用初始净值）
                    if position_start == 0:
                        buy_equity = base_equity
                    else:
                        buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
                        if pd.isna(buy_cumulative_return):
                            # 如果cumulative_returns是NaN，使用前一天的或者初始净值
                            if position_start > 0:
                                prev_cumulative_return = data['cumulative_returns'].iloc[position_start - 1]
                                if pd.isna(prev_cumulative_return):
                                    buy_equity = base_equity
                                else:
                                    buy_equity = base_equity * prev_cumulative_return
                            else:
                                buy_equity = base_equity
                        else:
                            buy_equity = base_equity * buy_cumulative_return
                    
                    # 计算卖出时的净值
                    sell_cumulative_return = data['cumulative_returns'].iloc[i]
                    if pd.isna(sell_cumulative_return):
                        sell_equity = base_equity
                    else:
                        sell_equity = base_equity * sell_cumulative_return
                    
                    # 计算交易收益
                    trade_return_pct = (sell_price - buy_price) / buy_price
                    trade_profit = sell_equity - buy_equity
                    
                    # 获取买入和卖出日期的MA值
                    buy_short_ma = data['short_ma'].iloc[position_start]
                    buy_long_ma = data['long_ma'].iloc[position_start]
                    sell_short_ma = data['short_ma'].iloc[i]
                    sell_long_ma = data['long_ma'].iloc[i]
                    
                    # 打印交易详情
                    print(f"\n交易 #{trade_count}")
                    print(f"  买入日期: {buy_date.strftime('%Y-%m-%d')}")
                    print(f"  买入价格: ${buy_price:,.2f}")
                    print(f"  买入日 Short MA: ${buy_short_ma:,.2f}" if not pd.isna(buy_short_ma) else "  买入日 Short MA: N/A")
                    print(f"  买入日 Long MA: ${buy_long_ma:,.2f}" if not pd.isna(buy_long_ma) else "  买入日 Long MA: N/A")
                    print(f"  买入前净值: ${buy_equity:,.2f}")
                    print(f"  卖出日期: {sell_date.strftime('%Y-%m-%d')}")
                    print(f"  卖出价格: ${sell_price:,.2f}")
                    print(f"  卖出日 Short MA: ${sell_short_ma:,.2f}" if not pd.isna(sell_short_ma) else "  卖出日 Short MA: N/A")
                    print(f"  卖出日 Long MA: ${sell_long_ma:,.2f}" if not pd.isna(sell_long_ma) else "  卖出日 Long MA: N/A")
                    print(f"  卖出后净值: ${sell_equity:,.2f}")
                    print(f"  持仓天数: {(sell_date - buy_date).days} 天")
                    print(f"  交易收益: ${trade_profit:,.2f}")
                    print(f"  收益率: {trade_return_pct:.2%}")
                    print(f"  {'盈利' if trade_profit > 0 else '亏损' if trade_profit < 0 else '持平'}")
                    print("-" * 100)
                    
                    # 平仓后重置
                    has_position = False
                    position_start = None
    
    # 如果最后还有持仓，显示当前持仓信息
    if has_position and position_start is not None:
        buy_date = data.index[position_start]
        buy_price = data['open'].iloc[position_start]  # 买入使用open价格
        
        # 计算买入时的净值（如果是第一天，使用初始净值）
        if position_start == 0:
            buy_equity = base_equity
        else:
            buy_cumulative_return = data['cumulative_returns'].iloc[position_start]
            if pd.isna(buy_cumulative_return):
                if position_start > 0:
                    prev_cumulative_return = data['cumulative_returns'].iloc[position_start - 1]
                    if pd.isna(prev_cumulative_return):
                        buy_equity = base_equity
                    else:
                        buy_equity = base_equity * prev_cumulative_return
                else:
                    buy_equity = base_equity
            else:
                buy_equity = base_equity * buy_cumulative_return
        
        last_date = data.index[-1]
        last_price = data['close'].iloc[-1]
        last_cumulative_return = data['cumulative_returns'].iloc[-1]
        if pd.isna(last_cumulative_return):
            last_equity = base_equity
        else:
            last_equity = base_equity * last_cumulative_return
        
        # 获取买入日和当前日的MA值
        buy_short_ma = data['short_ma'].iloc[position_start]
        buy_long_ma = data['long_ma'].iloc[position_start]
        last_short_ma = data['short_ma'].iloc[-1]
        last_long_ma = data['long_ma'].iloc[-1]
        
        print(f"\n当前持仓 (未平仓)")
        print(f"  买入日期: {buy_date.strftime('%Y-%m-%d')}")
        print(f"  买入价格: ${buy_price:,.2f}")
        print(f"  买入日 Short MA: ${buy_short_ma:,.2f}" if not pd.isna(buy_short_ma) else "  买入日 Short MA: N/A")
        print(f"  买入日 Long MA: ${buy_long_ma:,.2f}" if not pd.isna(buy_long_ma) else "  买入日 Long MA: N/A")
        print(f"  买入时净值: ${buy_equity:,.2f}")
        print(f"  当前日期: {last_date.strftime('%Y-%m-%d')}")
        print(f"  当前价格: ${last_price:,.2f}")
        print(f"  当前日 Short MA: ${last_short_ma:,.2f}" if not pd.isna(last_short_ma) else "  当前日 Short MA: N/A")
        print(f"  当前日 Long MA: ${last_long_ma:,.2f}" if not pd.isna(last_long_ma) else "  当前日 Long MA: N/A")
        print(f"  当前净值: ${last_equity:,.2f}")
        print(f"  持仓天数: {(last_date - buy_date).days} 天")
        print(f"  浮动盈亏: ${last_equity - buy_equity:,.2f}")
        print(f"  浮动收益率: {(last_price - buy_price) / buy_price:.2%}")
        print("-" * 100)
    
    print(f"\n总交易次数: {trade_count}")
    print("="*100 + "\n")

def calculate_performance_metrics(data: pd.DataFrame, verbose: bool = False) -> dict:
    """Calculate detailed performance metrics for the strategy"""
    # 确保cumulative_returns是权益曲线（从1.0开始）
    # cumulative_returns应该是(1+returns).cumprod()，表示累计财富倍数
    equity_curve = data['cumulative_returns']
    
    # 计算年化收益率
    total_days = (data.index[-1] - data.index[0]).days
    if total_days == 0:
        total_days = 1  # 避免除零错误
    total_return = equity_curve.iloc[-1] - 1  # 总收益率
    annual_return = (1 + total_return) ** (365 / total_days) - 1

    # 计算夏普比率（假设无风险利率为2%）
    # 使用日频均值与标准差年化的标准方法
    risk_free_rate = 0.02
    rf_daily = (1 + risk_free_rate) ** (1/365) - 1  # 日频无风险利率
    mean_daily = data['returns'].mean()
    std_daily = data['returns'].std()
    
    if std_daily == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = ((mean_daily - rf_daily) / std_daily) * np.sqrt(365)

    # 计算最大回撤（更稳健的写法）
    # equity_curve应该从1.0开始，表示累计财富
    cumulative_max = equity_curve.cummax()
    drawdowns = equity_curve / cumulative_max - 1
    max_drawdown = drawdowns.min()

    # 计算交易统计
    # 使用与print_trade_details相同的逻辑：只统计signal=-1（卖出）时的交易
    # 找出所有卖出信号（signal=-1），这些就是平仓点
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
                    # 计算这笔交易的累计收益
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
                    # 使用开仓到平仓区间的(1+returns).prod()-1
                    trade_period_returns = data['returns'].iloc[position_start:i+1]
                    # 计算累计收益：(1+r1)*(1+r2)*...*(1+rn) - 1
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
    
    # 调试信息：打印交易统计详情
    if verbose:
        print(f"\n交易统计详情:")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利交易: {winning_trades}")
        print(f"  亏损交易: {losing_trades}")
        print(f"  每笔交易收益: {[f'{r:.4%}' for r in trade_returns]}")
    
    # 计算平均收益率和波动率（基于交易）
    avg_return = np.mean(trade_returns) if trade_returns else 0
    volatility = np.std(trade_returns) if trade_returns else 0
    
    # 计算Profit Factor（盈利因子）
    # Profit Factor = 总盈利交易的总盈利 / 总亏损交易的总亏损
    # 如果总亏损为0，Profit Factor设为无穷大或一个很大的数
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

def calculate_btc_ma_signals(data: pd.DataFrame, short_window=20, long_window=50, verbose=True) -> pd.DataFrame:
    """Calculate moving average crossover signals for BTC price data"""
    # 确保数据类型正确
    data = data.copy()  # 创建副本避免修改原始数据
    data['open'] = pd.to_numeric(data['open'], errors='coerce')
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data.set_index('timestamp', inplace=True)
    
    if verbose:
        print(f"Data types after conversion: {data.dtypes}")
        print(f"Sample data after conversion:\n{data.head()}")
    
    # 确定开始日期（三年前）
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=365)  # 三年前
    
    if verbose:
        print(f"Strategy start date: {start_date.strftime('%Y-%m-%d')}")
        print(f"Strategy end date: {end_date.strftime('%Y-%m-%d')}")
    
    # 使用收盘价计算移动平均线（使用所有数据，包括开始日期之前的数据）
    # 注意：MA计算后shift(1)，确保当天的MA是基于前一天的数据计算的
    # 例如：第10天的short_ma（window=1）应该是第9天的close
    #      第10天的long_ma（window=2）应该是第8天和第9天的close的平均值
    data['short_ma'] = data['close'].rolling(window=short_window).mean().shift(1)
    data['long_ma'] = data['close'].rolling(window=long_window).mean().shift(1)
    
    if verbose:
        print(f"Moving averages calculated. NaN count - short_ma: {data['short_ma'].isna().sum()}, long_ma: {data['long_ma'].isna().sum()}")
        print(f"Total data points: {len(data)}")
    
    # 找到开始日期对应的索引
    start_idx = data.index.get_indexer([start_date], method='nearest')[0]
    if start_idx < 0:
        start_idx = 0
    
    if verbose:
        print(f"Start index: {start_idx}, Start date: {data.index[start_idx]}")
    
    # 保存第一天的MA值（在切片之前）
    first_day_short_ma = data['short_ma'].iloc[start_idx] if start_idx < len(data) else None
    first_day_long_ma = data['long_ma'].iloc[start_idx] if start_idx < len(data) else None
    
    # 只保留开始日期之后的数据用于回测
    data = data.iloc[start_idx:].copy()
    
    # 生成交易信号：只在交叉时产生信号
    # 上穿：short_ma从下方穿过long_ma -> signal = 1 (买入)
    # 下穿：short_ma从上方穿过long_ma -> signal = -1 (卖出)
    # 其他时候：signal = 0 (不操作)
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
    
    # 计算持仓状态：signal只在交叉时产生，需要前向填充来保持持仓状态
    # signal=1时全部买入 -> position=1（持仓）
    # signal=-1时全部卖出 -> position=0（空仓）
    # signal=0时不操作 -> 保持前一个position的值
    # 先将signal转换为position：1->1, -1->0, 0->NaN
    data['position'] = data['signal'].replace({1: 1, -1: 0, 0: np.nan}).ffill().fillna(0).astype(int)
    
    # 计算每日收益（简化版：持仓期间不计算每日收益，只在卖出时计算一次总收益）
    # 买入时使用open价格，卖出时使用close价格
    # 买入当天：收益 = 0
    # 持仓期间：收益 = 0（不计算每日变化）
    # 卖出当天：收益 = (卖出close - 买入open) / 买入open
    data['returns'] = 0.0
    current_buy_open = None  # 记录当前持仓的买入open价格
    current_buy_idx = None  # 记录买入的索引位置
    
    for i in range(len(data)):
        # 如果当天有买入信号，使用open价格买入，当天收益为0
        if data['signal'].iloc[i] == 1:
            current_buy_open = data['open'].iloc[i]
            current_buy_idx = i
            data.iloc[i, data.columns.get_loc('returns')] = 0.0  # 买入当天收益为0
        
        # 如果当天有卖出信号，使用close价格卖出
        elif data['signal'].iloc[i] == -1:
            if current_buy_open is not None:
                # 卖出当天的收益：基于买入open到卖出close的变化
                sell_price = data['close'].iloc[i]
                trade_return = (sell_price - current_buy_open) / current_buy_open
                data.iloc[i, data.columns.get_loc('returns')] = trade_return
                current_buy_open = None
                current_buy_idx = None
            else:
                # 如果没有买入价格（理论上不应该发生），收益为0
                data.iloc[i, data.columns.get_loc('returns')] = 0.0
        
        # 如果有持仓但没有交易信号，收益为0（不计算每日变化）
        # returns保持为0.0，不需要修改
    
    # 检查最后一天是否还有持仓，如果有则强制平仓
    if len(data) > 0 and current_buy_open is not None:
        last_idx = len(data) - 1
        # 在最后一天添加卖出信号
        data.iloc[last_idx, data.columns.get_loc('signal')] = -1
        # 计算强制平仓的收益
        sell_price = data['close'].iloc[last_idx]
        trade_return = (sell_price - current_buy_open) / current_buy_open
        data.iloc[last_idx, data.columns.get_loc('returns')] = trade_return
        if verbose:
            print(f"\n警告：策略结束时仍有持仓，已在最后一天强制平仓")
            print(f"  买入价格: ${current_buy_open:,.2f}")
            print(f"  卖出价格: ${sell_price:,.2f}")
            print(f"  收益率: {trade_return:.2%}")
    
    # 重新计算持仓状态（因为可能添加了强制平仓信号）
    data['position'] = data['signal'].replace({1: 1, -1: 0, 0: np.nan}).ffill().fillna(0).astype(int)
    
    # 计算累计收益
    # 注意：由于持仓期间收益为0，cumulative_returns在持仓期间保持不变，只在卖出时变化
    data['cumulative_returns'] = (1 + data['returns'].fillna(0)).cumprod()
    
    if verbose:
        # 打印每笔交易的详细信息
        print_trade_details(data, base_equity=100000.0, verbose=verbose)
        
        # 计算并打印性能指标
        metrics = calculate_performance_metrics(data, verbose=verbose)
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
        
        # 获取指定交易对的价格数据（需要open和close来计算买入卖出价格）
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
            # 格式化profit_factor（如果是无穷大，显示为"∞"或很大的数）
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
