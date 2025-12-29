# util/price_collector.py
"""
价格数据收集器，用于从 Binance API 获取价格数据并保存到 CSV 文件
支持增量更新，只获取缺失的数据
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests


class PriceCollector:
    """
    价格数据收集器
    
    从 Binance API 获取 K 线数据，支持增量更新到 CSV 文件
    """
    
    def __init__(
        self,
        base_url: str = "https://api.binance.us",
        api_delay: float = 1.0,
    ):
        """
        初始化价格收集器
        
        Args:
            base_url: Binance API 基础 URL
            api_delay: API 请求之间的延迟（秒），避免触发限流
        """
        self.base_url = base_url
        self.klines_endpoint = "/api/v3/klines"
        self.api_delay = api_delay
    
    def get_klines_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        从 Binance API 获取 K 线数据
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            interval: K 线间隔，如 '1h', '1d', '1m'
            start_time: 开始时间（可选）
            end_time: 结束时间（可选，默认为当前时间）
            limit: 每次请求的最大数据条数（默认 1000，Binance API 限制）
            
        Returns:
            K 线数据列表，每个元素包含 timestamp, open, high, low, close, volume
        """
        all_data = []
        current_start = int(start_time.timestamp() * 1000) if start_time else None
        end_timestamp = int(end_time.timestamp() * 1000) if end_time else int(datetime.now().timestamp() * 1000)
        
        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
            }
            
            if current_start:
                params['startTime'] = current_start
            
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            
            try:
                response = requests.get(
                    f"{self.base_url}{self.klines_endpoint}",
                    params=params,
                    headers=headers,
                    timeout=10,
                )
                
                if response.status_code != 200:
                    print(f"[PriceCollector] API request failed with status {response.status_code}")
                    break
                
                response.raise_for_status()
                klines = response.json()
                
                if not klines:  # 如果没有更多数据，退出循环
                    break
                
                # 转换数据格式
                batch_data = []
                for k in klines:
                    k_timestamp = datetime.fromtimestamp(k[0] / 1000)
                    
                    # 如果超过结束时间，停止
                    if k[0] >= end_timestamp:
                        break
                    
                    batch_data.append({
                        'open_time_ms': int(k[0]),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'close_time_ms': int(k[6]),
                        'quote_asset_volume': float(k[7]),
                        'num_trades': int(k[8]),
                        'taker_buy_base': float(k[9]),
                        'taker_buy_quote': float(k[10]),
                    })
                
                all_data.extend(batch_data)
                
                # 如果这批数据已经到达或超过结束时间，退出
                if klines[-1][0] >= end_timestamp:
                    break
                
                # 更新开始时间为最后一条数据的时间 + 1
                current_start = klines[-1][0] + 1
                
                # 避免触发 API 限制
                time.sleep(self.api_delay)
                
            except requests.exceptions.RequestException as e:
                print(f"[PriceCollector] Request error: {e}")
                break
            except Exception as e:
                print(f"[PriceCollector] Error processing data: {e}")
                break
        
        return all_data
    
    def read_existing_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        读取现有的 CSV 数据文件
        
        Args:
            file_path: CSV 文件路径
            
        Returns:
            DataFrame 或 None（如果文件不存在）
        """
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path)
            # 确保 open_time_ms 是整数类型
            if 'open_time_ms' in df.columns:
                df['open_time_ms'] = df['open_time_ms'].astype('Int64')
            return df
        except Exception as e:
            print(f"[PriceCollector] Error reading existing file: {e}")
            return None
    
    def get_latest_timestamp(self, df: pd.DataFrame) -> Optional[datetime]:
        """
        获取 DataFrame 中最新数据的时间戳
        
        Args:
            df: 包含 open_time_ms 列的 DataFrame
            
        Returns:
            最新的时间戳，如果没有数据则返回 None
        """
        if df is None or len(df) == 0 or 'open_time_ms' not in df.columns:
            return None
        
        max_ms = df['open_time_ms'].max()
        if pd.isna(max_ms):
            return None
        
        return datetime.fromtimestamp(int(max_ms) / 1000)
    
    def is_data_up_to_date(
        self,
        file_path: Path,
        max_age_hours: int = 1,
    ) -> bool:
        """
        检查数据是否最新
        
        Args:
            file_path: CSV 文件路径
            max_age_hours: 数据最大允许年龄（小时），超过则认为需要更新
            
        Returns:
            True 如果数据是最新的，False 如果需要更新
        """
        df = self.read_existing_data(file_path)
        if df is None:
            return False
        
        latest_timestamp = self.get_latest_timestamp(df)
        if latest_timestamp is None:
            return False
        
        # 检查最新数据是否在允许的时间范围内
        age = datetime.now() - latest_timestamp
        
        # 根据 timeframe 调整允许的年龄
        # 对于 1h 数据，如果最新数据超过 2 小时，需要更新
        # 对于 1d 数据，如果最新数据超过 1 天，需要更新
        return age.total_seconds() < (max_age_hours * 3600)
    
    def update_data(
        self,
        symbol: str,
        interval: str,
        file_path: Path,
        max_age_hours: int = 1,
        force_update: bool = False,
    ) -> bool:
        """
        更新价格数据（增量更新）
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            interval: K 线间隔，如 '1h', '1d'
            file_path: CSV 文件保存路径
            max_age_hours: 数据最大允许年龄（小时）
            force_update: 是否强制更新（忽略数据是否最新）
            
        Returns:
            True 如果成功更新，False 如果跳过或失败
        """
        # 检查数据是否最新
        if not force_update and self.is_data_up_to_date(file_path, max_age_hours):
            print(f"[PriceCollector] Data is up to date: {file_path}")
            return False
        
        # 读取现有数据
        existing_df = self.read_existing_data(file_path)
        
        # 确定开始时间
        if existing_df is not None and len(existing_df) > 0:
            latest_timestamp = self.get_latest_timestamp(existing_df)
            if latest_timestamp:
                # 从最新数据之后开始获取（增量更新）
                start_time = latest_timestamp + timedelta(seconds=1)
                print(f"[PriceCollector] Incremental update: starting from {start_time}")
            else:
                # 如果无法确定最新时间，获取最近 3 年的数据
                start_time = datetime.now() - timedelta(days=1095)
                print(f"[PriceCollector] Full update: starting from {start_time}")
        else:
            # 如果文件不存在，获取最近 3 年的数据
            start_time = datetime.now() - timedelta(days=1095)
            print(f"[PriceCollector] New file: fetching 3 years of data from {start_time}")
        
        # 获取新数据
        print(f"[PriceCollector] Fetching data for {symbol} {interval}...")
        new_data = self.get_klines_data(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=datetime.now(),
        )
        
        if not new_data:
            print(f"[PriceCollector] No new data available")
            return False
        
        print(f"[PriceCollector] Fetched {len(new_data)} new records")
        
        # 转换为 DataFrame
        new_df = pd.DataFrame(new_data)
        
        # 定义列顺序（与 Binance API 返回顺序一致）
        column_order = [
            'open_time_ms',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'close_time_ms',
            'quote_asset_volume',
            'num_trades',
            'taker_buy_base',
            'taker_buy_quote',
        ]
        
        # 确保列顺序一致
        new_df = new_df[column_order]
        
        # 合并数据
        if existing_df is not None and len(existing_df) > 0:
            # 确保现有数据的列顺序也正确
            if set(existing_df.columns) == set(column_order):
                existing_df = existing_df[column_order]
            
            # 合并并去重（基于 open_time_ms）
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['open_time_ms'], keep='last')
            combined_df = combined_df.sort_values('open_time_ms').reset_index(drop=True)
        else:
            combined_df = new_df.sort_values('open_time_ms').reset_index(drop=True)
        
        # 确保最终 DataFrame 的列顺序正确
        combined_df = combined_df[column_order]
        
        # 保存到文件
        file_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(file_path, index=False)
        print(f"[PriceCollector] Saved {len(combined_df)} total records to {file_path}")
        
        return True
    
    def collect_data(
        self,
        symbol: str,
        interval: str,
        output_path: Path,
        max_age_hours: int = 1,
        force_update: bool = False,
    ) -> bool:
        """
        收集数据的便捷方法（别名，与 update_data 相同）
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            interval: K 线间隔，如 '1h', '1d'
            output_path: CSV 文件保存路径
            max_age_hours: 数据最大允许年龄（小时）
            force_update: 是否强制更新
            
        Returns:
            True 如果成功更新，False 如果跳过或失败
        """
        return self.update_data(
            symbol=symbol,
            interval=interval,
            file_path=output_path,
            max_age_hours=max_age_hours,
            force_update=force_update,
        )

