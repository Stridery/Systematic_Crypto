import os
import sys
import time
import random
from datetime import datetime, timedelta
import logging
import requests
import psycopg2
from typing import List, Dict, Any
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PriceCollector:
    def __init__(self):
        # Binance.us API endpoints
        self.base_url = "https://api.binance.us"
        self.klines_endpoint = "/api/v3/klines"
        
        # 数据库连接配置
        self.db_params = {
            'dbname': os.getenv('DB_NAME', 'trading_db'),
            'user': os.getenv('DB_USER', 'zhenghaoyou'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
        # 交易对配置
        self.trading_pairs = [
            'BTCUSDT',  # Bitcoin
            'ETHUSDT',  # Ethereum
            'LTCUSDT',  # Litecoin
            'DOGEUSDT'  # Dogecoin
        ]

    def connect_db(self) -> psycopg2.extensions.connection:
        """建立数据库连接"""
        try:
            conn = psycopg2.connect(**self.db_params)
            logger.info("Successfully connected to database")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def get_klines_data(self, symbol: str, interval: str = '1d') -> List[Dict[str, Any]]:
        """从Binance.us获取K线数据"""
        try:
            # 获取最近4年的数据
            three_years_ago = int((datetime.now() - timedelta(days=1460)).timestamp() * 1000)
            
            # 由于API限制，我们需要分批获取数据
            all_data = []
            current_start = three_years_ago
            
            logger.info(f"Starting data collection from {datetime.fromtimestamp(three_years_ago/1000)}")
            
            while True:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'limit': 1000  # Binance API 每次最多返回1000条数据
                }
            
                headers = {
                    'User-Agent': 'Mozilla/5.0'
                }
                
                response = requests.get(
                    f"{self.base_url}{self.klines_endpoint}",
                    params=params,
                    headers=headers
                )
                
                # 打印请求URL和响应状态码，用于调试
                logger.info(f"Request URL: {response.url}")
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Error response: {response.text}")
                    break
                
                response.raise_for_status()
                
                # Binance K线数据格式转换为我们需要的格式
                klines = response.json()
                if not klines:  # 如果没有更多数据，退出循环
                    break
                    
                for k in klines:
                    all_data.append({
                        'timestamp': datetime.fromtimestamp(k[0] / 1000),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
                
                # 更新开始时间为最后一条数据的时间 + 1
                current_start = klines[-1][0] + 1
                
                # 如果已经到达当前时间，退出循环
                if current_start >= int(datetime.now().timestamp() * 1000):
                    break
                    
                # 避免触发API限制
                time.sleep(1)
                
                # 打印进度
                latest_time = datetime.fromtimestamp(klines[-1][0] / 1000)
                logger.info(f"Progress: Collected data up to {latest_time}")
            
            logger.info(f"Successfully fetched {len(all_data)} records for {symbol}")
            return all_data
        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            return []

    def save_to_db(self, data: List[Dict[str, Any]], trading_pair: str, conn: psycopg2.extensions.connection) -> None:
        """将数据保存到price_data表"""
        if not data:
            return
        
        cursor = conn.cursor()
        try:
            # 保存原始价格数据到price_data表
            args = ','.join(cursor.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s)", (
                item['timestamp'],
                trading_pair,
                'BINANCE_US',
                float(item['open']),
                float(item['high']),
                float(item['low']),
                float(item['close']),
                float(item['volume']),
                datetime.now()
            )).decode('utf-8') for item in data)
            
            cursor.execute(f"""
                INSERT INTO price_data (
                    timestamp, trading_pair, exchange,
                    open, high, low, close, volume, created_at
                )
                VALUES {args}
                ON CONFLICT (timestamp, trading_pair, exchange) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """)
            conn.commit()
            logger.info(f"Successfully saved {len(data)} records to price_data table")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving data to database: {e}")
        finally:
            cursor.close()

    def collect_data(self, clear_existing: bool = True) -> None:
        """主要数据收集函数"""
        conn = self.connect_db()
        try:
            if clear_existing:
                cursor = conn.cursor()
                logger.info("Clearing existing price data...")
                cursor.execute("TRUNCATE TABLE price_data;")
                conn.commit()
                logger.info("Existing price data cleared.")
                cursor.close()

            for pair in self.trading_pairs:
                logger.info(f"Collecting data for {pair}")
                data = self.get_klines_data(pair)
                if data:
                    # 将USDT转换为USD
                    trading_pair = pair.replace('USDT', 'USD')
                    self.save_to_db(data, trading_pair, conn)
                time.sleep(2)  # Binance API 限制
        finally:
            conn.close()

def main():
    collector = PriceCollector()
    collector.collect_data()

if __name__ == "__main__":
    main()