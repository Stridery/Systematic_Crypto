import os
import time
from datetime import datetime, timedelta
import requests
import psycopg2
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent  # backend -> project_root
load_dotenv(BASE_DIR / ".env.local")


class PriceCollector:
    """
    Price Data Collector Class
    
    Handles fetching historical price data from Binance.us API
    and saving to PostgreSQL database.
    """
    
    def __init__(self):
        """Initialize the Price Collector"""
        # Binance.us API endpoints
        self.base_url = "https://api.binance.us"
        self.klines_endpoint = "/api/v3/klines"
        
        # 数据库连接配置
        self.db_params = {
            'dbname': os.getenv('DB_NAME', 'trading_db'),
            'user': os.getenv('DB_USER', 'zhenghaoyou'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432'))
        }
        
        # 交易对配置
        self.trading_pairs = [
            'BTCUSDT',  # Bitcoin
            'ETHUSDT',  # Ethereum
            'XRPUSDT',  # XRP
            'SOLUSDT',  # Solana
            'BNBUSDT',  # Binance Coin
            'DOGEUSDT',  # Dogecoin
            'LTCUSDT'   # Litecoin
        ]

    def connect_db(self) -> psycopg2.extensions.connection:
        """Establish database connection"""
        return psycopg2.connect(**self.db_params)

    def get_klines_data(self, symbol: str, interval: str = '1d') -> List[Dict[str, Any]]:
        """
        Fetch K-line data from Binance.us
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: K-line interval (default: '1d' for daily)
            
        Returns:
            List of dictionaries containing price data
        """
        try:
            # 获取最近3年的数据
            three_years_ago = int((datetime.now() - timedelta(days=1095)).timestamp() * 1000)
            
            # 由于API限制，我们需要分批获取数据
            all_data = []
            current_start = three_years_ago
            
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
                
                if response.status_code != 200:
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
            
            return all_data
        except Exception as e:
            return []

    def save_to_db(self, data: List[Dict[str, Any]], trading_pair: str, conn: psycopg2.extensions.connection) -> None:
        """
        Save price data to database
        
        Args:
            data: List of price data dictionaries
            trading_pair: Trading pair name (e.g., 'BTCUSD')
            conn: Database connection object
        """
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
        except Exception as e:
            conn.rollback()
        finally:
            cursor.close()

    def collect_data(self, clear_existing: bool = True) -> None:
        """
        Main data collection function
        
        Args:
            clear_existing: Whether to clear existing data before collecting
        """
        conn = self.connect_db()
        try:
            if clear_existing:
                cursor = conn.cursor()
                cursor.execute("TRUNCATE TABLE price_data;")
                conn.commit()
                cursor.close()

            for pair in self.trading_pairs:
                data = self.get_klines_data(pair)
                if data:
                    # 将USDT转换为USD
                    trading_pair = pair.replace('USDT', 'USD')
                    self.save_to_db(data, trading_pair, conn)
                time.sleep(2)  # Binance API 限制
        finally:
            conn.close()


def main():
    """Main entry point"""
    collector = PriceCollector()
    collector.collect_data()


if __name__ == "__main__":
    main()
