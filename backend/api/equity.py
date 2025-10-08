from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Database Configuration
DB_NAME = "trading_db"
DB_USER = "zhenghaoyou"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432

@router.get("/equity-curve")
def get_equity_curve():
    """
    Connects to the database, fetches the equity curve data,
    and formats it into the required nested JSON structure.
    """
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 首先检查表中是否有数据
        cur.execute("SELECT COUNT(*) FROM equity_curves")
        count = cur.fetchone()['count']
        logger.info(f"Found {count} records in equity_curves table")

        # 检查每个策略的数据
        cur.execute("""
            SELECT strategy_key, COUNT(*) as count 
            FROM equity_curves 
            GROUP BY strategy_key
        """)
        strategy_counts = cur.fetchall()
        for row in strategy_counts:
            logger.info(f"Strategy {row['strategy_key']}: {row['count']} records")

        # 获取实际数据
        cur.execute("""
            SELECT 
                strategy_key, 
                strategy_name, 
                strategy_color, 
                month, 
                equity,
                snapshot_time
            FROM equity_curves
            ORDER BY strategy_key, snapshot_time ASC;
        """)
        rows = cur.fetchall()
        logger.info(f"Fetched {len(rows)} rows from database")

        if not rows:
            logger.warning("No data found in equity_curves table")
            raise HTTPException(status_code=404, detail="Equity curve data not found in database.")

        # Process the data
        equity_data = defaultdict(lambda: {"name": "", "color": "", "data": []})
        
        for row in rows:
            key = row['strategy_key']
            if not equity_data[key]['name']:  # 只设置一次名称和颜色
                equity_data[key]['name'] = row['strategy_name']
                equity_data[key]['color'] = row['strategy_color']
            
            equity_data[key]['data'].append({
                "month": row['month'],
                "equity": float(row['equity'])
            })

        logger.info(f"Processed data for {len(equity_data)} strategies")
        return dict(equity_data)  # 转换defaultdict为普通dict

    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")
    except Exception as e:
        logger.error(f"Error in get_equity_curve: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {type(e).__name__} - {e}")
    finally:
        if conn:
            conn.close()