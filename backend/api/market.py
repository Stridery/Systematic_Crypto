from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
import os

router = APIRouter()

# Database Configuration
DB_NAME = os.getenv('DB_NAME', 'trading_db')
DB_USER = os.getenv('DB_USER', 'zhenghaoyou')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))


@router.get("/market-metrics")
def get_market_metrics():
    """
    Connects to the database and fetches the market metrics for the last hour.
    """
    conn = None
    try:
        # Step 1: Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            sslmode='disable'
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Step 2: Fetch all records from the last hour
        cur.execute("""
            SELECT timestamp, volatility, hype_index
            FROM market_metrics
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp ASC;
        """)
        rows = cur.fetchall()
        cur.close()

        if not rows:
             raise HTTPException(
                status_code=404,
                detail="No market metrics found for the last hour. Please run the population script."
            )

        # Step 3: Format the database rows into the required list of dictionaries
        data = [
            {
                "timestamp": row['timestamp'].isoformat().replace('+00:00', 'Z'),
                "volatility": row['volatility'],
                "hypeIndex": row['hype_index'] # Map the 'hype_index' column to the 'hypeIndex' key
            } for row in rows
        ]

        return data

    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {type(e).__name__} - {e}")

    finally:
        # Step 4: Always make sure the database connection is closed
        if conn:
            conn.close()