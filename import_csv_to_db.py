import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'trading_bot.db'
CSV_DIR = 'E:/Trade Chat Bot/'
CSV_PREFIX = 'historical_prices_'

COIN_SYMBOLS = {
    'bitcoin': 'BTC/USD',
    'ethereum': 'ETH/USD',
    'cardano': 'ADA/USD',
    'shiba-inu': 'SHIB/USD',
    'ripple': 'XRP/USD',
    'solana': 'SOL/USD',
    'sui': 'SUI/USD',
    # Add more mappings if needed
}

def import_csvs_to_db():
    conn = sqlite3.connect(DB_PATH)
    for coin_id, symbol in COIN_SYMBOLS.items():
        csv_path = os.path.join(CSV_DIR, f'{CSV_PREFIX}{coin_id}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Add dummy values for volume/change_24h if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        if 'change_24h' not in df.columns:
            df['change_24h'] = 0
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR IGNORE INTO market_data
                (symbol, price, volume, change_24h, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                float(row['price']),
                float(row['volume']),
                float(row['change_24h']),
                row['timestamp'].isoformat()
            ))
        print(f"Imported {len(df)} rows for {symbol}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import_csvs_to_db()