from dotenv import load_dotenv
load_dotenv()

import os
import alpaca_trade_api as tradeapi

# Prefer official Alpaca env variables, fallback to your custom ones for compatibility
API_KEY = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_PAPER_API_KEY')
API_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_PAPER_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_PAPER_BASE_URL', 'https://paper-api.alpaca.markets')

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def submit_order(symbol, qty, side, type='market', time_in_force='gtc'):
    try:
        # For crypto, Alpaca expects e.g. 'BTCUSD', not 'BTC/USD'
        symbol = symbol.replace('/', '')
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        return order
    except Exception as e:
        print(f"Alpaca order error: {e}")
        return None