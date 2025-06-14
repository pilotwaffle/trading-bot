import requests
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch_historical_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_historical_data(
    coin_ids=['bitcoin', 'ethereum', 'cardano', 'solana', 'sui', 'shiba-inu', 'ripple'],
    vs_currency='usd',
    days=30,
    retries=3,
    delay=5,
    request_interval=15   # Increased to avoid CoinGecko rate limits
):
    """
    Fetch historical price data from CoinGecko for multiple coins and save to CSVs.
    
    Args:
        coin_ids (list): List of cryptocurrency IDs (e.g., ['bitcoin', 'ethereum']).
        vs_currency (str): Currency to price against (e.g., 'usd').
        days (int): Number of days of historical data.
        retries (int): Number of retry attempts for API failures.
        delay (int): Delay between retries in seconds.
        request_interval (int): Delay between requests to avoid rate limits.
    
    Returns:
        bool: True if all fetches succeeded, False if any failed.
    """
    success = True
    base_path = 'E:/Trade Chat Bot/historical_prices_'

    for coin_id in coin_ids:
        for attempt in range(retries):
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {'vs_currency': vs_currency, 'days': days}
                logger.info(f"Fetching data for {coin_id}/{vs_currency} over {days} days (Attempt {attempt + 1}/{retries})")
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if 'prices' not in data:
                    logger.error(f"No 'prices' key in response for {coin_id}. Response: {data}")
                    raise KeyError(f"No 'prices' key in API response for {coin_id}")
                
                prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')

                # Optionally, include volume if present
                if 'total_volumes' in data and len(data['total_volumes']) == len(prices):
                    prices['volume'] = [v[1] for v in data['total_volumes']]
                else:
                    prices['volume'] = 0

                output_path = f"{base_path}{coin_id}.csv"
                prices.to_csv(output_path, index=False)
                logger.info(f"Saved historical data for {coin_id} to {output_path}")
                break  # Success, move to next coin

            except requests.exceptions.HTTPError as e:
                status = getattr(response, 'status_code', None)
                # 429 = Too Many Requests (rate limit)
                if status == 429 or 'rate limit' in str(e).lower():
                    logger.warning(f"Rate limit hit for {coin_id}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(delay)
                    continue
                logger.error(f"HTTP error for {coin_id}: {e}")
                success = False
                break
            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                logger.error(f"Error fetching data for {coin_id}: {e}")
                success = False
                break

        # Avoid rate limits between coins (increase if you still hit limits)
        time.sleep(request_interval)
    
    return success

if __name__ == "__main__":
    fetch_historical_data()