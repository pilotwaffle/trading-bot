import os
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

# --- CoinMarketCap Proxy Route ---
@router.get("/api/coinmarketcap-prices")
def coinmarketcap_prices():
    """
    Proxy endpoint to fetch crypto prices from CoinMarketCap.
    You must set your CoinMarketCap API key in the environment as COINMARKETCAP_API_KEY.
    """
    API_KEY = os.getenv("COINMARKETCAP_API_KEY")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="CoinMarketCap API key is not set")
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {
        "start": "1",
        "limit": "10",
        "convert": "USD"
    }
    headers = {
        "X-CMC_PRO_API_KEY": API_KEY,
        "Accepts": "application/json"
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
    # Transform to CoinGecko-like structure for easier frontend use
    data = resp.json()
    output = []
    for coin in data.get("data", []):
        output.append({
            "symbol": coin["symbol"],
            "name": coin["name"],
            "price": coin["quote"]["USD"]["price"],
            "change_24h": coin["quote"]["USD"]["percent_change_24h"]
        })
    return output

# --- Alpaca API Connection Status Route ---
@router.get("/api/alpaca-status")
def alpaca_status():
    """
    Checks connection to Alpaca API (paper account).
    Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.
    """
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    base_url = "https://paper-api.alpaca.markets"
    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="Alpaca API keys not set")
    account_url = f"{base_url}/v2/account"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret
    }
    try:
        resp = requests.get(account_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "online",
                "account_status": data.get("status", ""),
                "account_number": data.get("account_number", ""),
                "buying_power": data.get("buying_power", ""),
            }
        else:
            return JSONResponse(status_code=resp.status_code, content={"status": "offline", "error": resp.text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "offline", "error": str(e)})