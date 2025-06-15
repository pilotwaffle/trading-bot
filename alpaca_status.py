from fastapi import APIRouter
from alpaca_trade_api.rest import REST, APIError
import os

router = APIRouter()

# You probably already have these somewhere in your config
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

@router.get("/api/alpaca_status")
async def alpaca_status():
    try:
        api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
        account = api.get_account()
        return {
            "status": "connected",
            "account_status": account.status,
            "account_id": account.id
        }
    except APIError as e:
        return {
            "status": "error",
            "detail": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": "Unexpected error: " + str(e)
        }