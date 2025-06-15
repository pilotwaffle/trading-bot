from fastapi import APIRouter

router = APIRouter()

@router.get("/api/alpaca_status")
async def alpaca_status():
    return {"status": "test OK"}