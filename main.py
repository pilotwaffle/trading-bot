import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# Import our trading engine
from trading_engine import get_trading_engine, MarketData, Trade, Position

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Industrial Crypto Trading Bot", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
for directory in ["static", "templates", "logs", "models", "data"]:
    os.makedirs(directory, exist_ok=True)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# Authentication
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = os.getenv("APP_USER_ID", "admin")
    pw = os.getenv("APP_PASSWORD", "admin123")
    if credentials.username != user or credentials.password != pw:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

# Initialize trading engine
engine, chat_bot = get_trading_engine()

# Pydantic models
class TradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    side: str = Field(..., pattern="^(buy|sell)$")
    amount: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)

class StrategyConfig(BaseModel):
    name: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)

class MLTrainingRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    days: int = Field(default=30, ge=1, le=365)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:  # Create a copy to avoid modification during iteration
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.disconnect(connection)

manager = ConnectionManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, credentials: HTTPBasicCredentials = Depends(authenticate)):
    return templates.TemplateResponse("industrial_dashboard.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_running": engine.running,
        "market_data_symbols": len(engine.current_market_data),
        "active_positions": len(engine.positions),
        "ml_model_trained": engine.ml_model.is_trained
    }

# Real-time market data endpoints
@app.get("/api/market-data")
async def get_market_data(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get current market data for all tracked cryptocurrencies"""
    market_data = {}
    for symbol, data in engine.current_market_data.items():
        market_data[symbol] = {
            "symbol": data.symbol,
            "price": data.price,
            "volume": data.volume,
            "change_24h": data.change_24h,
            "timestamp": data.timestamp.isoformat()
        }
    
    return {"market_data": market_data, "last_updated": datetime.now().isoformat()}

@app.get("/api/crypto-rankings")
async def get_crypto_rankings(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get top cryptocurrencies with rankings"""
    rankings = []
    for symbol, data in engine.current_market_data.items():
        rankings.append({
            "symbol": symbol,
            "name": symbol.replace('/USD', ''),
            "price": data.price,
            "change_24h": data.change_24h,
            "volume": data.volume,
            "market_cap": data.price * data.volume  # Simplified calculation
        })
    
    # Sort by market cap (descending)
    rankings.sort(key=lambda x: x["market_cap"], reverse=True)
    
    return {"rankings": rankings[:15]}  # Top 15

# Trading endpoints
@app.post("/api/trade")
async def execute_trade(trade_request: TradeRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Execute a manual trade"""
    try:
        symbol = trade_request.symbol.upper()
        
        if symbol not in engine.current_market_data:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found in market data")
        
        market_data = engine.current_market_data[symbol]
        
        if trade_request.side == "buy":
            # Simulate buy order
            if engine.balance >= trade_request.amount:
                quantity = trade_request.amount / (trade_request.price or market_data.price)
                
                # Create position
                position = Position(
                    symbol=symbol,
                    amount=quantity,
                    entry_price=trade_request.price or market_data.price,
                    current_price=market_data.price,
                    unrealized_pnl=0.0,
                    side='long'
                )
                
                engine.positions[symbol] = position
                engine.balance -= trade_request.amount
                
                # Broadcast update
                await manager.broadcast(json.dumps({
                    "type": "trade_executed",
                    "data": {
                        "symbol": symbol,
                        "side": trade_request.side,
                        "amount": quantity,
                        "price": trade_request.price or market_data.price
                    }
                }))
                
                return {"status": "success", "message": f"Buy order executed: {quantity:.6f} {symbol}"}
            else:
                raise HTTPException(status_code=400, detail="Insufficient balance")
                
        else:  # sell
            if symbol in engine.positions:
                position = engine.positions[symbol]
                sale_value = position.amount * (trade_request.price or market_data.price)
                profit_loss = (trade_request.price or market_data.price - position.entry_price) * position.amount
                
                engine.balance += sale_value
                del engine.positions[symbol]
                
                # Broadcast update
                await manager.broadcast(json.dumps({
                    "type": "trade_executed",
                    "data": {
                        "symbol": symbol,
                        "side": trade_request.side,
                        "amount": position.amount,
                        "price": trade_request.price or market_data.price,
                        "profit_loss": profit_loss
                    }
                }))
                
                return {"status": "success", "message": f"Sell order executed: {position.amount:.6f} {symbol}, P&L: ${profit_loss:.2f}"}
            else:
                raise HTTPException(status_code=400, detail=f"No position found for {symbol}")
                
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/positions")
async def get_positions(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get current positions"""
    positions = []
    for symbol, position in engine.positions.items():
        positions.append({
            "symbol": position.symbol,
            "amount": position.amount,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "side": position.side,
            "percentage_change": ((position.current_price - position.entry_price) / position.entry_price) * 100
        })
    
    return {"positions": positions}

@app.get("/api/performance")
async def get_performance(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get performance metrics"""
    metrics = engine.get_performance_metrics()
    
    # Calculate additional metrics
    initial_balance = 10000.0
    total_return = ((metrics['total_value'] - initial_balance) / initial_balance) * 100
    
    return {
        "total_value": metrics['total_value'],
        "cash_balance": metrics['cash_balance'],
        "unrealized_pnl": metrics['unrealized_pnl'],
        "total_profit": metrics['total_profit'],
        "total_return_percentage": total_return,
        "num_positions": metrics['num_positions'],
        "last_updated": datetime.now().isoformat()
    }

# Strategy management
@app.get("/api/strategies")
async def get_strategies(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get all trading strategies"""
    strategies = []
    for strategy in engine.strategies:
        strategies.append({
            "name": strategy.name,
            "enabled": strategy.enabled,
            "params": strategy.params
        })
    
    return {"strategies": strategies}

@app.post("/api/strategies/{strategy_name}/toggle")
async def toggle_strategy(strategy_name: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Enable/disable a trading strategy"""
    for strategy in engine.strategies:
        if strategy.name.lower() == strategy_name.lower():
            strategy.enabled = not strategy.enabled
            return {"status": "success", "enabled": strategy.enabled}
    
    raise HTTPException(status_code=404, detail="Strategy not found")

# ML Training endpoints
@app.post("/api/ml/train")
async def train_ml_model(request: MLTrainingRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Train the ML model"""
    try:
        # Start training in background
        import threading
        threading.Thread(target=engine.train_ml_model, daemon=True).start()
        
        # Broadcast training started
        await manager.broadcast(json.dumps({
            "type": "ml_training_started",
            "data": {"symbols": request.symbols, "days": request.days}
        }))
        
        return {"status": "success", "message": "ML model training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/status")
async def get_ml_status(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get ML model status"""
    return {
        "is_trained": engine.ml_model.is_trained,
        "last_training": "N/A",  # You can add this tracking
        "model_type": "Random Forest",
        "features_count": len(engine.ml_model.features) if engine.ml_model.features else 0
    }

# Bot control
@app.post("/api/bot/start")
async def start_bot(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Start automated trading"""
    engine.start()
    await manager.broadcast(json.dumps({"type": "bot_started", "data": {}}))
    return {"status": "success", "message": "Trading bot started"}

@app.post("/api/bot/stop")
async def stop_bot(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Stop automated trading"""
    engine.stop()
    await manager.broadcast(json.dumps({"type": "bot_stopped", "data": {}}))
    return {"status": "success", "message": "Trading bot stopped"}

@app.get("/api/bot/status")
async def get_bot_status():
    """Get bot status (no auth required for health checks)"""
    return {
        "running": engine.running,
        "uptime": "N/A",  # You can add uptime tracking
        "last_trade": "N/A",  # You can add last trade tracking
        "strategies_active": sum(1 for s in engine.strategies if s.enabled)
    }

# Chat interface
@app.post("/api/chat")
async def chat_message(message: ChatMessage, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Process chat message"""
    try:
        response = chat_bot.process_message(message.message)
        
        # Save to database
        with engine.db.db_path as db_path:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO chat_messages (user_id, message, response)
                    VALUES (?, ?, ?)
                """, (credentials.username, message.message, response))
        
        # Broadcast chat message
        await manager.broadcast(json.dumps({
            "type": "chat_message",
            "data": {
                "user": credentials.username,
                "message": message.message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        }))
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error processing message: {e}"}

@app.get("/api/chat/history")
async def get_chat_history(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get chat history"""
    try:
        import sqlite3
        with sqlite3.connect(engine.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, message, response, timestamp
                FROM chat_messages
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "user": row[0],
                    "message": row[1],
                    "response": row[2],
                    "timestamp": row[3]
                })
            
            return {"messages": messages}
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        return {"messages": []}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Historical data endpoints
@app.get("/api/historical/{symbol}")
async def get_historical_data(symbol: str, days: int = 7, credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get historical data for a symbol"""
    try:
        data = engine.db.get_historical_data(symbol.replace('/', ''), days)
        
        if data.empty:
            return {"error": "No historical data found"}
        
        # Convert to list of dictionaries
        result = []
        for timestamp, row in data.iterrows():
            result.append({
                "timestamp": timestamp.isoformat(),
                "price": row['close'],
                "volume": row.get('volume', 0)
            })
        
        return {"symbol": symbol, "data": result}
    except Exception as e:
        logger.error(f"Historical data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics/summary")
async def get_analytics_summary(credentials: HTTPBasicCredentials = Depends(authenticate)):
    """Get trading analytics summary"""
    try:
        import sqlite3
        with sqlite3.connect(engine.db.db_path) as conn:
            # Get trade statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(profit_loss) as total_pnl,
                    AVG(profit_loss) as avg_pnl,
                    MAX(profit_loss) as best_trade,
                    MIN(profit_loss) as worst_trade
                FROM trades
                WHERE timestamp > datetime('now', '-30 days')
            """)
            
            trade_stats = cursor.fetchone()
            
            win_rate = 0
            if trade_stats[0] > 0:
                win_rate = (trade_stats[1] / trade_stats[0]) * 100
            
            return {
                "total_trades": trade_stats[0] or 0,
                "winning_trades": trade_stats[1] or 0,
                "win_rate": win_rate,
                "total_pnl": trade_stats[2] or 0,
                "average_pnl": trade_stats[3] or 0,
                "best_trade": trade_stats[4] or 0,
                "worst_trade": trade_stats[5] or 0,
                "period": "30 days"
            }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "average_pnl": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "period": "30 days"
        }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the trading engine on startup"""
    logger.info("Starting industrial trading bot...")
    
    # Start the trading engine
    engine.start()
    
    # Start periodic market data broadcasting
    import asyncio
    asyncio.create_task(broadcast_market_data())

async def broadcast_market_data():
    """Broadcast market data updates to all connected clients"""
    while True:
        try:
            if engine.current_market_data:
                market_update = {
                    "type": "market_data_update",
                    "data": {
                        symbol: {
                            "symbol": data.symbol,
                            "price": data.price,
                            "change_24h": data.change_24h,
                            "volume": data.volume
                        }
                        for symbol, data in engine.current_market_data.items()
                    }
                }
                
                await manager.broadcast(json.dumps(market_update))
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)