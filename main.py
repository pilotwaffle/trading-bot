#!/usr/bin/env python3
"""
Industrial Crypto Trading Bot - FastAPI Unified Main
- Merges CLI, backtesting, optimization, and FastAPI dashboard/API
- Modern entry point for bot, analytics, dashboard, API, and CLI/backtest
"""

import sys
import os
import argparse
import asyncio
import signal
import logging
from pathlib import Path
from datetime import datetime
import traceback

# --- Directory Bootstrapping ---
for directory in ["static", "templates", "logs", "models", "data"]:
    os.makedirs(directory, exist_ok=True)

# --- CLI Args & Config Loader ---
def parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Industrial Crypto Trading Bot (Web/CLI Hybrid)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run FastAPI dashboard with default config
  %(prog)s -c custom.yaml     # Run with custom config
  %(prog)s --mode paper       # Override bot mode
  %(prog)s --backtest         # Run backtesting
  %(prog)s --optimize         # Run optimization
        """
    )
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to configuration file (default: config.yaml)')
    parser.add_argument('-m', '--mode', choices=['live', 'paper', 'backtest'], help='Override trading mode')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting mode')
    parser.add_argument('--optimize', action='store_true', help='Run strategy optimization')
    parser.add_argument('--strategy', help='Specific strategy to backtest/optimize')
    parser.add_argument('--symbol', help='Specific symbol to trade/backtest')
    parser.add_argument('--start-date', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
    return parser.parse_args()

# --- Logging Setup ---
def setup_logging(log_file='logs/trading_bot.log', log_level='INFO'):
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# --- ASCII Art Banner ---
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ____                  _          ____        _            ║
║    / ___|_ __ _   _ _ __ | |_ ___   | __ )  ___ | |_          ║
║   | |   | '__| | | | '_ \| __/ _ \  |  _ \ / _ \| __|         ║
║   | |___| |  | |_| | |_) | || (_) | | |_) | (_) | |_          ║
║    \____|_|   \__, | .__/ \__\___/  |____/ \___/ \__|         ║
║               |___/|_|                                         ║
║                                                               ║
║           Industrial Crypto Trading Bot v2.0                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

# --- FastAPI Imports and Setup ---
from fastapi import FastAPI, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Load Environment Variables ---
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Industrial Crypto Trading Bot",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# --- Authentication (for API endpoints) ---
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = os.getenv("APP_USER_ID", "admin")
    pw = os.getenv("APP_PASSWORD", "admin123")
    if credentials.username != user or credentials.password != pw:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

# --- Import Trading Engine, Config, DB, etc. ---
from trading_engine import get_trading_engine, MarketData, Trade, Position
from config_manager import ConfigManager
from database import Database

# --- CLI/Config/Engine Boot ---
args = parse_cli_args()
config_manager = ConfigManager(args.config)
config = config_manager.config
setup_logging(
    log_file=config.get('logging', {}).get('file', 'logs/trading_bot.log'),
    log_level=config.get('logging', {}).get('level', 'INFO')
)
logger = logging.getLogger("crypto-bot")
print(BANNER)
logger.info("Configuration loaded")

# --- Database and Engine ---
database = Database(config['database']['path'])
database.initialize()
logger.info("Database initialized")
engine, chat_bot = get_trading_engine()
logger.info("Trading engine created")

# --- Trading Bot Control (for CLI/async main) ---
async def start_engine():
    logger.info(f"Starting Trading Bot in {config['bot']['mode']} mode")
    await engine.initialize()
    await engine.start()
    logger.info("Trading Bot started")

async def stop_engine():
    logger.info("Stopping Trading Bot...")
    await engine.stop()
    database.close()
    logger.info("Trading Bot stopped")

# --- Backtesting/Optimization logic from legacy main.py ---
async def run_backtest():
    from backtesting import BacktestEngine
    from datetime import datetime, timedelta
    print("\n" + "="*60)
    print("BACKTESTING MODE")
    print("="*60 + "\n")
    backtest_engine = BacktestEngine(database=database, risk_manager=engine.risk_manager)
    strategy_name = args.strategy or config_manager.get_enabled_strategies()[0]['name']
    symbol = args.symbol or 'BTC/USDT'
    # Date parsing
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=365)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    print(f"Strategy: {strategy_name}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${config['bot']['initial_capital']:,.2f}")
    print("\nRunning backtest...\n")
    strategy = engine._load_strategy(config_manager.get_strategy_config(strategy_name))
    result = backtest_engine.backtest(
        strategy=strategy,
        symbol=symbol,
        timeframe='1h',
        start_date=start_date,
        end_date=end_date,
        initial_balance=config['bot']['initial_capital']
    )
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"\nTotal Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Average Win: ${result.avg_win:.2f}")
    print(f"Average Loss: ${result.avg_loss:.2f}")
    report_path = f"backtest_report_{strategy_name}_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = backtest_engine.generate_report(result, report_path)
    print(f"\nDetailed report saved to: {report_path}")

async def run_optimization():
    from backtesting import BacktestEngine
    from datetime import datetime, timedelta
    print("\n" + "="*60)
    print("STRATEGY OPTIMIZATION")
    print("="*60 + "\n")
    backtest_engine = BacktestEngine(database=database, risk_manager=engine.risk_manager)
    strategy_name = args.strategy or config_manager.get_enabled_strategies()[0]['name']
    symbol = args.symbol or 'BTC/USDT'
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=180)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    print(f"Strategy: {strategy_name}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("\nOptimizing parameters...\n")
    strategy_module = __import__(f'strategies.{strategy_name.lower()}', fromlist=[strategy_name])
    strategy_class = getattr(strategy_module, strategy_name)
    param_ranges = strategy_class().get_parameter_ranges()
    results = backtest_engine.optimize_parameters(
        strategy_class=strategy_class,
        symbol=symbol,
        timeframe='1h',
        start_date=start_date,
        end_date=end_date,
        param_ranges=param_ranges,
        metric='sharpe_ratio'
    )
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest Sharpe Ratio: {results['best_metric']:.2f}")
    print("\nTop 5 Parameter Sets:")
    sorted_results = sorted(results['all_results'], key=lambda x: x['result'].sharpe_ratio, reverse=True)[:5]
    for i, res in enumerate(sorted_results, 1):
        print(f"\n{i}. Sharpe: {res['result'].sharpe_ratio:.2f}, "
              f"Return: {res['result'].total_return:.2%}")
        print(f"   Parameters: {res['params']}")

# --- FastAPI API/Routes (from your file 2, unchanged) ---
# (Paste all routes from your second file here, unchanged)

# --- WebSocket ConnectionManager, API, and Routers ---
# (Paste all classes and API endpoints from your second file here, unchanged)

# --- Startup Event: Bot Start & Market Data Broadcast ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting industrial trading bot...")
    await start_engine()
    import asyncio
    asyncio.create_task(broadcast_market_data())

async def broadcast_market_data():
    import asyncio
    from fastapi import WebSocket
    from app import manager  # ensure manager is available/imported!
    while True:
        try:
            if getattr(engine, "current_market_data", {}):
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
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(10)

# --- Entrypoint: CLI vs FastAPI ---
if __name__ == "__main__":
    if args.backtest:
        asyncio.run(run_backtest())
    elif args.optimize:
        asyncio.run(run_optimization())
    else:
        import uvicorn
        # Use config API port if set, else fallback to 8000
        port = int(config.get('api', {}).get('port', 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)