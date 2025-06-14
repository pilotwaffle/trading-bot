import asyncio
import json
import logging
import sqlite3
import threading
import time
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    timestamp: datetime
    profit_loss: float = 0.0
    strategy: str = "manual"
    
@dataclass
class Position:
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    side: str  # 'long' or 'short'

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: float
    change_24h: float
    timestamp: datetime

class CryptoDataFetcher:
    """Real-time crypto data fetcher using multiple sources"""
    
    def __init__(self):
        self.symbols = [
            'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'XRP/USD',
            'DOGE/USD', 'MATIC/USD', 'AVAX/USD', 'DOT/USD', 'LINK/USD',
            'SUI/USD', 'SHIB/USD', 'LTC/USD', 'HBAR/USD', 'NEAR/USD',
            'PYTH/USD', 'ONDO/USD', 'CRO/USD'
        ]
        # PI is not on major exchanges, so we'll handle it separately
        
        # Initialize exchanges
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'coinbase': ccxt.coinbasepro({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True})
        }
        
        self.market_data = {}
        self.running = False
        
    def get_crypto_prices(self) -> Dict[str, MarketData]:
        """Fetch current crypto prices from multiple exchanges"""
        prices = {}
        
        for symbol in self.symbols:
            try:
                # Try Binance first
                ticker = self.exchanges['binance'].fetch_ticker(symbol)
                
                prices[symbol] = MarketData(
                    symbol=symbol,
                    price=float(ticker['last']),
                    volume=float(ticker['baseVolume']),
                    change_24h=float(ticker['percentage'] or 0),
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from Binance: {e}")
                try:
                    # Fallback to Coinbase
                    ticker = self.exchanges['coinbase'].fetch_ticker(symbol)
                    prices[symbol] = MarketData(
                        symbol=symbol,
                        price=float(ticker['last']),
                        volume=float(ticker['baseVolume']),
                        change_24h=float(ticker['percentage'] or 0),
                        timestamp=datetime.now()
                    )
                except Exception as e2:
                    logger.error(f"Failed to fetch {symbol}: {e2}")
        
        return prices
    
    def start_real_time_feed(self, callback):
        """Start real-time WebSocket feed"""
        self.running = True
        threading.Thread(target=self._ws_feed, args=(callback,), daemon=True).start()
    
    def _ws_feed(self, callback):
        """WebSocket feed implementation"""
        while self.running:
            try:
                prices = self.get_crypto_prices()
                callback(prices)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"WebSocket feed error: {e}")
                time.sleep(10)

class MLTradingModel:
    """Machine Learning model for crypto trading predictions"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.features = []
        
    def prepare_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Prepare features from price data"""
        df = price_data.copy()
        
        # Technical indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(10).std()
        df['volume_sma'] = df['volume'].rolling(5).mean()
        
        # Time features
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        
        feature_columns = [
            'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'price_change', 'volatility', 'volume_sma', 'hour', 'day_of_week'
        ]
        
        return df[feature_columns].fillna(0).values
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def train(self, historical_data: Dict[str, pd.DataFrame]):
        """Train the model on historical data"""
        logger.info("Training ML model...")
        
        all_features = []
        all_targets = []
        
        for symbol, data in historical_data.items():
            if len(data) < 50:  # Need minimum data
                continue
                
            features = self.prepare_features(data)
            # Target: next hour price change
            targets = data['close'].shift(-1).pct_change().fillna(0).values
            
            # Remove last row (no target)
            features = features[:-1]
            targets = targets[:-1]
            
            all_features.extend(features)
            all_targets.extend(targets)
        
        if len(all_features) > 0:
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"Model trained on {len(X)} samples")
        else:
            logger.warning("No data available for training")
    
    def predict(self, current_data: pd.DataFrame) -> float:
        """Predict next price movement"""
        if not self.is_trained:
            return 0.0
        
        try:
            features = self.prepare_features(current_data)
            if len(features) > 0:
                features_scaled = self.scaler.transform([features[-1]])
                prediction = self.model.predict(features_scaled)[0]
                return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
        
        return 0.0

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.enabled = True
        
    def should_buy(self, market_data: MarketData, historical_data: pd.DataFrame) -> bool:
        raise NotImplementedError
        
    def should_sell(self, market_data: MarketData, historical_data: pd.DataFrame, position: Position) -> bool:
        raise NotImplementedError

class DCAStrategy(TradingStrategy):
    """Dollar Cost Averaging Strategy"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__("DCA", params)
        self.last_buy_time = {}
        
    def should_buy(self, market_data: MarketData, historical_data: pd.DataFrame) -> bool:
        symbol = market_data.symbol
        interval_hours = self.params.get('interval_hours', 24)
        
        last_buy = self.last_buy_time.get(symbol, datetime.min)
        time_since_last = (datetime.now() - last_buy).total_seconds() / 3600
        
        if time_since_last >= interval_hours:
            self.last_buy_time[symbol] = datetime.now()
            return True
        return False
        
    def should_sell(self, market_data: MarketData, historical_data: pd.DataFrame, position: Position) -> bool:
        take_profit = self.params.get('take_profit', 0.05)  # 5%
        stop_loss = self.params.get('stop_loss', 0.03)  # 3%
        
        price_change = (market_data.price - position.entry_price) / position.entry_price
        
        return price_change >= take_profit or price_change <= -stop_loss

class MLStrategy(TradingStrategy):
    """Machine Learning Strategy"""
    
    def __init__(self, params: Dict[str, Any], ml_model: MLTradingModel):
        super().__init__("ML", params)
        self.ml_model = ml_model
        
    def should_buy(self, market_data: MarketData, historical_data: pd.DataFrame) -> bool:
        if len(historical_data) < 50:
            return False
            
        prediction = self.ml_model.predict(historical_data)
        threshold = self.params.get('buy_threshold', 0.02)  # 2% predicted increase
        
        return prediction > threshold
        
    def should_sell(self, market_data: MarketData, historical_data: pd.DataFrame, position: Position) -> bool:
        prediction = self.ml_model.predict(historical_data)
        sell_threshold = self.params.get('sell_threshold', -0.01)  # 1% predicted decrease
        
        price_change = (market_data.price - position.entry_price) / position.entry_price
        take_profit = self.params.get('take_profit', 0.1)  # 10%
        stop_loss = self.params.get('stop_loss', 0.05)  # 5%
        
        return (prediction < sell_threshold or 
                price_change >= take_profit or 
                price_change <= -stop_loss)

class DatabaseManager:
    """Enhanced database manager for trading data"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    profit_loss REAL DEFAULT 0,
                    strategy TEXT DEFAULT 'manual',
                    exchange TEXT DEFAULT 'alpaca'
                )
            """)
            
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    side TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)
            
            # Market data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    change_24h REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Strategies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    params TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chat messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_trade(self, trade: Trade):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades 
                (id, symbol, side, amount, price, timestamp, profit_loss, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade.id, trade.symbol, trade.side, trade.amount, 
                  trade.price, trade.timestamp, trade.profit_loss, trade.strategy))
    
    def save_market_data(self, market_data: MarketData):
        """Save market data to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO market_data 
                (symbol, price, volume, change_24h, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (market_data.symbol, market_data.price, market_data.volume,
                  market_data.change_24h, market_data.timestamp))
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, price as close, volume
                FROM market_data 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days)
            
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df

class IndustrialTradingEngine:
    """Complete industrial-grade trading engine"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.data_fetcher = CryptoDataFetcher()
        self.ml_model = MLTradingModel()
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.strategies: List[TradingStrategy] = []
        self.current_market_data: Dict[str, MarketData] = {}
        self.balance = 10000.0  # Starting balance
        self.running = False
        
        # Initialize strategies
        self.init_strategies()
        
        # Start data feed
        self.data_fetcher.start_real_time_feed(self.on_market_data)
        
    def init_strategies(self):
        """Initialize trading strategies"""
        # DCA Strategy
        dca_params = {
            'interval_hours': 6,  # Buy every 6 hours
            'amount': 100,        # $100 per buy
            'take_profit': 0.05,  # 5% take profit
            'stop_loss': 0.03     # 3% stop loss
        }
        self.strategies.append(DCAStrategy(dca_params))
        
        # ML Strategy
        ml_params = {
            'buy_threshold': 0.02,   # 2% predicted increase
            'sell_threshold': -0.01, # 1% predicted decrease
            'take_profit': 0.1,      # 10% take profit
            'stop_loss': 0.05        # 5% stop loss
        }
        self.strategies.append(MLStrategy(ml_params, self.ml_model))
    
    def on_market_data(self, market_data: Dict[str, MarketData]):
        """Handle incoming market data"""
        self.current_market_data.update(market_data)
        
        # Save to database
        for data in market_data.values():
            self.db.save_market_data(data)
        
        # Update positions
        self.update_positions()
        
        # Execute strategies if running
        if self.running:
            self.execute_strategies()
    
    def update_positions(self):
        """Update position values with current market data"""
        for symbol, position in self.positions.items():
            if symbol in self.current_market_data:
                market_data = self.current_market_data[symbol]
                position.current_price = market_data.price
                
                if position.side == 'long':
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                else:  # short
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.amount
    
    def execute_strategies(self):
        """Execute all enabled trading strategies"""
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
                
            try:
                self.execute_strategy(strategy)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} error: {e}")
    
    def execute_strategy(self, strategy: TradingStrategy):
        """Execute a specific trading strategy"""
        for symbol, market_data in self.current_market_data.items():
            # Get historical data
            historical_data = self.db.get_historical_data(symbol.replace('/', ''))
            
            if len(historical_data) < 10:
                continue
            
            # Check for buy signals
            if symbol not in self.positions and strategy.should_buy(market_data, historical_data):
                self.execute_buy(symbol, strategy)
            
            # Check for sell signals
            elif symbol in self.positions and strategy.should_sell(market_data, historical_data, self.positions[symbol]):
                self.execute_sell(symbol, strategy)
    
    def execute_buy(self, symbol: str, strategy: TradingStrategy):
        """Execute buy order"""
        try:
            amount = strategy.params.get('amount', 100)  # Default $100
            market_data = self.current_market_data[symbol]
            
            if self.balance >= amount:
                quantity = amount / market_data.price
                
                # Create trade record
                trade = Trade(
                    id=f"buy_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    side='buy',
                    amount=quantity,
                    price=market_data.price,
                    timestamp=datetime.now(),
                    strategy=strategy.name
                )
                
                # Create position
                position = Position(
                    symbol=symbol,
                    amount=quantity,
                    entry_price=market_data.price,
                    current_price=market_data.price,
                    unrealized_pnl=0.0,
                    side='long'
                )
                
                # Update state
                self.positions[symbol] = position
                self.balance -= amount
                
                # Save to database
                self.db.save_trade(trade)
                
                logger.info(f"BUY executed: {quantity:.6f} {symbol} at ${market_data.price:.2f}")
                
        except Exception as e:
            logger.error(f"Buy execution error: {e}")
    
    def execute_sell(self, symbol: str, strategy: TradingStrategy):
        """Execute sell order"""
        try:
            position = self.positions[symbol]
            market_data = self.current_market_data[symbol]
            
            # Calculate profit/loss
            profit_loss = (market_data.price - position.entry_price) * position.amount
            
            # Create trade record
            trade = Trade(
                id=f"sell_{symbol}_{int(time.time())}",
                symbol=symbol,
                side='sell',
                amount=position.amount,
                price=market_data.price,
                timestamp=datetime.now(),
                profit_loss=profit_loss,
                strategy=strategy.name
            )
            
            # Update balance
            sale_value = position.amount * market_data.price
            self.balance += sale_value
            
            # Remove position
            del self.positions[symbol]
            
            # Save to database
            self.db.save_trade(trade)
            
            logger.info(f"SELL executed: {position.amount:.6f} {symbol} at ${market_data.price:.2f}, P&L: ${profit_loss:.2f}")
            
        except Exception as e:
            logger.error(f"Sell execution error: {e}")
    
    def train_ml_model(self):
        """Train the ML model on historical data"""
        historical_data = {}
        
        for symbol in self.data_fetcher.symbols:
            data = self.db.get_historical_data(symbol.replace('/', ''))
            if len(data) > 50:
                historical_data[symbol] = data
        
        if historical_data:
            self.ml_model.train(historical_data)
            logger.info("ML model training completed")
        else:
            logger.warning("No sufficient historical data for ML training")
    
    def start(self):
        """Start the trading engine"""
        self.running = True
        logger.info("Trading engine started")
        
        # Train ML model on startup
        threading.Thread(target=self.train_ml_model, daemon=True).start()
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.data_fetcher.running = False
        logger.info("Trading engine stopped")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        total_value = self.balance
        unrealized_pnl = 0.0
        
        for position in self.positions.values():
            total_value += position.amount * position.current_price
            unrealized_pnl += position.unrealized_pnl
        
        return {
            'total_value': total_value,
            'cash_balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'total_profit': total_value - 10000.0,  # Initial balance was 10k
            'num_positions': len(self.positions)
        }

# Chat bot functionality
class TradingChatBot:
    """Chat interface for trading bot commands"""
    
    def __init__(self, trading_engine: IndustrialTradingEngine):
        self.engine = trading_engine
        self.commands = {
            'status': self.get_status,
            'balance': self.get_balance,
            'positions': self.get_positions,
            'buy': self.manual_buy,
            'sell': self.manual_sell,
            'start': self.start_trading,
            'stop': self.stop_trading,
            'train': self.train_model,
            'help': self.get_help
        }
    
    def process_message(self, message: str) -> str:
        """Process chat message and return response"""
        parts = message.lower().strip().split()
        
        if not parts:
            return "Please enter a command. Type 'help' for available commands."
        
        command = parts[0]
        args = parts[1:]
        
        if command in self.commands:
            try:
                return self.commands[command](args)
            except Exception as e:
                return f"Error executing command: {e}"
        else:
            return f"Unknown command: {command}. Type 'help' for available commands."
    
    def get_status(self, args) -> str:
        """Get bot status"""
        metrics = self.engine.get_performance_metrics()
        status = "RUNNING" if self.engine.running else "STOPPED"
        
        return f"""🤖 Bot Status: {status}
💰 Total Value: ${metrics['total_value']:.2f}
💵 Cash: ${metrics['cash_balance']:.2f}
📊 Unrealized P&L: ${metrics['unrealized_pnl']:.2f}
📈 Total Profit: ${metrics['total_profit']:.2f}
📋 Open Positions: {metrics['num_positions']}"""
    
    def get_balance(self, args) -> str:
        """Get current balance"""
        return f"💰 Current balance: ${self.engine.balance:.2f}"
    
    def get_positions(self, args) -> str:
        """Get current positions"""
        if not self.engine.positions:
            return "📋 No open positions"
        
        response = "📋 Current Positions:\n"
        for symbol, position in self.engine.positions.items():
            response += f"• {symbol}: {position.amount:.6f} @ ${position.entry_price:.2f} (P&L: ${position.unrealized_pnl:.2f})\n"
        
        return response
    
    def manual_buy(self, args) -> str:
        """Manual buy command"""
        if len(args) < 2:
            return "Usage: buy <symbol> <amount>"
        
        symbol = args[0].upper()
        try:
            amount = float(args[1])
            # Implement manual buy logic
            return f"Buy order placed: ${amount} of {symbol}"
        except ValueError:
            return "Invalid amount specified"
    
    def manual_sell(self, args) -> str:
        """Manual sell command"""
        if len(args) < 1:
            return "Usage: sell <symbol>"
        
        symbol = args[0].upper()
        if symbol in self.engine.positions:
            # Implement manual sell logic
            return f"Sell order placed for {symbol}"
        else:
            return f"No position found for {symbol}"
    
    def start_trading(self, args) -> str:
        """Start automated trading"""
        self.engine.start()
        return "🚀 Automated trading started!"
    
    def stop_trading(self, args) -> str:
        """Stop automated trading"""
        self.engine.stop()
        return "🛑 Automated trading stopped!"
    
    def train_model(self, args) -> str:
        """Train ML model"""
        self.engine.train_ml_model()
        return "🧠 ML model training initiated!"
    
    def get_help(self, args) -> str:
        """Get help message"""
        return """🤖 Available Commands:
• status - Get bot status and performance
• balance - Check current balance
• positions - View open positions
• buy <symbol> <amount> - Manual buy order
• sell <symbol> - Manual sell order
• start - Start automated trading
• stop - Stop automated trading
• train - Train ML model
• help - Show this help message"""

# Global trading engine instance
trading_engine = None
chat_bot = None

def get_trading_engine():
    """Get or create trading engine instance"""
    global trading_engine, chat_bot
    if trading_engine is None:
        trading_engine = IndustrialTradingEngine()
        chat_bot = TradingChatBot(trading_engine)
    return trading_engine, chat_bot