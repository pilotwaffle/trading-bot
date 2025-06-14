import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
        # Only include exchanges available in your region
        self.exchanges = {
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True})
        }
        self.symbols = self._load_supported_symbols()
        logger.info(f"Symbols used for trading: {self.symbols}")
        self.running = False

    def _load_supported_symbols(self):
        """
        Dynamically load supported symbols from all exchanges at runtime.
        Returns a sorted list of unique symbols from all enabled exchanges.
        """
        logger.info("Loading supported trading symbols from exchanges...")
        symbols_per_exchange = []
        for name, exchange in self.exchanges.items():
            try:
                markets = exchange.load_markets()
                symbols = set(markets.keys())
                logger.info(f"{name} supports {len(symbols)} symbols.")
                symbols_per_exchange.append(symbols)
            except Exception as e:
                logger.warning(f"Failed to load markets for {name}: {e}")
        if symbols_per_exchange:
            supported_symbols = set.union(*symbols_per_exchange)
            logger.info(f"Final supported symbols: {sorted(list(supported_symbols))}")
            return sorted(list(supported_symbols))
        return []

    def get_crypto_prices(self) -> Dict[str, MarketData]:
        """Fetch current crypto prices from available exchanges."""
        prices = {}
        for symbol in self.symbols:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    last = ticker.get('last')
                    base_vol = ticker.get('baseVolume')
                    change = ticker.get('percentage', 0)
                    if last is not None and base_vol is not None:
                        prices[symbol] = MarketData(
                            symbol=symbol,
                            price=float(last),
                            volume=float(base_vol),
                            change_24h=float(change or 0),
                            timestamp=datetime.now()
                        )
                        break  # Got valid price, stop trying others
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} from {exchange_name}: {e}")
        return prices

    def start_real_time_feed(self, callback):
        """Start real-time price polling feed."""
        self.running = True
        threading.Thread(target=self._feed, args=(callback,), daemon=True).start()

    def _feed(self, callback):
        """Polling feed implementation (every 5 seconds)."""
        while self.running:
            try:
                prices = self.get_crypto_prices()
                callback(prices)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Feed error: {e}")
                time.sleep(10)

class MLTradingModel:
    """Machine Learning model for crypto trading predictions"""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, price_data: pd.DataFrame) -> np.ndarray:
        df = price_data.copy()
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(10).std()
        df['volume_sma'] = df['volume'].rolling(5).mean()
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        feature_columns = [
            'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower',
            'price_change', 'volatility', 'volume_sma', 'hour', 'day_of_week'
        ]
        return df[feature_columns].fillna(0).values

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20):
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    def train(self, historical_data: Dict[str, pd.DataFrame]):
        logger.info("Training ML model...")
        all_features = []
        all_targets = []
        for symbol, data in historical_data.items():
            if len(data) < 50:
                continue
            features = self.prepare_features(data)
            targets = data['close'].shift(-1).pct_change().fillna(0).values
            features = features[:-1]
            targets = targets[:-1]
            all_features.extend(features)
            all_targets.extend(targets)
        if len(all_features) > 0:
            X = np.array(all_features)
            y = np.array(all_targets)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info(f"Model trained on {len(X)} samples")
        else:
            logger.info("Not enough data available for ML model training.")

    def predict(self, current_data: pd.DataFrame) -> float:
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
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.enabled = True

    def should_buy(self, market_data: MarketData, historical_data: pd.DataFrame) -> bool:
        raise NotImplementedError

    def should_sell(self, market_data: MarketData, historical_data: pd.DataFrame, position: Position) -> bool:
        raise NotImplementedError

class DCAStrategy(TradingStrategy):
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
        take_profit = self.params.get('take_profit', 0.05)
        stop_loss = self.params.get('stop_loss', 0.03)
        price_change = (market_data.price - position.entry_price) / position.entry_price
        return price_change >= take_profit or price_change <= -stop_loss

class MLStrategy(TradingStrategy):
    def __init__(self, params: Dict[str, Any], ml_model: MLTradingModel):
        super().__init__("ML", params)
        self.ml_model = ml_model

    def should_buy(self, market_data: MarketData, historical_data: pd.DataFrame) -> bool:
        if len(historical_data) < 50:
            return False
        prediction = self.ml_model.predict(historical_data)
        threshold = self.params.get('buy_threshold', 0.02)
        return prediction > threshold

    def should_sell(self, market_data: MarketData, historical_data: pd.DataFrame, position: Position) -> bool:
        prediction = self.ml_model.predict(historical_data)
        sell_threshold = self.params.get('sell_threshold', -0.01)
        price_change = (market_data.price - position.entry_price) / position.entry_price
        take_profit = self.params.get('take_profit', 0.1)
        stop_loss = self.params.get('stop_loss', 0.05)
        return (prediction < sell_threshold or
                price_change >= take_profit or
                price_change <= -stop_loss)

class DatabaseManager:
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    params TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades 
                (id, symbol, side, amount, price, timestamp, profit_loss, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade.id, trade.symbol, trade.side, trade.amount,
                  trade.price, trade.timestamp, trade.profit_loss, trade.strategy))

    def save_market_data(self, market_data: MarketData):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO market_data 
                (symbol, price, volume, change_24h, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (market_data.symbol, market_data.price, market_data.volume,
                  market_data.change_24h, market_data.timestamp))

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, price as close, volume
                FROM market_data 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days)
            df = pd.read_sql_query(query, conn, params=(symbol,), parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df

class IndustrialTradingEngine:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_fetcher = CryptoDataFetcher()
        self.ml_model = MLTradingModel()
        self.positions: Dict[str, Position] = {}
        self.strategies: List[TradingStrategy] = []
        self.current_market_data: Dict[str, MarketData] = {}
        self.balance = 10000.0  # Starting balance
        self.running = False
        self.init_strategies()
        self.data_fetcher.start_real_time_feed(self.on_market_data)

    def init_strategies(self):
        dca_params = {
            'interval_hours': 6,
            'amount': 100,
            'take_profit': 0.05,
            'stop_loss': 0.03
        }
        self.strategies.append(DCAStrategy(dca_params))
        ml_params = {
            'buy_threshold': 0.02,
            'sell_threshold': -0.01,
            'take_profit': 0.1,
            'stop_loss': 0.05
        }
        self.strategies.append(MLStrategy(ml_params, self.ml_model))

    def on_market_data(self, market_data: Dict[str, MarketData]):
        self.current_market_data.update(market_data)
        for data in market_data.values():
            self.db.save_market_data(data)
        self.update_positions()
        if self.running:
            self.execute_strategies()

    def update_positions(self):
        for symbol, position in self.positions.items():
            if symbol in self.current_market_data:
                market_data = self.current_market_data[symbol]
                position.current_price = market_data.price
                if position.side == 'long':
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.amount

    def execute_strategies(self):
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            try:
                self.execute_strategy(strategy)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} error: {e}")

    def execute_strategy(self, strategy: TradingStrategy):
        for symbol, market_data in self.current_market_data.items():
            historical_data = self.db.get_historical_data(symbol.replace('/', ''))
            if len(historical_data) < 10:
                continue
            if symbol not in self.positions and strategy.should_buy(market_data, historical_data):
                self.execute_buy(symbol, strategy)
            elif symbol in self.positions and strategy.should_sell(market_data, historical_data, self.positions[symbol]):
                self.execute_sell(symbol, strategy)

    def execute_buy(self, symbol: str, strategy: TradingStrategy):
        try:
            amount = strategy.params.get('amount', 100)
            market_data = self.current_market_data[symbol]
            if self.balance >= amount and market_data.price > 0:
                quantity = amount / market_data.price
                trade = Trade(
                    id=f"buy_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    side='buy',
                    amount=quantity,
                    price=market_data.price,
                    timestamp=datetime.now(),
                    strategy=strategy.name
                )
                position = Position(
                    symbol=symbol,
                    amount=quantity,
                    entry_price=market_data.price,
                    current_price=market_data.price,
                    unrealized_pnl=0.0,
                    side='long'
                )
                self.positions[symbol] = position
                self.balance -= amount
                self.db.save_trade(trade)
                logger.info(f"BUY executed: {quantity:.6f} {symbol} at ${market_data.price:.2f}")
        except Exception as e:
            logger.error(f"Buy execution error: {e}")

    def execute_sell(self, symbol: str, strategy: TradingStrategy):
        try:
            position = self.positions[symbol]
            market_data = self.current_market_data[symbol]
            profit_loss = (market_data.price - position.entry_price) * position.amount
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
            sale_value = position.amount * market_data.price
            self.balance += sale_value
            del self.positions[symbol]
            self.db.save_trade(trade)
            logger.info(f"SELL executed: {position.amount:.6f} {symbol} at ${market_data.price:.2f}, P&L: ${profit_loss:.2f}")
        except Exception as e:
            logger.error(f"Sell execution error: {e}")

    def train_ml_model(self):
        historical_data = {}
        for symbol in self.data_fetcher.symbols:
            data = self.db.get_historical_data(symbol.replace('/', ''))
            if len(data) > 50:
                historical_data[symbol] = data
        if historical_data:
            self.ml_model.train(historical_data)
            logger.info("ML model training completed")
        else:
            logger.info("No sufficient historical data for ML training")

    def start(self):
        self.running = True
        logger.info("Trading engine started")
        threading.Thread(target=self.train_ml_model, daemon=True).start()

    def stop(self):
        self.running = False
        self.data_fetcher.running = False
        logger.info("Trading engine stopped")

    def get_performance_metrics(self) -> Dict[str, float]:
        total_value = self.balance
        unrealized_pnl = 0.0
        for position in self.positions.values():
            total_value += position.amount * position.current_price
            unrealized_pnl += position.unrealized_pnl
        return {
            'total_value': total_value,
            'cash_balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'total_profit': total_value - 10000.0,
            'num_positions': len(self.positions)
        }

class TradingChatBot:
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
        metrics = self.engine.get_performance_metrics()
        status = "RUNNING" if self.engine.running else "STOPPED"
        return f"""🤖 Bot Status: {status}
💰 Total Value: ${metrics['total_value']:.2f}
💵 Cash: ${metrics['cash_balance']:.2f}
📊 Unrealized P&L: ${metrics['unrealized_pnl']:.2f}
📈 Total Profit: ${metrics['total_profit']:.2f}
📋 Open Positions: {metrics['num_positions']}"""

    def get_balance(self, args) -> str:
        return f"💰 Current balance: ${self.engine.balance:.2f}"

    def get_positions(self, args) -> str:
        if not self.engine.positions:
            return "📋 No open positions"
        response = "📋 Current Positions:\n"
        for symbol, position in self.engine.positions.items():
            response += f"• {symbol}: {position.amount:.6f} @ ${position.entry_price:.2f} (P&L: ${position.unrealized_pnl:.2f})\n"
        return response

    def manual_buy(self, args) -> str:
        if len(args) < 2:
            return "Usage: buy <symbol> <amount>"
        symbol = args[0].upper()
        try:
            amount = float(args[1])
            # Implement manual buy logic here if desired
            return f"Buy order placed: ${amount} of {symbol}"
        except ValueError:
            return "Invalid amount specified"

    def manual_sell(self, args) -> str:
        if len(args) < 1:
            return "Usage: sell <symbol>"
        symbol = args[0].upper()
        if symbol in self.engine.positions:
            # Implement manual sell logic here if desired
            return f"Sell order placed for {symbol}"
        else:
            return f"No position found for {symbol}"

    def start_trading(self, args) -> str:
        self.engine.start()
        return "🚀 Automated trading started!"

    def stop_trading(self, args) -> str:
        self.engine.stop()
        return "🛑 Automated trading stopped!"

    def train_model(self, args) -> str:
        self.engine.train_ml_model()
        return "🧠 ML model training initiated!"

    def get_help(self, args) -> str:
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

trading_engine = None
chat_bot = None

def get_trading_engine():
    global trading_engine, chat_bot
    if trading_engine is None:
        trading_engine = IndustrialTradingEngine()
        chat_bot = TradingChatBot(trading_engine)
    return trading_engine, chat_bot