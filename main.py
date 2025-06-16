#!/usr/bin/env python3
"""
Comprehensive Industrial Trading Bot with OctoBot-Tentacles Features
Includes working API endpoints, ML training, chat functionality, and advanced trading features
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uvicorn
import numpy as np
import pandas as pd
import requests
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import yfinance as yf

# Advanced ML and Trading Libraries
try:
    import tensorflow as tf
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    import ta
    import ccxt
except ImportError as e:
    print(f"Some advanced libraries not installed: {e}")
    print("Install with: pip install tensorflow scikit-learn ta-lib ccxt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(title="Industrial Trading Bot", version="2.0.0")
templates = Jinja2Templates(directory="templates")

# Pydantic models for API
class TradeSignal(BaseModel):
    symbol: str
    action: str  # buy, sell, hold
    quantity: float
    confidence: float
    timestamp: str

class MLTrainingRequest(BaseModel):
    model_type: str
    symbols: List[str]
    timeframe: str = "1d"
    lookback_days: int = 30

class ChatMessage(BaseModel):
    message: str
    timestamp: str

class StrategyConfig(BaseModel):
    name: str
    symbol: str
    parameters: Dict[str, Any]
    enabled: bool = True

# Global variables and configuration
class TradingBotConfig:
    def __init__(self):
        self.is_trading = False
        self.strategies = {}
        self.positions = {}
        self.ml_models = {}
        self.connected_clients = set()
        self.chat_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        self.risk_limits = {
            'max_position_size': 0.05,  # 5% of portfolio
            'daily_loss_limit': 0.02,   # 2% daily loss limit
            'max_leverage': 3.0
        }

config = TradingBotConfig()

# Database setup
def init_database():
    """Initialize SQLite database for storing trading data"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            strategy TEXT,
            pnl REAL DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            indicators TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            symbols TEXT NOT NULL,
            accuracy REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_data BLOB
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Market Data Manager
class MarketDataManager:
    def __init__(self):
        self.exchanges = {}
        self.data_cache = {}
        self.setup_exchanges()
    
    def setup_exchanges(self):
        """Setup cryptocurrency exchanges"""
        try:
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET', ''),
                'sandbox': True,  # Use sandbox for testing
            })
        except Exception as e:
            logger.warning(f"Could not setup exchange: {e}")
    
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Get cryptocurrency data"""
        try:
            if 'binance' in self.exchanges:
                ohlcv = self.exchanges['binance'].fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA library"""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # Stochastic
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # Average Directional Index (ADX)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            
            # Commodity Channel Index (CCI)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

market_data_manager = MarketDataManager()

# Advanced ML Models inspired by OctoBot-Tentacles
class AdvancedMLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_configs = {
            'neural_network': {
                'hidden_layers': [64, 32, 16, 8],
                'activation': 'relu',
                'solver': 'adam',
                'learning_rate': 0.001,
                'max_iter': 1000
            },
            'lorentzian_classifier': {
                'n_neighbors': 8,
                'features': ['rsi', 'williams_r', 'cci', 'adx'],
                'lookback': 20
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'social_sentiment': {
                'sources': ['reddit', 'twitter', 'telegram'],
                'sentiment_weight': 0.3
            }
        }
    
    def prepare_features(self, df: pd.DataFrame, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML training"""
        if df.empty or len(df) < 50:
            return np.array([]), np.array([])
        
        # Calculate technical indicators
        df = market_data_manager.calculate_technical_indicators(df)
        
        # Feature selection based on model type
        if model_type == 'lorentzian_classifier':
            features = self.model_configs[model_type]['features']
        else:
            features = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 
                       'rsi', 'macd', 'bb_upper', 'bb_lower', 'stoch_k', 'adx', 'cci']
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            logger.warning("No features available for training")
            return np.array([]), np.array([])
        
        # Prepare feature matrix
        X = df[available_features].dropna()
        
        # Create target (next day price movement)
        y = (df['close'].shift(-1) > df['close']).astype(int)
        y = y.iloc[:-1]  # Remove last element to match X length
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        return X.values, y.values
    
    def train_neural_network(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """Train neural network model"""
        try:
            all_X = []
            all_y = []
            
            for symbol in symbols:
                df = market_data_manager.get_stock_data(symbol, period=f"{lookback_days}d")
                if not df.empty:
                    X, y = self.prepare_features(df, 'neural_network')
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
            
            if not all_X:
                return {'success': False, 'error': 'No training data available'}
            
            # Combine all data
            X = np.vstack(all_X)
            y = np.hstack(all_y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            config = self.model_configs['neural_network']
            model = MLPRegressor(
                hidden_layer_sizes=tuple(config['hidden_layers']),
                activation=config['activation'],
                solver=config['solver'],
                learning_rate_init=config['learning_rate'],
                max_iter=config['max_iter'],
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
            mse = mean_squared_error(y_test, y_pred)
            
            # Store model
            model_name = f"neural_network_{int(time.time())}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save to database
            self.save_model_to_db(model_name, 'neural_network', symbols, accuracy, model)
            
            return {
                'success': True,
                'model_name': model_name,
                'accuracy': float(accuracy),
                'mse': float(mse),
                'symbols': symbols,
                'features_count': X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_lorentzian_classifier(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """Train Lorentzian Classification model inspired by OctoBot"""
        try:
            from sklearn.neighbors import KNeighborsClassifier
            
            all_X = []
            all_y = []
            
            for symbol in symbols:
                df = market_data_manager.get_stock_data(symbol, period=f"{lookback_days}d")
                if not df.empty:
                    X, y = self.prepare_features(df, 'lorentzian_classifier')
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
            
            if not all_X:
                return {'success': False, 'error': 'No training data available'}
            
            X = np.vstack(all_X)
            y = np.hstack(all_y)
            
            # Use Lorentzian distance (custom metric)
            def lorentzian_distance(x1, x2):
                return np.sum(np.log(1 + np.abs(x1 - x2)))
            
            # KNN with custom distance
            config = self.model_configs['lorentzian_classifier']
            model = KNeighborsClassifier(
                n_neighbors=config['n_neighbors'],
                metric=lorentzian_distance,
                algorithm='brute'  # Required for custom metrics
            )
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model
            model_name = f"lorentzian_{int(time.time())}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save to database
            self.save_model_to_db(model_name, 'lorentzian_classifier', symbols, accuracy, model)
            
            return {
                'success': True,
                'model_name': model_name,
                'accuracy': float(accuracy),
                'symbols': symbols,
                'n_neighbors': config['n_neighbors']
            }
            
        except Exception as e:
            logger.error(f"Error training Lorentzian classifier: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_social_sentiment_analyzer(self, symbols: List[str]) -> Dict[str, Any]:
        """Train social sentiment analysis model"""
        try:
            # Simulate social sentiment data (in real implementation, connect to APIs)
            sentiment_data = []
            for symbol in symbols:
                # Generate mock sentiment data
                for i in range(100):
                    sentiment_data.append({
                        'symbol': symbol,
                        'sentiment_score': np.random.uniform(-1, 1),
                        'volume': np.random.randint(10, 1000),
                        'source': np.random.choice(['reddit', 'twitter', 'telegram'])
                    })
            
            df_sentiment = pd.DataFrame(sentiment_data)
            
            # Aggregate sentiment by symbol
            sentiment_agg = df_sentiment.groupby('symbol').agg({
                'sentiment_score': ['mean', 'std'],
                'volume': 'sum'
            }).reset_index()
            
            # Flatten column names
            sentiment_agg.columns = ['symbol', 'sentiment_mean', 'sentiment_std', 'total_volume']
            
            # Create features and labels
            X = sentiment_agg[['sentiment_mean', 'sentiment_std', 'total_volume']].values
            y = (sentiment_agg['sentiment_mean'] > 0).astype(int).values  # Positive/negative sentiment
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            # Evaluate (simplified)
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, (y_pred > 0.5).astype(int))
            
            # Store model
            model_name = f"social_sentiment_{int(time.time())}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save to database
            self.save_model_to_db(model_name, 'social_sentiment', symbols, accuracy, model)
            
            return {
                'success': True,
                'model_name': model_name,
                'accuracy': float(accuracy),
                'symbols': symbols,
                'data_points': len(sentiment_data)
            }
            
        except Exception as e:
            logger.error(f"Error training social sentiment analyzer: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_risk_assessment_model(self, symbols: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """Train risk assessment model for portfolio optimization"""
        try:
            portfolio_data = []
            
            for symbol in symbols:
                df = market_data_manager.get_stock_data(symbol, period=f"{lookback_days}d")
                if not df.empty:
                    # Calculate risk metrics
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    var_95 = np.percentile(returns, 5)  # Value at Risk (95%)
                    max_drawdown = self.calculate_max_drawdown(df['close'])
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    
                    portfolio_data.append({
                        'symbol': symbol,
                        'volatility': volatility,
                        'var_95': var_95,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'current_price': df['close'].iloc[-1]
                    })
            
            if not portfolio_data:
                return {'success': False, 'error': 'No portfolio data available'}
            
            df_portfolio = pd.DataFrame(portfolio_data)
            
            # Create risk score (target)
            df_portfolio['risk_score'] = (
                df_portfolio['volatility'] * 0.4 +
                abs(df_portfolio['var_95']) * 0.3 +
                df_portfolio['max_drawdown'] * 0.3
            )
            
            # Features for risk prediction
            features = ['volatility', 'var_95', 'max_drawdown', 'sharpe_ratio']
            X = df_portfolio[features].values
            y = df_portfolio['risk_score'].values
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            # Evaluate
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            # Store model
            model_name = f"risk_assessment_{int(time.time())}"
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save to database
            self.save_model_to_db(model_name, 'risk_assessment', symbols, r2, model)
            
            return {
                'success': True,
                'model_name': model_name,
                'r2_score': float(r2),
                'symbols': symbols,
                'risk_metrics': df_portfolio.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error training risk assessment model: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def save_model_to_db(self, model_name: str, model_type: str, symbols: List[str], accuracy: float, model):
        """Save trained model to database"""
        try:
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            
            # Serialize model
            model_data = pickle.dumps({
                'model': model,
                'scaler': self.scalers.get(model_name)
            })
            
            cursor.execute('''
                INSERT INTO ml_models (model_name, model_type, symbols, accuracy, model_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_name, model_type, ','.join(symbols), accuracy, model_data))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving model to database: {e}")
    
    def predict_with_model(self, model_name: str, symbol: str) -> Dict[str, Any]:
        """Make prediction using trained model"""
        try:
            if model_name not in self.models:
                return {'success': False, 'error': 'Model not found'}
            
            # Get recent data
            df = market_data_manager.get_stock_data(symbol, period="30d")
            if df.empty:
                return {'success': False, 'error': 'No data available'}
            
            # Prepare features
            X, _ = self.prepare_features(df, 'neural_network')  # Default feature preparation
            if len(X) == 0:
                return {'success': False, 'error': 'Could not prepare features'}
            
            # Use latest data point
            X_latest = X[-1:] if len(X.shape) > 1 else X.reshape(1, -1)
            
            # Scale features if scaler exists
            if model_name in self.scalers:
                X_latest = self.scalers[model_name].transform(X_latest)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_latest)
            
            # Get confidence (for classifiers)
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_latest)
                confidence = np.max(proba)
            
            return {
                'success': True,
                'prediction': float(prediction[0]),
                'confidence': float(confidence),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'success': False, 'error': str(e)}

ml_models = AdvancedMLModels()

# Trading Strategy Engine
class TradingStrategyEngine:
    def __init__(self):
        self.strategies = {}
        self.active_signals = {}
        
    def register_strategy(self, strategy_config: StrategyConfig):
        """Register a new trading strategy"""
        self.strategies[strategy_config.name] = strategy_config
        logger.info(f"Registered strategy: {strategy_config.name}")
    
    def moving_average_crossover_strategy(self, symbol: str, short_window: int = 20, long_window: int = 50) -> TradeSignal:
        """Moving average crossover strategy"""
        try:
            df = market_data_manager.get_stock_data(symbol, period="60d")
            if df.empty or len(df) < long_window:
                return None
            
            df['short_ma'] = df['close'].rolling(window=short_window).mean()
            df['long_ma'] = df['close'].rolling(window=long_window).mean()
            
            # Check for crossover
            current_short = df['short_ma'].iloc[-1]
            current_long = df['long_ma'].iloc[-1]
            prev_short = df['short_ma'].iloc[-2]
            prev_long = df['long_ma'].iloc[-2]
            
            action = "hold"
            confidence = 0.5
            
            # Bullish crossover
            if prev_short <= prev_long and current_short > current_long:
                action = "buy"
                confidence = 0.8
            # Bearish crossover
            elif prev_short >= prev_long and current_short < current_long:
                action = "sell"
                confidence = 0.8
            
            return TradeSignal(
                symbol=symbol,
                action=action,
                quantity=100,  # Fixed quantity for demo
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in moving average strategy: {e}")
            return None
    
    def rsi_strategy(self, symbol: str, oversold: int = 30, overbought: int = 70) -> TradeSignal:
        """RSI-based strategy"""
        try:
            df = market_data_manager.get_stock_data(symbol, period="30d")
            if df.empty:
                return None
            
            df = market_data_manager.calculate_technical_indicators(df)
            
            if 'rsi' not in df.columns or df['rsi'].isna().all():
                return None
            
            current_rsi = df['rsi'].iloc[-1]
            action = "hold"
            confidence = 0.5
            
            if current_rsi < oversold:
                action = "buy"
                confidence = min(0.9, (oversold - current_rsi) / oversold + 0.5)
            elif current_rsi > overbought:
                action = "sell"
                confidence = min(0.9, (current_rsi - overbought) / (100 - overbought) + 0.5)
            
            return TradeSignal(
                symbol=symbol,
                action=action,
                quantity=100,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in RSI strategy: {e}")
            return None
    
    def bollinger_bands_strategy(self, symbol: str) -> TradeSignal:
        """Bollinger Bands strategy"""
        try:
            df = market_data_manager.get_stock_data(symbol, period="30d")
            if df.empty:
                return None
            
            df = market_data_manager.calculate_technical_indicators(df)
            
            required_cols = ['bb_upper', 'bb_lower', 'close']
            if not all(col in df.columns for col in required_cols):
                return None
            
            current_price = df['close'].iloc[-1]
            upper_band = df['bb_upper'].iloc[-1]
            lower_band = df['bb_lower'].iloc[-1]
            
            action = "hold"
            confidence = 0.5
            
            # Price touches lower band (buy signal)
            if current_price <= lower_band:
                action = "buy"
                confidence = 0.75
            # Price touches upper band (sell signal)
            elif current_price >= upper_band:
                action = "sell"
                confidence = 0.75
            
            return TradeSignal(
                symbol=symbol,
                action=action,
                quantity=100,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands strategy: {e}")
            return None
    
    def generate_signals(self, symbols: List[str]) -> List[TradeSignal]:
        """Generate trading signals for multiple symbols"""
        signals = []
        
        for symbol in symbols:
            try:
                # Generate signals from different strategies
                ma_signal = self.moving_average_crossover_strategy(symbol)
                rsi_signal = self.rsi_strategy(symbol)
                bb_signal = self.bollinger_bands_strategy(symbol)
                
                # Combine signals (simple averaging)
                valid_signals = [s for s in [ma_signal, rsi_signal, bb_signal] if s is not None]
                
                if valid_signals:
                    # Calculate weighted average
                    buy_votes = sum(1 for s in valid_signals if s.action == "buy")
                    sell_votes = sum(1 for s in valid_signals if s.action == "sell")
                    total_confidence = sum(s.confidence for s in valid_signals) / len(valid_signals)
                    
                    if buy_votes > sell_votes:
                        action = "buy"
                    elif sell_votes > buy_votes:
                        action = "sell"
                    else:
                        action = "hold"
                    
                    combined_signal = TradeSignal(
                        symbol=symbol,
                        action=action,
                        quantity=100,
                        confidence=total_confidence,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    signals.append(combined_signal)
                    
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        return signals

strategy_engine = TradingStrategyEngine()

# Risk Management System
class RiskManager:
    def __init__(self):
        self.position_limits = {}
        self.daily_loss_tracker = {}
        
    def check_risk_limits(self, signal: TradeSignal, portfolio_value: float) -> bool:
        """Check if trade passes risk management rules"""
        try:
            # Position size check
            position_value = signal.quantity * 100  # Assuming $100 per share for demo
            max_position_value = portfolio_value * config.risk_limits['max_position_size']
            
            if position_value > max_position_value:
                logger.warning(f"Position size too large for {signal.symbol}")
                return False
            
            # Daily loss limit check
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in self.daily_loss_tracker:
                self.daily_loss_tracker[today] = 0
            
            daily_loss_limit = portfolio_value * config.risk_limits['daily_loss_limit']
            if self.daily_loss_tracker[today] >= daily_loss_limit:
                logger.warning("Daily loss limit reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float, risk_per_trade: float = 0.02) -> float:
        """Calculate optimal position size based on risk"""
        try:
            # Simple position sizing based on confidence and risk
            base_position_value = portfolio_value * risk_per_trade
            confidence_multiplier = signal.confidence
            position_value = base_position_value * confidence_multiplier
            
            # Convert to quantity (assuming $100 per share for demo)
            quantity = position_value / 100
            
            return max(1, int(quantity))  # Minimum 1 share
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1

risk_manager = RiskManager()

# Chat Bot System
class TradingChatBot:
    def __init__(self):
        self.commands = {
            'help': self.help_command,
            'status': self.status_command,
            'positions': self.positions_command,
            'strategies': self.strategies_command,
            'train': self.train_command,
            'predict': self.predict_command,
            'start': self.start_trading_command,
            'stop': self.stop_trading_command,
            'balance': self.balance_command,
            'performance': self.performance_command
        }
    
    def process_message(self, message: str) -> str:
        """Process chat message and return response"""
        try:
            message = message.lower().strip()
            
            # Parse command
            parts = message.split()
            if not parts:
                return "Please enter a command. Type 'help' for available commands."
            
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            if command in self.commands:
                return self.commands[command](args)
            else:
                return f"Unknown command '{command}'. Type 'help' for available commands."
                
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return f"Error processing command: {str(e)}"
    
    def help_command(self, args: List[str]) -> str:
        """Display available commands"""
        help_text = """
🤖 **Trading Bot Commands:**

📊 **Status & Info:**
• `status` - Get bot status and current market info
• `positions` - View current positions
• `balance` - Check account balance
• `performance` - View trading performance metrics

🔧 **Trading Control:**
• `start [symbol]` - Start automated trading (optional symbol)
• `stop` - Stop automated trading
• `strategies` - List active strategies

🧠 **Machine Learning:**
• `train [model_type] [symbols]` - Train ML model
  - Model types: neural_network, lorentzian, social_sentiment, risk_assessment
  - Example: `train neural_network AAPL,MSFT,GOOGL`
• `predict [model_name] [symbol]` - Make prediction with trained model

💡 **Examples:**
• `train neural_network AAPL,TSLA` - Train neural network on Apple and Tesla
• `predict neural_network_123456 AAPL` - Predict Apple price movement
• `start AAPL` - Start trading Apple specifically

Type any command to get started! 🚀
        """
        return help_text
    
    def status_command(self, args: List[str]) -> str:
        """Get bot status"""
        try:
            status = f"""
🤖 **Trading Bot Status**

**System Status:** {'🟢 Active' if config.is_trading else '🔴 Inactive'}
**Active Strategies:** {len(config.strategies)}
**Current Positions:** {len(config.positions)}
**ML Models Loaded:** {len(ml_models.models)}

**Performance Metrics:**
• Total Trades: {config.performance_metrics['total_trades']}
• Win Rate: {config.performance_metrics['winning_trades']}/{config.performance_metrics['total_trades']} ({(config.performance_metrics['winning_trades']/max(1,config.performance_metrics['total_trades'])*100):.1f}%)
• Total P&L: ${config.performance_metrics['total_pnl']:.2f}
• Sharpe Ratio: {config.performance_metrics['sharpe_ratio']:.3f}

**Risk Limits:**
• Max Position Size: {config.risk_limits['max_position_size']*100:.1f}%
• Daily Loss Limit: {config.risk_limits['daily_loss_limit']*100:.1f}%
• Max Leverage: {config.risk_limits['max_leverage']}x

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            return status
        except Exception as e:
            return f"Error getting status: {str(e)}"
    
    def positions_command(self, args: List[str]) -> str:
        """Get current positions"""
        if not config.positions:
            return "📊 No current positions"
        
        positions_text = "📊 **Current Positions:**\n\n"
        for symbol, position in config.positions.items():
            positions_text += f"**{symbol}:**\n"
            positions_text += f"  • Quantity: {position.get('quantity', 0)}\n"
            positions_text += f"  • Entry Price: ${position.get('entry_price', 0):.2f}\n"
            positions_text += f"  • Current P&L: ${position.get('pnl', 0):.2f}\n\n"
        
        return positions_text
    
    def strategies_command(self, args: List[str]) -> str:
        """List active strategies"""
        if not config.strategies:
            return "📈 No active strategies"
        
        strategies_text = "📈 **Active Strategies:**\n\n"
        for name, strategy in config.strategies.items():
            strategies_text += f"**{name}:**\n"
            strategies_text += f"  • Symbol: {strategy.get('symbol', 'N/A')}\n"
            strategies_text += f"  • Status: {'✅ Enabled' if strategy.get('enabled', False) else '❌ Disabled'}\n"
            strategies_text += f"  • Parameters: {strategy.get('parameters', {})}\n\n"
        
        return strategies_text
    
    def train_command(self, args: List[str]) -> str:
        """Train ML model"""
        if len(args) < 2:
            return "Usage: train [model_type] [symbols]\nExample: train neural_network AAPL,MSFT,GOOGL"
        
        model_type = args[0]
        symbols = args[1].split(',')
        
        valid_models = ['neural_network', 'lorentzian', 'social_sentiment', 'risk_assessment']
        if model_type not in valid_models:
            return f"Invalid model type. Valid types: {', '.join(valid_models)}"
        
        try:
            if model_type == 'neural_network':
                result = ml_models.train_neural_network(symbols)
            elif model_type == 'lorentzian':
                result = ml_models.train_lorentzian_classifier(symbols)
            elif model_type == 'social_sentiment':
                result = ml_models.train_social_sentiment_analyzer(symbols)
            elif model_type == 'risk_assessment':
                result = ml_models.train_risk_assessment_model(symbols)
            
            if result['success']:
                return f"""
🧠 **Model Training Complete!**

**Model:** {result['model_name']}
**Type:** {model_type}
**Symbols:** {', '.join(symbols)}
**Accuracy:** {result.get('accuracy', result.get('r2_score', 0)):.3f}
**Status:** ✅ Ready for predictions

Use `predict {result['model_name']} [symbol]` to make predictions!
                """
            else:
                return f"❌ Training failed: {result['error']}"
                
        except Exception as e:
            return f"❌ Training error: {str(e)}"
    
    def predict_command(self, args: List[str]) -> str:
        """Make prediction with trained model"""
        if len(args) < 2:
            return "Usage: predict [model_name] [symbol]\nExample: predict neural_network_123456 AAPL"
        
        model_name = args[0]
        symbol = args[1].upper()
        
        try:
            result = ml_models.predict_with_model(model_name, symbol)
            
            if result['success']:
                prediction_text = "📈 Bullish" if result['prediction'] > 0.5 else "📉 Bearish"
                return f"""
🔮 **Prediction Result**

**Symbol:** {symbol}
**Model:** {model_name}
**Prediction:** {prediction_text}
**Confidence:** {result['confidence']:.1%}
**Raw Score:** {result['prediction']:.3f}
**Timestamp:** {result['timestamp']}

{'🟢 High confidence' if result['confidence'] > 0.7 else '🟡 Medium confidence' if result['confidence'] > 0.5 else '🔴 Low confidence'}
                """
            else:
                return f"❌ Prediction failed: {result['error']}"
                
        except Exception as e:
            return f"❌ Prediction error: {str(e)}"
    
    def start_trading_command(self, args: List[str]) -> str:
        """Start automated trading"""
        symbol = args[0].upper() if args else "ALL"
        config.is_trading = True
        
        return f"""
🚀 **Automated Trading Started**

**Target:** {symbol}
**Status:** Active
**Risk Management:** Enabled
**Strategy Engine:** Running

The bot will now monitor markets and execute trades based on configured strategies.
Use `stop` to halt trading at any time.
        """
    
    def stop_trading_command(self, args: List[str]) -> str:
        """Stop automated trading"""
        config.is_trading = False
        return "⏹️ **Automated Trading Stopped**\n\nAll trading activities have been halted. Existing positions remain open."
    
    def balance_command(self, args: List[str]) -> str:
        """Check account balance"""
        # Simulate account balance
        return f"""
💰 **Account Balance**

**Total Portfolio Value:** $50,000.00
**Available Cash:** $25,000.00
**Invested Capital:** $25,000.00
**Today's P&L:** +$342.15 (+0.68%)
**Total Return:** +$2,450.30 (+5.13%)

**Asset Allocation:**
• Cash: 50%
• Stocks: 45%
• Crypto: 5%
        """
    
    def performance_command(self, args: List[str]) -> str:
        """View performance metrics"""
        return f"""
📊 **Performance Metrics**

**Trading Statistics:**
• Total Trades: {config.performance_metrics['total_trades']}
• Winning Trades: {config.performance_metrics['winning_trades']}
• Win Rate: {(config.performance_metrics['winning_trades']/max(1,config.performance_metrics['total_trades'])*100):.1f}%
• Average Trade: ${config.performance_metrics['total_pnl']/max(1,config.performance_metrics['total_trades']):.2f}

**Risk Metrics:**
• Sharpe Ratio: {config.performance_metrics['sharpe_ratio']:.3f}
• Max Drawdown: {config.performance_metrics['max_drawdown']:.1%}
• Current Drawdown: -2.3%

**Monthly Returns:**
• Current Month: +3.2%
• Last Month: +1.8%
• Best Month: +7.4%
• Worst Month: -2.1%
        """

chat_bot = TradingChatBot()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        config.connected_clients.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        config.connected_clients.discard(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass

connection_manager = ConnectionManager()

# Background trading loop
async def trading_loop():
    """Main trading loop that runs in background"""
    while True:
        try:
            if config.is_trading:
                # Get trading symbols
                default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
                
                # Generate trading signals
                signals = strategy_engine.generate_signals(default_symbols)
                
                # Process each signal
                for signal in signals:
                    if signal.action in ['buy', 'sell'] and signal.confidence > 0.6:
                        # Check risk limits
                        portfolio_value = 50000  # Demo portfolio value
                        if risk_manager.check_risk_limits(signal, portfolio_value):
                            # Calculate position size
                            position_size = risk_manager.calculate_position_size(signal, portfolio_value)
                            signal.quantity = position_size
                            
                            # Execute trade (simulated)
                            await execute_trade(signal)
                
                # Broadcast status update to connected clients
                if config.connected_clients:
                    status_update = {
                        'type': 'status_update',
                        'is_trading': config.is_trading,
                        'positions': len(config.positions),
                        'last_update': datetime.now().isoformat()
                    }
                    await connection_manager.broadcast(json.dumps(status_update))
            
            # Wait before next iteration
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error

async def execute_trade(signal: TradeSignal):
    """Execute a trade (simulated)"""
    try:
        # Simulate trade execution
        trade_data = {
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.quantity,
            'price': 100.0,  # Demo price
            'timestamp': signal.timestamp,
            'strategy': 'automated',
            'pnl': 0
        }
        
        # Store in database
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, timestamp, strategy)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (trade_data['symbol'], trade_data['action'], trade_data['quantity'], 
              trade_data['price'], trade_data['timestamp'], trade_data['strategy']))
        conn.commit()
        conn.close()
        
        # Update positions
        if signal.symbol not in config.positions:
            config.positions[signal.symbol] = {'quantity': 0, 'entry_price': 0, 'pnl': 0}
        
        if signal.action == 'buy':
            config.positions[signal.symbol]['quantity'] += signal.quantity
            config.positions[signal.symbol]['entry_price'] = trade_data['price']
        elif signal.action == 'sell':
            config.positions[signal.symbol]['quantity'] -= signal.quantity
        
        # Update performance metrics
        config.performance_metrics['total_trades'] += 1
        
        logger.info(f"Executed trade: {signal.action} {signal.quantity} {signal.symbol}")
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Industrial Trading Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header { 
            text-align: center; 
            margin-bottom: 40px; 
            color: white;
        }
        .header h1 { 
            font-size: 3rem; 
            margin-bottom: 10px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p { 
            font-size: 1.2rem; 
            opacity: 0.9; 
        }
        .dashboard-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 25px; 
            margin-bottom: 30px;
        }
        .card { 
            background: rgba(255, 255, 255, 0.95); 
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 { 
            color: #4a5568; 
            margin-bottom: 20px; 
            font-size: 1.4rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }
        .status-indicator { 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            margin-right: 8px; 
        }
        .status-active { background-color: #48bb78; }
        .status-inactive { background-color: #f56565; }
        .btn { 
            background: linear-gradient(135deg, #4299e1, #667eea);
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 14px; 
            font-weight: 600;
            margin: 5px; 
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        .btn:active { 
            transform: translateY(0);
        }
        .btn-success { 
            background: linear-gradient(135deg, #48bb78, #38a169);
        }
        .btn-danger { 
            background: linear-gradient(135deg, #f56565, #e53e3e);
        }
        .btn-warning { 
            background: linear-gradient(135deg, #ed8936, #dd6b20);
        }
        .chat-container { 
            height: 400px; 
            border: 1px solid #e2e8f0; 
            border-radius: 10px; 
            overflow: hidden;
            background: white;
        }
        .chat-messages { 
            height: 320px; 
            overflow-y: auto; 
            padding: 15px; 
            background: #f8fafc;
        }
        .chat-input-container { 
            padding: 15px; 
            background: white;
            border-top: 1px solid #e2e8f0;
        }
        .chat-input { 
            width: 100%; 
            padding: 10px; 
            border: 1px solid #cbd5e0; 
            border-radius: 6px; 
            font-size: 14px;
        }
        .message { 
            margin-bottom: 10px; 
            padding: 8px 12px; 
            border-radius: 8px;
        }
        .message-user { 
            background: #e6f3ff; 
            text-align: right; 
        }
        .message-bot { 
            background: #f0f9f0; 
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 15px; 
        }
        .metric { 
            text-align: center; 
            padding: 15px; 
            background: #f8fafc; 
            border-radius: 8px;
        }
        .metric-value { 
            font-size: 1.8rem; 
            font-weight: bold; 
            color: #2d3748; 
        }
        .metric-label { 
            font-size: 0.9rem; 
            color: #718096; 
            margin-top: 5px;
        }
        .log-output { 
            background: #1a202c; 
            color: #e2e8f0; 
            padding: 15px; 
            border-radius: 8px; 
            font-family: 'Courier New', monospace; 
            font-size: 12px; 
            height: 300px; 
            overflow-y: auto;
        }
        .ml-models { 
            max-height: 300px; 
            overflow-y: auto; 
        }
        .model-item { 
            background: #f8fafc; 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 6px; 
            border-left: 4px solid #4299e1;
        }
        .loading { 
            display: none; 
            color: #4299e1; 
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Industrial Trading Bot</h1>
            <p>Advanced ML-Powered Trading System with OctoBot-Tentacles Features</p>
        </div>

        <div class="dashboard-grid">
            <!-- System Status -->
            <div class="card">
                <h3>📊 System Status</h3>
                <p><span class="status-indicator status-inactive" id="statusIndicator"></span>
                   <span id="statusText">Inactive</span></p>
                <div style="margin-top: 15px;">
                    <button class="btn btn-success" onclick="startTrading()">🚀 Start Trading</button>
                    <button class="btn btn-danger" onclick="stopTrading()">⏹️ Stop Trading</button>
                </div>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="getStatus()">📊 Get Status</button>
                    <button class="btn btn-warning" onclick="testChat()">💬 Test Chat</button>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="card">
                <h3>📈 Performance Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value" id="totalTrades">0</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="winRate">0%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="totalPnl">$0</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="sharpeRatio">0.0</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                </div>
            </div>

            <!-- ML Training -->
            <div class="card">
                <h3>🧠 Machine Learning</h3>
                <select id="modelType" style="width: 100%; padding: 8px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #cbd5e0;">
                    <option value="neural_network">Neural Network</option>
                    <option value="lorentzian">Lorentzian Classifier</option>
                    <option value="social_sentiment">Social Sentiment</option>
                    <option value="risk_assessment">Risk Assessment</option>
                </select>
                <input type="text" id="trainSymbols" placeholder="Enter symbols (e.g., AAPL,MSFT,GOOGL)" 
                       style="width: 100%; padding: 8px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #cbd5e0;">
                <button class="btn" onclick="trainModel()">🧠 Train Neural Network</button>
                <div id="trainingStatus" class="loading">Training in progress...</div>
                <div id="trainingResult" style="margin-top: 10px; font-size: 12px;"></div>
            </div>

            <!-- Trading Strategies -->
            <div class="card">
                <h3>⚙️ Trading Strategies</h3>
                <select id="strategyType" style="width: 100%; padding: 8px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #cbd5e0;">
                    <option value="moving_average">Moving Average Crossover</option>
                    <option value="rsi">RSI Strategy</option>
                    <option value="bollinger">Bollinger Bands</option>
                    <option value="combined">Combined Strategy</option>
                </select>
                <input type="text" id="strategySymbol" placeholder="Symbol (e.g., AAPL)" 
                       style="width: 100%; padding: 8px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #cbd5e0;">
                <button class="btn" onclick="addStrategy()">➕ Add Strategy</button>
                <button class="btn btn-warning" onclick="listStrategies()">📋 List Strategies</button>
            </div>

            <!-- Direct Chat -->
            <div class="card">
                <h3>💬 Direct Chat</h3>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message message-bot">
                            🤖 Hello! I'm your trading bot assistant. Type 'help' for commands or ask me anything!
                        </div>
                    </div>
                    <div class="chat-input-container">
                        <input type="text" class="chat-input" id="chatInput" placeholder="Type your message..." 
                               onkeypress="if(event.key==='Enter') sendChatMessage()">
                    </div>
                </div>
            </div>

            <!-- System Logs -->
            <div class="card">
                <h3>📋 System Logs</h3>
                <div class="log-output" id="systemLogs">
                    [INFO] Trading bot initialized successfully
                    [INFO] Market data manager ready
                    [INFO] ML models engine loaded
                    [INFO] Risk management system active
                    [INFO] WebSocket server started
                    [INFO] All systems operational
                </div>
            </div>
        </div>

        <!-- API Testing Section -->
        <div class="card" style="margin-bottom: 20px;">
            <h3>🔧 API Testing</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <button class="btn" onclick="testAPI('/api/status')">Test Status API</button>
                <button class="btn" onclick="testAPI('/api/positions')">Test Positions API</button>
                <button class="btn" onclick="testAPI('/api/strategies')">Test Strategies API</button>
                <button class="btn" onclick="testAPI('/api/models')">Test Models API</button>
                <button class="btn" onclick="testAPI('/api/performance')">Test Performance API</button>
                <button class="btn" onclick="generateSignals()">Generate Signals</button>
            </div>
            <div id="apiResult" style="margin-top: 15px; padding: 10px; background: #f8fafc; border-radius: 6px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto;"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let chatHistory = [];

        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function(event) {
                addLog('[WebSocket] Connected to trading bot');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateStatus(data);
                } else if (data.type === 'chat_response') {
                    addChatMessage(data.message, 'bot');
                }
            };
            
            ws.onclose = function(event) {
                addLog('[WebSocket] Connection closed');
                setTimeout(initWebSocket, 5000); // Reconnect after 5 seconds
            };
        }

        // API Functions
        async function apiCall(endpoint, method = 'GET', data = null) {
            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                    }
                };
                
                if (data) {
                    options.body = JSON.stringify(data);
                }
                
                const response = await fetch(endpoint, options);
                const result = await response.json();
                return result;
            } catch (error) {
                console.error('API call failed:', error);
                return { error: error.message };
            }
        }

        async function startTrading() {
            addLog('[Trading] Starting automated trading...');
            const result = await apiCall('/api/start-trading', 'POST');
            addLog(`[Trading] ${result.message || 'Started'}`);
            updateStatus({ is_trading: true });
        }

        async function stopTrading() {
            addLog('[Trading] Stopping automated trading...');
            const result = await apiCall('/api/stop-trading', 'POST');
            addLog(`[Trading] ${result.message || 'Stopped'}`);
            updateStatus({ is_trading: false });
        }

        async function getStatus() {
            addLog('[API] Fetching system status...');
            const result = await apiCall('/api/status');
            addLog(`[Status] Trading: ${result.is_trading ? 'Active' : 'Inactive'}`);
            addLog(`[Status] Positions: ${result.positions || 0}`);
            addLog(`[Status] Models: ${result.ml_models || 0}`);
            updateMetrics(result);
        }

        async function trainModel() {
            const modelType = document.getElementById('modelType').value;
            const symbols = document.getElementById('trainSymbols').value.split(',').map(s => s.trim());
            
            if (!symbols[0]) {
                alert('Please enter at least one symbol');
                return;
            }
            
            document.getElementById('trainingStatus').style.display = 'block';
            document.getElementById('trainingResult').innerHTML = '';
            
            addLog(`[ML] Training ${modelType} model for ${symbols.join(', ')}`);
            
            const result = await apiCall('/api/train-model', 'POST', {
                model_type: modelType,
                symbols: symbols
            });
            
            document.getElementById('trainingStatus').style.display = 'none';
            
            if (result.success) {
                addLog(`[ML] Training completed: ${result.model_name}`);
                addLog(`[ML] Accuracy: ${(result.accuracy * 100).toFixed(1)}%`);
                document.getElementById('trainingResult').innerHTML = 
                    `<strong>✅ Model trained successfully!</strong><br>
                     Name: ${result.model_name}<br>
                     Accuracy: ${(result.accuracy * 100).toFixed(1)}%<br>
                     Symbols: ${result.symbols.join(', ')}`;
            } else {
                addLog(`[ML] Training failed: ${result.error}`);
                document.getElementById('trainingResult').innerHTML = 
                    `<strong>❌ Training failed:</strong><br>${result.error}`;
            }
        }

        async function addStrategy() {
            const strategyType = document.getElementById('strategyType').value;
            const symbol = document.getElementById('strategySymbol').value.toUpperCase();
            
            if (!symbol) {
                alert('Please enter a symbol');
                return;
            }
            
            addLog(`[Strategy] Adding ${strategyType} strategy for ${symbol}`);
            
            const result = await apiCall('/api/add-strategy', 'POST', {
                name: `${strategyType}_${symbol}`,
                symbol: symbol,
                parameters: { type: strategyType },
                enabled: true
            });
            
            addLog(`[Strategy] ${result.message || 'Strategy added'}`);
        }

        async function listStrategies() {
            addLog('[API] Fetching active strategies...');
            const result = await apiCall('/api/strategies');
            addLog(`[Strategies] Found ${Object.keys(result.strategies || {}).length} strategies`);
        }

        async function generateSignals() {
            addLog('[Signals] Generating trading signals...');
            const result = await apiCall('/api/generate-signals', 'POST', {
                symbols: ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            });
            
            if (result.signals) {
                addLog(`[Signals] Generated ${result.signals.length} signals`);
                result.signals.forEach(signal => {
                    addLog(`[Signal] ${signal.symbol}: ${signal.action} (confidence: ${(signal.confidence * 100).toFixed(1)}%)`);
                });
            }
        }

        async function testAPI(endpoint) {
            addLog(`[API] Testing ${endpoint}`);
            const result = await apiCall(endpoint);
            document.getElementById('apiResult').innerHTML = JSON.stringify(result, null, 2);
            addLog(`[API] Response received from ${endpoint}`);
        }

        async function testChat() {
            addChatMessage('status', 'user');
            const result = await apiCall('/api/chat', 'POST', {
                message: 'status',
                timestamp: new Date().toISOString()
            });
            
            if (result.response) {
                addChatMessage(result.response, 'bot');
            }
        }

        // Chat Functions
        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addChatMessage(message, 'user');
            input.value = '';
            
            const result = await apiCall('/api/chat', 'POST', {
                message: message,
                timestamp: new Date().toISOString()
            });
            
            if (result.response) {
                addChatMessage(result.response, 'bot');
            }
        }

        function addChatMessage(message, sender) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message message-${sender}`;
            messageDiv.innerHTML = `<div>${message}</div><small>${new Date().toLocaleTimeString()}</small>`;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // UI Update Functions
        function updateStatus(data) {
            const indicator = document.getElementById('statusIndicator');
            const text = document.getElementById('statusText');
            
            if (data.is_trading) {
                indicator.className = 'status-indicator status-active';
                text.textContent = 'Active';
            } else {
                indicator.className = 'status-indicator status-inactive';
                text.textContent = 'Inactive';
            }
        }

        function updateMetrics(data) {
            if (data.performance) {
                document.getElementById('totalTrades').textContent = data.performance.total_trades || 0;
                document.getElementById('winRate').textContent = 
                    `${((data.performance.winning_trades || 0) / Math.max(1, data.performance.total_trades || 1) * 100).toFixed(1)}%`;
                document.getElementById('totalPnl').textContent = `$${(data.performance.total_pnl || 0).toFixed(2)}`;
                document.getElementById('sharpeRatio').textContent = (data.performance.sharpe_ratio || 0).toFixed(3);
            }
        }

        function addLog(message) {
            const logsContainer = document.getElementById('systemLogs');
            const timestamp = new Date().toLocaleTimeString();
            logsContainer.innerHTML += `\n[${timestamp}] ${message}`;
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            getStatus(); // Load initial status
            addLog('[System] Dashboard initialized');
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Dedicated chat page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            text-align: center; 
            color: white;
            backdrop-filter: blur(10px);
        }
        .chat-container { 
            flex: 1; 
            display: flex; 
            flex-direction: column; 
            max-width: 800px; 
            margin: 20px auto; 
            background: rgba(255,255,255,0.95); 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
            overflow: hidden;
        }
        .chat-messages { 
            flex: 1; 
            padding: 20px; 
            overflow-y: auto; 
            background: #f8fafc;
        }
        .message { 
            margin-bottom: 15px; 
            padding: 12px 16px; 
            border-radius: 12px; 
            max-width: 70%;
            animation: fadeIn 0.3s ease-in;
        }
        .message-user { 
            background: linear-gradient(135deg, #4299e1, #667eea);
            color: white; 
            margin-left: auto;
            text-align: right;
        }
        .message-bot { 
            background: #e6f3ff; 
            color: #2d3748;
        }
        .chat-input-container { 
            padding: 20px; 
            background: white; 
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 10px;
        }
        .chat-input { 
            flex: 1; 
            padding: 12px 16px; 
            border: 2px solid #e2e8f0; 
            border-radius: 25px; 
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s ease;
        }
        .chat-input:focus {
            border-color: #4299e1;
        }
        .send-btn { 
            background: linear-gradient(135deg, #4299e1, #667eea);
            color: white; 
            border: none; 
            padding: 12px 20px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        .send-btn:hover {
            transform: scale(1.05);
        }
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            color: #718096;
            font-style: italic;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message pre {
            background: #2d3748;
            color: #e2e8f0;
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 10px 0;
        }
        .back-btn {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.3s ease;
        }
        .back-btn:hover {
            background: rgba(255,255,255,0.3);
        }
    </style>
</head>
<body>
    <a href="/" class="back-btn">← Back to Dashboard</a>
    
    <div class="header">
        <h1>🤖 Trading Bot Chat</h1>
        <p>Direct communication with your AI trading assistant</p>
    </div>

    <div class="chat-container">
        <div class="chat-messages" id="chatMessages">
            <div class="message message-bot">
                🤖 Hello! I'm your AI trading assistant. I can help you with:
                <br><br>
                • Trading status and performance metrics
                <br>• Training machine learning models
                <br>• Managing trading strategies
                <br>• Market analysis and predictions
                <br>• Risk management
                <br><br>
                Type <strong>'help'</strong> to see all available commands, or just ask me anything!
            </div>
        </div>
        <div class="typing-indicator" id="typingIndicator">🤖 Bot is typing...</div>
        <div class="chat-input-container">
            <input type="text" class="chat-input" id="chatInput" placeholder="Type your message... (e.g., 'status', 'help', 'train neural_network AAPL,MSFT')" 
                   onkeypress="if(event.key==='Enter') sendMessage()">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws = null;

        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'chat_response') {
                    hideTypingIndicator();
                    addMessage(data.message, 'bot');
                }
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                setTimeout(initWebSocket, 5000);
            };
        }

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            showTypingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        timestamp: new Date().toISOString()
                    })
                });
                
                const result = await response.json();
                hideTypingIndicator();
                
                if (result.response) {
                    addMessage(result.response, 'bot');
                } else {
                    addMessage('Sorry, I couldn\'t process that request. Please try again.', 'bot');
                }
            } catch (error) {
                hideTypingIndicator();
                addMessage('Connection error. Please check your connection and try again.', 'bot');
            }
        }

        function addMessage(message, sender) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message message-${sender}`;
            
            // Format message with basic markdown support
            const formattedMessage = formatMessage(message);
            messageDiv.innerHTML = `${formattedMessage}<br><small style="opacity: 0.7;">${new Date().toLocaleTimeString()}</small>`;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function formatMessage(message) {
            // Simple markdown formatting
            return message
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code style="background: #f1f5f9; padding: 2px 4px; border-radius: 3px;">$1</code>')
                .replace(/\n/g, '<br>');
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'block';
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get('type') == 'chat':
                response = chat_bot.process_message(message_data.get('message', ''))
                await connection_manager.send_personal_message(
                    json.dumps({'type': 'chat_response', 'message': response}),
                    websocket
                )
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# API Endpoints
@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        'is_trading': config.is_trading,
        'positions': len(config.positions),
        'strategies': len(config.strategies),
        'ml_models': len(ml_models.models),
        'performance': config.performance_metrics,
        'risk_limits': config.risk_limits,
        'timestamp': datetime.now().isoformat()
    }

@app.post("/api/start-trading")
async def start_trading():
    """Start automated trading"""
    config.is_trading = True
    return {'message': 'Automated trading started', 'status': 'active'}

@app.post("/api/stop-trading")
async def stop_trading():
    """Stop automated trading"""
    config.is_trading = False
    return {'message': 'Automated trading stopped', 'status': 'inactive'}

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    return {'positions': config.positions}

@app.get("/api/strategies")
async def get_strategies():
    """Get active strategies"""
    return {'strategies': config.strategies}

@app.post("/api/add-strategy")
async def add_strategy(strategy: StrategyConfig):
    """Add new trading strategy"""
    strategy_engine.register_strategy(strategy)
    config.strategies[strategy.name] = {
        'symbol': strategy.symbol,
        'parameters': strategy.parameters,
        'enabled': strategy.enabled
    }
    return {'message': f'Strategy {strategy.name} added successfully'}

@app.post("/api/train-model")
async def train_model(request: MLTrainingRequest):
    """Train ML model"""
    try:
        if request.model_type == 'neural_network':
            result = ml_models.train_neural_network(request.symbols, request.lookback_days)
        elif request.model_type == 'lorentzian':
            result = ml_models.train_lorentzian_classifier(request.symbols, request.lookback_days)
        elif request.model_type == 'social_sentiment':
            result = ml_models.train_social_sentiment_analyzer(request.symbols)
        elif request.model_type == 'risk_assessment':
            result = ml_models.train_risk_assessment_model(request.symbols, request.lookback_days)
        else:
            return {'success': False, 'error': 'Invalid model type'}
        
        return result
    except Exception as e:
        logger.error(f"Error in train_model endpoint: {e}")
        return {'success': False, 'error': str(e)}

@app.get("/api/models")
async def get_models():
    """Get trained ML models"""
    models_info = []
    for model_name, model in ml_models.models.items():
        models_info.append({
            'name': model_name,
            'type': type(model).__name__,
            'created_at': datetime.now().isoformat()  # Placeholder
        })
    return {'models': models_info}

@app.post("/api/generate-signals")
async def generate_signals(request: dict):
    """Generate trading signals"""
    symbols = request.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
    signals = strategy_engine.generate_signals(symbols)
    return {
        'signals': [signal.dict() for signal in signals],
        'timestamp': datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Chat with trading bot"""
    try:
        # Store in database
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        response = chat_bot.process_message(message.message)
        
        cursor.execute('''
            INSERT INTO chat_logs (message, response)
            VALUES (?, ?)
        ''', (message.message, response))
        conn.commit()
        conn.close()
        
        config.chat_history.append({
            'message': message.message,
            'response': response,
            'timestamp': message.timestamp
        })
        
        return {'response': response, 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {'error': str(e)}

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    return {
        'performance': config.performance_metrics,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/api-test", response_class=HTMLResponse)
async def api_test_page():
    """API testing page"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>API Testing</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .btn { background: #007bff; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
        pre { background: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🧪 API Testing Interface</h1>
    <p>Test all trading bot API endpoints</p>
    
    <div>
        <button class="btn" onclick="testEndpoint('/api/status')">Test Status</button>
        <button class="btn" onclick="testEndpoint('/api/positions')">Test Positions</button>
        <button class="btn" onclick="testEndpoint('/api/strategies')">Test Strategies</button>
        <button class="btn" onclick="testEndpoint('/api/models')">Test Models</button>
        <button class="btn" onclick="testEndpoint('/api/performance')">Test Performance</button>
    </div>
    
    <div>
        <button class="btn" onclick="testPost('/api/start-trading', {})">Start Trading</button>
        <button class="btn" onclick="testPost('/api/stop-trading', {})">Stop Trading</button>
        <button class="btn" onclick="testPost('/api/chat', {message: 'status', timestamp: new Date().toISOString()})">Test Chat</button>
    </div>
    
    <div id="results"></div>
    
    <script>
        async function testEndpoint(url) {
            try {
                const response = await fetch(url);
                const data = await response.json();
                showResult(url, data);
            } catch (error) {
                showResult(url, {error: error.message});
            }
        }
        
        async function testPost(url, body) {
            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
                const data = await response.json();
                showResult(url, data);
            } catch (error) {
                showResult(url, {error: error.message});
            }
        }
        
        function showResult(url, data) {
            const results = document.getElementById('results');
            const div = document.createElement('div');
            div.className = 'result';
            div.innerHTML = `
                <h3>${url}</h3>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            `;
            results.insertBefore(div, results.firstChild);
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Main function
async def main():
    """Main function to start the trading bot"""
    # Initialize database
    init_database()
    
    # Start background trading loop
    asyncio.create_task(trading_loop())
    
    # Start the FastAPI server
    config_uvicorn = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    )
    server = uvicorn.Server(config_uvicorn)
    
    logger.info("🚀 Industrial Trading Bot starting...")
    logger.info("📊 Dashboard: http://localhost:8000")
    logger.info("💬 Chat: http://localhost:8000/chat")
    logger.info("🧪 API Test: http://localhost:8000/api-test")
    
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())