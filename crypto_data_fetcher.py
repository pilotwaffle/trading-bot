"""
Real-time Crypto Data Fetcher
Handles live market data from multiple exchanges
"""

import asyncio
import json
import logging
import threading
import time
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import requests
import ccxt
from database import get_database, MarketData

logger = logging.getLogger(__name__)

@dataclass
class CryptoPrice:
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    exchange: str = "binance"

class CryptoDataFetcher:
    """
    Real-time cryptocurrency data fetcher supporting multiple exchanges
    """
    
    def __init__(self):
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT',
            'DOGE/USDT', 'MATIC/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
            'SUI/USDT', 'SHIB/USDT', 'LTC/USDT', 'HBAR/USDT', 'NEAR/USDT',
            'PYTH/USDT', 'ONDO/USDT', 'CRO/USDT'
        ]
        
        self.exchanges = self._init_exchanges()
        self.current_prices: Dict[str, CryptoPrice] = {}
        self.subscribers: List[Callable] = []
        self.running = False
        self.update_interval = 5  # seconds
        self.database = get_database()
        self._ws_connections = {}
        
    def _init_exchanges(self) -> Dict[str, Any]:
        """Initialize exchange connections"""
        exchanges = {}
        
        try:
            # Binance (primary)
            exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Add your API key if needed
                'secret': '',  # Add your secret if needed
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Coinbase Pro (backup)
            exchanges['coinbase'] = ccxt.coinbasepro({
                'enableRateLimit': True,
                'sandbox': False
            })
            
            # Kraken (backup)
            exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True
            })
            
            logger.info(f"Initialized {len(exchanges)} exchanges")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
        
        return exchanges
    
    def subscribe(self, callback: Callable[[Dict[str, CryptoPrice]], None]):
        """Subscribe to price updates"""
        self.subscribers.append(callback)
        logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from price updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber, total: {len(self.subscribers)}")
    
    def start(self):
        """Start fetching data"""
        if self.running:
            logger.warning("Data fetcher already running")
            return
        
        self.running = True
        logger.info("Starting crypto data fetcher...")
        
        # Start REST API polling in thread
        self.rest_thread = threading.Thread(target=self._rest_data_loop, daemon=True)
        self.rest_thread.start()
        
        # Start WebSocket connections for real-time data
        self._start_websockets()
        
        logger.info("Crypto data fetcher started")
    
    def stop(self):
        """Stop fetching data"""
        self.running = False
        
        # Close WebSocket connections
        for ws in self._ws_connections.values():
            if hasattr(ws, 'close'):
                ws.close()
        self._ws_connections.clear()
        
        logger.info("Crypto data fetcher stopped")
    
    def _rest_data_loop(self):
        """Main REST API data fetching loop"""
        while self.running:
            try:
                self._fetch_rest_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in REST data loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _fetch_rest_data(self):
        """Fetch data via REST API"""
        try:
            if 'binance' not in self.exchanges:
                return
            
            exchange = self.exchanges['binance']
            
            # Fetch tickers for all symbols
            tickers = exchange.fetch_tickers(self.symbols)
            
            updated_prices = {}
            
            for symbol, ticker in tickers.items():
                try:
                    price_data = CryptoPrice(
                        symbol=symbol,
                        price=float(ticker['last']),
                        volume_24h=float(ticker['baseVolume'] or 0),
                        change_24h=float(ticker['percentage'] or 0),
                        high_24h=float(ticker['high'] or 0),
                        low_24h=float(ticker['low'] or 0),
                        timestamp=datetime.now(),
                        exchange='binance'
                    )
                    
                    updated_prices[symbol] = price_data
                    self.current_prices[symbol] = price_data
                    
                    # Save to database
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=price_data.timestamp,
                        price=price_data.price,
                        volume=price_data.volume_24h,
                        high_24h=price_data.high_24h,
                        low_24h=price_data.low_24h,
                        change_24h=price_data.change_24h
                    )
                    self.database.save_market_data(market_data)
                    
                except Exception as e:
                    logger.error(f"Error processing ticker for {symbol}: {e}")
            
            # Notify subscribers
            if updated_prices:
                self._notify_subscribers(updated_prices)
                
        except Exception as e:
            logger.error(f"Error fetching REST data: {e}")
    
    def _start_websockets(self):
        """Start WebSocket connections for real-time data"""
        try:
            # Binance WebSocket
            self._start_binance_websocket()
        except Exception as e:
            logger.error(f"Error starting WebSockets: {e}")
    
    def _start_binance_websocket(self):
        """Start Binance WebSocket connection"""
        try:
            # Create stream names for all symbols
            streams = []
            for symbol in self.symbols:
                # Convert BTC/USDT to btcusdt format
                binance_symbol = symbol.replace('/', '').lower()
                streams.append(f"{binance_symbol}@ticker")
            
            # Binance WebSocket URL
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if 'stream' in data and 'data' in data:
                        self._process_binance_ticker(data['data'])
                except Exception as e:
                    logger