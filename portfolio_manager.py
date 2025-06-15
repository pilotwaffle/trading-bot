"""
Portfolio Management System for Crypto Trading Bot
Tracks portfolio performance, rebalancing, and optimization
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

from database import get_database, Trade, Position
from crypto_data_fetcher import get_crypto_data_fetcher

logger = logging.getLogger(__name__)

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    positions_count: int

@dataclass
class AssetAllocation:
    symbol: str
    target_weight: float
    current_weight: float
    current_value: float
    deviation: float
    rebalance_needed: bool

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    
class PortfolioManager:
    """
    Comprehensive portfolio management for crypto trading bot
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.database = get_database()
        self.data_fetcher = get_crypto_data_fetcher()
        self.initial_balance = initial_balance
        
        # Portfolio settings
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance
        self.min_position_value = 50.0   # Minimum $50 position
        self.cash_reserve_ratio = 0.1    # Keep 10% in cash
        
        # Target allocations (can be customized)
        self.target_allocations = {
            'BTC/USDT': 0.40,  # 40% Bitcoin
            'ETH/USDT': 0.30,  # 30% Ethereum
            'SOL/USDT': 0.10,  # 10% Solana
            'ADA/USDT': 0.05,  # 5% Cardano
            'DOT/USDT': 0.05,  # 5% Polkadot
            'CASH': 0.10       # 10% Cash
        }
        
        logger.info("Portfolio Manager initialized")
    
    def get_current_portfolio(self) -> Dict[str, Any]:
        """Get comprehensive current portfolio status"""
        try:
            positions = self.database.get_positions()
            performance = self.database.get_portfolio_performance()
            
            # Calculate current values
            total_positions_value = 0.0
            position_data = []
            
            for position in positions:
                position_value = position.quantity * position.current_price
                total_positions_value += position_value
                
                position_data.append({
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'position_value': position_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': (position.unrealized_pnl / (position.quantity * position.entry_price)) * 100,
                    'weight': 0.0  # Will be calculated below
                })
            
            # Cash balance calculation
            cash_balance = self.initial_balance + performance.get('total_pnl', 0.0)
            total_portfolio_value = cash_balance + total_positions_value
            
            # Calculate weights
            for pos_data in position_data:
                pos_data['weight'] = (pos_data['position_value'] / total_portfolio_value) * 100
            
            # Overall portfolio metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            total_return = ((total_portfolio_value - self.initial_balance) / self.initial_balance) * 100
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_value': total_portfolio_value,
                'cash_balance': cash_balance,
                'positions