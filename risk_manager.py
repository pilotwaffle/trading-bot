"""
Risk Management System for Crypto Trading Bot
Handles position sizing, risk controls, and portfolio risk management
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

from database import get_database, Trade, Position

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    var_1d: float  # Value at Risk 1 day
    var_7d: float  # Value at Risk 7 days
    max_drawdown: float
    sharpe_ratio: float
    portfolio_volatility: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    risk_level: RiskLevel

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    risk_amount: float
    risk_percentage: float
    stop_loss_price: float
    take_profit_price: float
    max_loss: float
    risk_reward_ratio: float
    
@dataclass
class TradeDecision:
    allowed: bool
    reason: str
    suggested_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

class RiskManager:
    """
    Comprehensive risk management system for crypto trading
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.database = get_database()
        self.initial_balance = initial_balance
        
        # Risk parameters (configurable)
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        self.max_position_size = 0.20   # 20% max position size
        self.max_correlation = 0.7      # Max correlation between positions
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.min_risk_reward = 1.5      # Min risk/reward ratio
        
        # Daily limits
        self.max_daily_trades = 10
        self.max_daily_loss = 0.05  # 5% max daily loss
        
        # Emergency controls
        self.circuit_breaker_loss = 0.20  # 20% portfolio loss triggers stop
        self.volatility_threshold = 0.05   # 5% volatility threshold
        
        logger.info("Risk Manager initialized")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, risk_amount: float = None) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            risk_amount: Amount to risk (if None, uses max_risk_per_trade)
        
        Returns:
            Optimal position size in base currency
        """
        try:
            current_balance = self.get_current_balance()
            
            if risk_amount is None:
                risk_amount = current_balance * self.max_risk_per_trade
            
            # Calculate risk per unit
            price_risk = abs(entry_price - stop_loss)
            risk_per_unit = price_risk / entry_price
            
            if risk_per_unit == 0:
                logger.warning("Invalid stop loss - no risk per unit")
                return 0.0
            
            # Calculate base position size
            position_size = risk_amount / price_risk
            
            # Apply position size limits
            max_position_value = current_balance * self.max_position_size
            max_size_by_limit = max_position_value / entry_price
            
            position_size = min(position_size, max_size_by_limit)
            
            # Check minimum trade size (avoid dust trades)
            min_trade_value = 10.0  # $10 minimum
            min_size = min_trade_value / entry_price
            
            if position_size < min_size:
                logger.info(f"Position size too small: {position_size:.6f}, minimum: {min_size:.6f}")
                return 0.0
            
            logger.info(f"Calculated position size for {symbol}: {position_size:.6f} units")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def validate_trade(self, symbol: str, side: str, quantity: float, 
                      price: float, stop_loss: float = None, 
                      take_profit: float = None) -> TradeDecision:
        """
        Validate if a trade should be allowed based on risk parameters
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Position size
            price: Entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        
        Returns:
            TradeDecision with validation result
        """
        try:
            # Check circuit breaker
            if self.is_circuit_breaker_triggered():
                return TradeDecision(
                    allowed=False,
                    reason="Circuit breaker triggered - excessive losses"
                )
            
            # Check daily limits
            daily_check = self.check_daily_limits()
            if not daily_check['allowed']:
                return TradeDecision(
                    allowed=False,
                    reason=daily_check['reason']
                )
            
            # Check position concentration
            concentration_check = self.check_position_concentration(symbol, quantity, price)
            if not concentration_check['allowed']:
                return TradeDecision(
                    allowed=False,
                    reason=concentration_check['reason']
                )
            
            # Validate stop loss and take profit
            if stop_loss:
                risk_reward_check = self.validate_risk_reward(
                    price, stop_loss, take_profit, side
                )
                if not risk_reward_check['allowed']:
                    return TradeDecision(
                        allowed=False,
                        reason=risk_reward_check['reason']
                    )
            
            # Check portfolio correlation
            correlation_check = self.check_correlation_risk(symbol)
            if not correlation_check['allowed']:
                return TradeDecision(
                    allowed=False,
                    reason=correlation_check['reason']
                )
            
            # Check market volatility
            volatility_check = self.check_market_volatility(symbol)
            if not volatility_check['allowed']:
                return TradeDecision(
                    allowed=False,
                    reason=volatility_check['reason']
                )
            
            # Calculate optimal position size if not provided
            if not stop_loss:
                # Use default 2% stop loss
                stop_loss = price * (0.98 if side == 'buy' else 1.02)
                
            optimal_size = self.calculate_position_size(symbol, price, stop_loss)
            
            # Suggest position size adjustment if needed
            if quantity > optimal_size * 1.1:  # 10% tolerance
                return TradeDecision(
                    allowed=True,
                    reason="Position size too large, suggesting optimal size",
                    suggested_size=optimal_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # Trade approved
            return TradeDecision(
                allowed=True,
                reason="Trade approved",
                suggested_size=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return TradeDecision(
                allowed=False,
                reason=f"Validation error: {e}"
            )
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            positions = self.database.get_positions()
            current_balance = self.get_current_balance()
            
            if not positions:
                return RiskMetrics(
                    var_1d=0.0, var_7d=0.0, max_drawdown=0.0,
                    sharpe_ratio=0.0, portfolio_volatility=0.0,
                    correlation_risk=0.0, concentration_risk=0.0,
                    leverage_ratio=0.0, risk_level=RiskLevel.LOW
                )
            
            # Calculate portfolio value and weights
            total_position_value = sum(
                pos.quantity * pos.current_price for pos in positions
            )
            
            portfolio_value = current_balance + total_position_value
            
            # Position weights
            weights = [
                (pos.quantity * pos.current_price) / portfolio_value 
                for pos in positions
            ]
            
            # Calculate concentration risk
            concentration_risk = max(weights) if weights else 0.0
            
            # Calculate portfolio volatility (simplified)
            returns_data = self.get_portfolio_returns(days=30)
            portfolio_volatility = np.std(returns_data) if len(returns_data) > 1 else 0.0
            
            # Calculate VaR (Value at Risk)
            if len(returns_data) > 1:
                var_1d = np.percentile(returns_data, 5) * portfolio_value  # 5% VaR
                var_7d = var_1d * np.sqrt(7)  # Scale to 7 days
            else:
                var_1d = var_7d = 0.0
            
            # Calculate max drawdown
            max_drawdown = self.calculate_max_drawdown(days=90)
            
            # Calculate Sharpe ratio (simplified)
            if portfolio_volatility > 0:
                avg_return = np.mean(returns_data) if len(returns_data) > 1 else 0.0
                sharpe_ratio = avg_return / portfolio_volatility
            else:
                sharpe_ratio = 0.0
            
            # Calculate leverage ratio
            leverage_ratio = total_position_value / current_balance if current_balance > 0 else 0.0
            
            # Calculate correlation risk (simplified)
            correlation_risk = self.calculate_correlation_risk()
            
            # Determine risk level
            risk_level = self.determine_risk_level(
                concentration_risk, portfolio_volatility, max_drawdown, leverage_ratio
            )
            
            return RiskMetrics(
                var_1d=var_1d,
                var_7d=var_7d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics(
                var_1d=0.0, var_7d=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, portfolio_volatility=0.0,
                correlation_risk=0.0, concentration_risk=0.0,
                leverage_ratio=0.0, risk_level=RiskLevel.CRITICAL
            )
    
    def get_current_balance(self) -> float:
        """Get current account balance including unrealized P&L"""
        try:
            performance = self.database.get_portfolio_performance()
            positions = self.database.get_positions()
            
            cash_balance = self.initial_balance + performance.get('total_pnl', 0.0)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            return cash_balance + unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error getting current balance: {e}")
            return self.initial_balance
    
    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker should halt trading"""
        try:
            current_balance = self.get_current_balance()
            total_loss = (self.initial_balance - current_balance) / self.initial_balance
            
            if total_loss >= self.circuit_breaker_loss:
                logger.critical(f"Circuit breaker triggered! Loss: {total_loss:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True  # Fail safe
    
    def check_daily_limits(self) -> Dict[str, Any]:
        """Check if daily trading limits are exceeded"""
        try:
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            # Get today's trades
            trades = self.database.get_trades(
                start_date=today_start,
                end_date=datetime.now()
            )
            
            # Check trade count limit
            if len(trades) >= self.max_daily_trades:
                return {
                    'allowed': False,
                    'reason': f"Daily trade limit exceeded: {len(trades)}/{self.max_daily_trades}"
                }
            
            # Check daily loss limit
            daily_pnl = sum(trade.profit_loss for trade in trades)
            daily_loss_pct = abs(daily_pnl) / self.get_current_balance()
            
            if daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss:
                return {
                    'allowed': False,
                    'reason': f"Daily loss limit exceeded: {daily_loss_pct:.2%}"
                }
            
            return {'allowed': True, 'reason': 'Daily limits OK'}
            
        except Exception as e:
            logger.error(f"Error checking daily limits: {e}")
            return {'allowed': False, 'reason': 'Error checking limits'}
    
    def check_position_concentration(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Check position concentration risk"""
        try:
            current_balance = self.get_current_balance()
            position_value = quantity * price
            concentration = position_value / current_balance
            
            if concentration > self.max_position_size:
                return {
                    'allowed': False,
                    'reason': f"Position size too large: {concentration:.2%} > {self.max_position_size:.2%}"
                }
            
            return {'allowed': True, 'reason': 'Position size OK'}
            
        except Exception as e:
            logger.error(f"Error checking position concentration: {e}")
            return {'allowed': False, 'reason': 'Error checking concentration'}
    
    def validate_risk_reward(self, entry_price: float, stop_loss: float, 
                           take_profit: float = None, side: str = 'buy') -> Dict[str, Any]:
        """Validate risk/reward ratio"""
        try:
            if side == 'buy':
                risk = entry_price - stop_loss
                reward = (take_profit - entry_price) if take_profit else risk * 2
            else:  # sell
                risk = stop_loss - entry_price
                reward = (entry_price - take_profit) if take_profit else risk * 2
            
            if risk <= 0:
                return {
                    'allowed': False,
                    'reason': 'Invalid stop loss - no risk defined'
                }
            
            risk_reward_ratio = reward / risk
            
            if risk_reward_ratio < self.min_risk_reward:
                return {
                    'allowed': False,
                    'reason': f"Poor risk/reward: {risk_reward_ratio:.2f} < {self.min_risk_reward}"
                }
            
            return {'allowed': True, 'reason': f'Risk/reward OK: {risk_reward_ratio:.2f}'}
            
        except Exception as e:
            logger.error(f"Error validating risk/reward: {e}")
            return {'allowed': False, 'reason': 'Error validating risk/reward'}
    
    def check_correlation_risk(self, symbol: str) -> Dict[str, Any]:
        """Check correlation risk with existing positions"""
        try:
            positions = self.database.get_positions()
            
            if not positions:
                return {'allowed': True, 'reason': 'No existing positions'}
            
            # Simplified correlation check (would need more sophisticated analysis)
            # For now, just check if we already have a position in the same asset
            existing_symbols = [pos.symbol for pos in positions]
            
            # Check for same base currency (simplified)
            symbol_base = symbol.split('/')[0] if '/' in symbol else symbol
            
            for existing_symbol in existing_symbols:
                existing_base = existing_symbol.split('/')[0] if '/' in existing_symbol else existing_symbol
                
                if symbol_base == existing_base:
                    return {
                        'allowed': False,
                        'reason': f'High correlation risk with existing {existing_symbol} position'
                    }
            
            return {'allowed': True, 'reason': 'Correlation risk acceptable'}
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return {'allowed': True, 'reason': 'Correlation check skipped due to error'}
    
    def check_market_volatility(self, symbol: str) -> Dict[str, Any]:
        """Check if market volatility is too high"""
        try:
            # Get recent price data
            df = self.database.get_market_data(symbol, hours=24)
            
            if df.empty or len(df) < 10:
                return {'allowed': True, 'reason': 'Insufficient data for volatility check'}
            
            # Calculate volatility
            returns = df['price'].pct_change().dropna()
            volatility = returns.std()
            
            if volatility > self.volatility_threshold:
                return {
                    'allowed': False,
                    'reason': f'High volatility: {volatility:.3f} > {self.volatility_threshold}'
                }
            
            return {'allowed': True, 'reason': f'Volatility OK: {volatility:.3f}'}
            
        except Exception as e:
            logger.error(f"Error checking market volatility: {e}")
            return {'allowed': True, 'reason': 'Volatility check skipped due to error'}
    
    def calculate_max_drawdown(self, days: int = 90) -> float:
        """Calculate maximum drawdown over specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            trades = self.database.get_trades(start_date=cutoff_date)
            
            if not trades:
                return 0.0
            
            # Calculate cumulative P&L
            cumulative_pnl = 0.0
            running_max = 0.0
            max_drawdown = 0.0
            
            for trade in sorted(trades, key=lambda x: x.timestamp):
                cumulative_pnl += trade.profit_loss
                running_max = max(running_max, cumulative_pnl)
                drawdown = (running_max - cumulative_pnl) / self.initial_balance
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def get_portfolio_returns(self, days: int = 30) -> List[float]:
        """Get daily portfolio returns for specified period"""
        try:
            # This is a simplified implementation
            # In reality, you'd want to calculate daily portfolio values
            cutoff_date = datetime.now() - timedelta(days=days)
            trades = self.database.get_trades(start_date=cutoff_date)
            
            if not trades:
                return []
            
            # Group trades by day and calculate daily returns
            daily_pnl = {}
            for trade in trades:
                day = trade.timestamp.date()
                if day not in daily_pnl:
                    daily_pnl[day] = 0.0
                daily_pnl[day] += trade.profit_loss
            
            # Convert to returns (simplified)
            returns = []
            balance = self.initial_balance
            
            for day in sorted(daily_pnl.keys()):
                daily_return = daily_pnl[day] / balance
                returns.append(daily_return)
                balance += daily_pnl[day]
            
            return returns
            
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {e}")
            return []
    
    def calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk (simplified)"""
        try:
            positions = self.database.get_positions()
            
            if len(positions) < 2:
                return 0.0
            
            # Simplified correlation calculation
            # In reality, you'd calculate actual price correlations
            
            # For crypto, assume higher correlation for similar assets
            correlation_score = 0.0
            symbols = [pos.symbol for pos in positions]
            
            # Count similar assets (simplified)
            base_currencies = {}
            for symbol in symbols:
                base = symbol.split('/')[0] if '/' in symbol else symbol
                base_currencies[base] = base_currencies.get(base, 0) + 1
            
            # High correlation if multiple positions in same base currency
            max_concentration = max(base_currencies.values())
            correlation_score = (max_concentration - 1) / len(positions)
            
            return min(correlation_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def determine_risk_level(self, concentration: float, volatility: float, 
                           drawdown: float, leverage: float) -> RiskLevel:
        """Determine overall portfolio risk level"""
        try:
            risk_score = 0
            
            # Concentration risk
            if concentration > 0.4:
                risk_score += 3
            elif concentration > 0.3:
                risk_score += 2
            elif concentration > 0.2:
                risk_score += 1
            
            # Volatility risk
            if volatility > 0.05:
                risk_score += 3
            elif volatility > 0.03:
                risk_score += 2
            elif volatility > 0.02:
                risk_score += 1
            
            # Drawdown risk
            if drawdown > 0.15:
                risk_score += 3
            elif drawdown > 0.10:
                risk_score += 2
            elif drawdown > 0.05:
                risk_score += 1
            
            # Leverage risk
            if leverage > 2.0:
                risk_score += 3
            elif leverage > 1.5:
                risk_score += 2
            elif leverage > 1.0:
                risk_score += 1
            
            # Determine level
            if risk_score >= 8:
                return RiskLevel.CRITICAL
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.CRITICAL
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            current_balance = self.get_current_balance()
            positions = self.database.get_positions()
            
            return {
                'current_balance': current_balance,
                'initial_balance': self.initial_balance,
                'total_return': (current_balance - self.initial_balance) / self.initial_balance,
                'risk_metrics': {
                    'var_1d': risk_metrics.var_1d,
                    'var_7d': risk_metrics.var_7d,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'portfolio_volatility': risk_metrics.portfolio_volatility,
                    'concentration_risk': risk_metrics.concentration_risk,
                    'correlation_risk': risk_metrics.correlation_risk,
                    'leverage_ratio': risk_metrics.leverage_ratio,
                    'risk_level': risk_metrics.risk_level.value
                },
                'risk_limits': {
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_position_size': self.max_position_size,
                    'max_daily_loss': self.max_daily_loss,
                    'circuit_breaker_loss': self.circuit_breaker_loss
                },
                'current_positions': len(positions),
                'circuit_breaker_triggered': self.is_circuit_breaker_triggered(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}

# Global instance
_risk_manager = None

def get_risk_manager() -> RiskManager:
    """Get singleton risk manager instance"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager

if __name__ == "__main__":
    # Test risk manager
    rm = get_risk_manager()
    
    # Test position sizing
    position_size = rm.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss=48000
    )
    print(f"Calculated position size: {position_size}")
    
    # Test trade validation
    decision = rm.validate_trade(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        price=50000,
        stop_loss=48000,
        take_profit=54000
    )
    print(f"Trade decision: {decision}")
    
    # Get risk summary
    summary = rm.get_risk_summary()
    print(f"Risk summary: {summary}")