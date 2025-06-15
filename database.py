"""
Complete Database Layer for Trading Bot
Integrates with your existing trading_bot.db
"""

import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import contextlib

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    strategy: str = "manual"
    profit_loss: float = 0.0
    fees: float = 0.0
    status: str = "filled"

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    side: str = "long"
    entry_time: datetime = None

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    high_24h: float = 0.0
    low_24h: float = 0.0
    change_24h: float = 0.0

class TradingDatabase:
    """Database manager for the trading bot using existing trading_bot.db"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure all required tables exist"""
        with self.get_connection() as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    strategy TEXT DEFAULT 'manual',
                    profit_loss REAL DEFAULT 0.0,
                    fees REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'filled'
                )
            """)
            
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    side TEXT DEFAULT 'long',
                    entry_time TEXT
                )
            """)
            
            # Market data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    high_24h REAL DEFAULT 0.0,
                    low_24h REAL DEFAULT 0.0,
                    change_24h REAL DEFAULT 0.0,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Strategies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY,
                    enabled BOOLEAN DEFAULT 1,
                    parameters TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    performance_data TEXT
                )
            """)
            
            # Portfolio snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT PRIMARY KEY,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                )
            """)
            
            # Alerts and notifications
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    symbol TEXT,
                    severity TEXT DEFAULT 'info',
                    is_read BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextlib.contextmanager
    def get_connection(self):
        """Thread-safe database connection"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    # Trade operations
    def save_trade(self, trade: Trade) -> bool:
        """Save a trade to database"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades 
                    (id, symbol, side, quantity, price, timestamp, strategy, profit_loss, fees, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.symbol, trade.side, trade.quantity, trade.price,
                    trade.timestamp.isoformat(), trade.strategy, trade.profit_loss,
                    trade.fees, trade.status
                ))
                conn.commit()
                logger.info(f"Saved trade: {trade.id}")
                return True
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def get_trades(self, symbol: str = None, limit: int = 100) -> List[Trade]:
        """Get trades from database"""
        try:
            with self.get_connection() as conn:
                if symbol:
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                        (symbol, limit)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                        (limit,)
                    )
                
                trades = []
                for row in cursor.fetchall():
                    trade = Trade(
                        id=row['id'],
                        symbol=row['symbol'],
                        side=row['side'],
                        quantity=row['quantity'],
                        price=row['price'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        strategy=row['strategy'],
                        profit_loss=row['profit_loss'],
                        fees=row['fees'],
                        status=row['status']
                    )
                    trades.append(trade)
                
                return trades
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    # Position operations
    def save_position(self, position: Position) -> bool:
        """Save position to database"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO positions 
                    (symbol, quantity, entry_price, current_price, unrealized_pnl, side, entry_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.symbol, position.quantity, position.entry_price,
                    position.current_price, position.unrealized_pnl, position.side,
                    position.entry_time.isoformat() if position.entry_time else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving position: {e}")
            return False
    
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM positions WHERE quantity != 0")
                
                positions = []
                for row in cursor.fetchall():
                    position = Position(
                        symbol=row['symbol'],
                        quantity=row['quantity'],
                        entry_price=row['entry_price'],
                        current_price=row['current_price'],
                        unrealized_pnl=row['unrealized_pnl'],
                        side=row['side'],
                        entry_time=datetime.fromisoformat(row['entry_time']) if row['entry_time'] else None
                    )
                    positions.append(position)
                
                return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            with self.get_connection() as conn:
                conn.execute("UPDATE positions SET quantity = 0 WHERE symbol = ?", (symbol,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    # Market data operations
    def save_market_data(self, market_data: MarketData) -> bool:
        """Save market data"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, price, volume, high_24h, low_24h, change_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_data.symbol, market_data.timestamp.isoformat(),
                    market_data.price, market_data.volume, market_data.high_24h,
                    market_data.low_24h, market_data.change_24h
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            return False
    
    def get_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get market data as DataFrame"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(
                    query, conn, 
                    params=(symbol, cutoff_time.isoformat()),
                    parse_dates=['timestamp']
                )
                return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    # Strategy operations
    def save_strategy(self, name: str, enabled: bool, parameters: dict, performance_data: dict = None) -> bool:
        """Save strategy configuration"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO strategies 
                    (name, enabled, parameters, performance_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    name, enabled, json.dumps(parameters),
                    json.dumps(performance_data) if performance_data else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            return False
    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM strategies")
                
                strategies = []
                for row in cursor.fetchall():
                    strategy = {
                        'name': row['name'],
                        'enabled': bool(row['enabled']),
                        'parameters': json.loads(row['parameters']),
                        'created_at': row['created_at'],
                        'performance_data': json.loads(row['performance_data']) if row['performance_data'] else None
                    }
                    strategies.append(strategy)
                
                return strategies
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []
    
    # Analytics operations
    def get_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_connection() as conn:
                # Get trades in period
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(profit_loss) as total_pnl,
                        AVG(profit_loss) as avg_pnl,
                        MAX(profit_loss) as best_trade,
                        MIN(profit_loss) as worst_trade
                    FROM trades 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                
                stats = cursor.fetchone()
                
                # Get current positions
                cursor = conn.execute("""
                    SELECT 
                        SUM(quantity * current_price) as positions_value,
                        SUM(unrealized_pnl) as unrealized_pnl,
                        COUNT(*) as positions_count
                    FROM positions 
                    WHERE quantity != 0
                """)
                
                positions = cursor.fetchone()
                
                total_trades = stats['total_trades'] or 0
                winning_trades = stats['winning_trades'] or 0
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_pnl': stats['total_pnl'] or 0.0,
                    'avg_pnl': stats['avg_pnl'] or 0.0,
                    'best_trade': stats['best_trade'] or 0.0,
                    'worst_trade': stats['worst_trade'] or 0.0,
                    'positions_value': positions['positions_value'] or 0.0,
                    'unrealized_pnl': positions['unrealized_pnl'] or 0.0,
                    'positions_count': positions['positions_count'] or 0
                }
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def save_portfolio_snapshot(self, total_value: float, cash_balance: float,
                               positions_value: float, unrealized_pnl: float,
                               realized_pnl: float) -> bool:
        """Save portfolio snapshot"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO portfolio_snapshots 
                    (timestamp, total_value, cash_balance, positions_value, unrealized_pnl, realized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(), total_value, cash_balance,
                    positions_value, unrealized_pnl, realized_pnl
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return False
    
    def create_alert(self, alert_type: str, message: str, symbol: str = None, 
                    severity: str = 'info') -> bool:
        """Create an alert"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO alerts (type, message, symbol, severity)
                    VALUES (?, ?, ?, ?)
                """, (alert_type, message, symbol, severity))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return False
    
    def get_alerts(self, unread_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts"""
        try:
            with self.get_connection() as conn:
                if unread_only:
                    cursor = conn.execute("""
                        SELECT * FROM alerts 
                        WHERE is_read = 0 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM alerts 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                
                alerts = []
                for row in cursor.fetchall():
                    alert = {
                        'id': row['id'],
                        'type': row['type'],
                        'message': row['message'],
                        'symbol': row['symbol'],
                        'severity': row['severity'],
                        'is_read': bool(row['is_read']),
                        'created_at': row['created_at']
                    }
                    alerts.append(alert)
                
                return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def mark_alert_read(self, alert_id: int) -> bool:
        """Mark alert as read"""
        try:
            with self.get_connection() as conn:
                conn.execute("UPDATE alerts SET is_read = 1 WHERE id = ?", (alert_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error marking alert read: {e}")
            return False
    
    def export_data(self, table_name: str, filename: str = None) -> str:
        """Export table data to CSV"""
        if not filename:
            filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                filepath = f"data/{filename}"
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {table_name} to {filepath}")
                return filepath
        except Exception as e:
            logger.error(f"Error exporting {table_name}: {e}")
            return ""
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Table row counts
                tables = ['trades', 'positions', 'market_data', 'strategies', 'alerts']
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database size
                try:
                    db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
                    stats['database_size_mb'] = round(db_size, 2)
                except:
                    stats['database_size_mb'] = 0
                
                # Latest trade
                cursor = conn.execute("SELECT timestamp FROM trades ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                stats['latest_trade'] = result[0] if result else None
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# Singleton instance
_db_instance = None

def get_database() -> TradingDatabase:
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradingDatabase()
    return _db_instance