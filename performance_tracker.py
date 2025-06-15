"""
Performance Tracker for Trading Bot
Real-time tracking and analysis of trading performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    
    # Volume metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_volume: float
    avg_position_size: float
    
    # Efficiency metrics
    expectancy: float
    kelly_percentage: float
    recovery_factor: float
    payoff_ratio: float
    
    def to_dict(self):
        return asdict(self)

class PerformanceTracker:
    def __init__(self, database, initial_capital: float = 10000):
        self.db = database
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades_history = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        self.positions = {}
        self.daily_snapshots = []
        self.metrics_cache = {}
        self.benchmark_data = {}
        
    async def track_trade(self, trade: Dict):
        """Track a completed trade"""
        try:
            # Add to history
            self.trades_history.append({
                **trade,
                'timestamp': datetime.now()
            })
            
            # Update capital
            self.current_capital += trade.get('profit', 0)
            self.equity_curve.append(self.current_capital)
            self.timestamps.append(datetime.now())
            
            # Save to database
            await self._save_trade_to_db(trade)
            
            # Update metrics cache
            self.metrics_cache = {}  # Clear cache to force recalculation
            
            logger.info(f"Tracked trade: {trade['symbol']} - Profit: {trade['profit']}")
            
        except Exception as e:
            logger.error(f"Error tracking trade: {str(e)}")
    
    async def update_position(self, symbol: str, position: Dict):
        """Update current position"""
        try:
            self.positions[symbol] = {
                **position,
                'updated_at': datetime.now()
            }
            
            # Calculate unrealized P&L
            total_equity = self.current_capital
            for pos in self.positions.values():
                if pos.get('unrealized_pnl'):
                    total_equity += pos['unrealized_pnl']
            
            # Update equity curve with current value
            if len(self.equity_curve) > 0:
                self.equity_curve[-1] = total_equity
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
    
    def calculate_metrics(self, period: str = 'all') -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Check cache
            cache_key = f"metrics_{period}"
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            # Filter trades by period
            trades = self._filter_trades_by_period(period)
            if not trades:
                return self._empty_metrics()
            
            # Calculate returns
            returns_metrics = self._calculate_returns_metrics(trades)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics()
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(trades)
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                trades, returns_metrics, risk_metrics
            )
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                **returns_metrics,
                **risk_metrics,
                **trading_metrics,
                **efficiency_metrics
            )
            
            # Cache results
            self.metrics_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._empty_metrics()
    
    def _calculate_returns_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate return-based metrics"""
        if not self.equity_curve:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'monthly_return': 0,
                'daily_return': 0
            }
        
        # Total return
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Time-based returns
        if len(self.timestamps) > 1:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            days = max(time_diff.days, 1)
            
            # Annualized return
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
            
            # Monthly return
            monthly_return = ((1 + total_return) ** (30 / days)) - 1
            
            # Daily return
            equity_array = np.array(self.equity_curve)
            daily_returns = np.diff(equity_array) / equity_array[:-1]
            daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
        else:
            annualized_return = total_return
            monthly_return = total_return / 12
            daily_return = total_return / 365
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'monthly_return': monthly_return,
            'daily_return': daily_return
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics"""
        if len(self.equity_curve) < 2:
            return {
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'current_drawdown': 0,
                'var_95': 0,
                'cvar_95': 0
            }
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else float('inf')
        
        # Drawdown metrics
        cumulative_returns = equity_array / equity_array[0] - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        current_drawdown = drawdown[-1]
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        # Calmar ratio
        if max_drawdown != 0:
            annual_return = (equity_array[-1] / equity_array[0]) ** (252 / len(returns)) - 1
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_trading_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trading-specific metrics"""
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade_duration': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_volume': 0,
                'avg_position_size': 0
            }
        
        # Separate winning and losing trades
        profits = [t['profit'] for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average win/loss
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Best/worst trade
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0
        
        # Trade duration
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                durations.append(duration)
        
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Volume metrics
        total_volume = sum(t.get('size', 0) * t.get('entry_price', 0) for t in trades)
        avg_position_size = total_volume / len(trades) if trades else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_volume': total_volume,
            'avg_position_size': avg_position_size
        }
    
    def _calculate_efficiency_metrics(self, trades: List[Dict], 
                                    returns_metrics: Dict, 
                                    risk_metrics: Dict) -> Dict:
        """Calculate efficiency metrics"""
        if not trades:
            return {
                'expectancy': 0,
                'kelly_percentage': 0,
                'recovery_factor': 0,
                'payoff_ratio': 0
            }
        
        # Expectancy
        profits = [t['profit'] for t in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Kelly percentage
        if avg_loss > 0 and win_rate > 0:
            payoff_ratio = avg_win / avg_loss
            kelly_percentage = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
            kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
        else:
            kelly_percentage = 0
            payoff_ratio = 0
        
        # Recovery factor
        max_drawdown = abs(risk_metrics['max_drawdown'])
        total_return = returns_metrics['total_return']
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        return {
            'expectancy': expectancy,
            'kelly_percentage': kelly_percentage,
            'recovery_factor': recovery_factor,
            'payoff_ratio': payoff_ratio
        }
    
    def generate_report(self, period: str = 'all', format: str = 'dict') -> Any:
        """Generate comprehensive performance report"""
        try:
            metrics = self.calculate_metrics(period)
            
            report = {
                'summary': {
                    'period': period,
                    'generated_at': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'total_trades': len(self.trades_history)
                },
                'metrics': metrics.to_dict(),
                'equity_curve': {
                    'values': self.equity_curve,
                    'timestamps': [t.isoformat() for t in self.timestamps]
                },
                'positions': self.positions,
                'recent_trades': self.trades_history[-10:] if self.trades_history else []
            }
            
            if format == 'json':
                return json.dumps(report, indent=2, default=str)
            elif format == 'dataframe':
                return self._report_to_dataframe(report)
            else:
                return report
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {}
    
    def plot_performance(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Generate performance charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from io import BytesIO
            import base64
            
            charts = {}
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Equity Curve
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.timestamps, self.equity_curve, linewidth=2)
            ax.fill_between(self.timestamps, self.initial_capital, self.equity_curve, 
                           alpha=0.3, where=[e >= self.initial_capital for e in self.equity_curve],
                           color='green', label='Profit')
            ax.fill_between(self.timestamps, self.initial_capital, self.equity_curve,
                           alpha=0.3, where=[e < self.initial_capital for e in self.equity_curve],
                           color='red', label='Loss')
            ax.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Equity Curve', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(f"{save_path}_equity_curve.png", dpi=300, bbox_inches='tight')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            charts['equity_curve'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # 2. Drawdown Chart
            equity_array = np.array(self.equity_curve)
            cumulative_returns = equity_array / equity_array[0] - 1
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(self.timestamps, 0, drawdown, color='red', alpha=0.3)
            ax.plot(self.timestamps, drawdown, color='red', linewidth=2)
            ax.set_title('Drawdown Chart', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(f"{save_path}_drawdown.png", dpi=300, bbox_inches='tight')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            charts['drawdown'] = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # 3. Returns Distribution
            if len(self.equity_curve) > 1:
                returns = np.diff(equity_array) / equity_array[:-1] * 100
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=np.mean(returns), color='green', linestyle='--', 
                          alpha=0.7, label=f'Mean: {np.mean(returns):.2f}%')
                ax.set_title('Returns Distribution', fontsize=16)
                ax.set_xlabel('Return (%)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(f"{save_path}_returns_dist.png", dpi=300, bbox_inches='tight')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                charts['returns_distribution'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            # 4. Monthly Returns Heatmap
            if len(self.trades_history) > 0:
                monthly_returns = self._calculate_monthly_returns()
                
                if not monthly_returns.empty:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn',
                               center=0, cbar_kws={'label': 'Return (%)'}, ax=ax)
                    ax.set_title('Monthly Returns Heatmap', fontsize=16)
                    
                    if save_path:
                        plt.savefig(f"{save_path}_monthly_returns.png", dpi=300, bbox_inches='tight')
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    charts['monthly_returns'] = base64.b64encode(buffer.read()).decode()
                    plt.close()
            
            # 5. Trade Analysis
            if self.trades_history:
                profits = [t['profit'] for t in self.trades_history]
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Profit/Loss by trade
                ax = axes[0, 0]
                colors = ['green' if p > 0 else 'red' for p in profits]
                ax.bar(range(len(profits)), profits, color=colors, alpha=0.7)
                ax.set_title('Profit/Loss by Trade')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Profit ($)')
                ax.grid(True, alpha=0.3)
                
                # Cumulative profit
                ax = axes[0, 1]
                cumulative_profit = np.cumsum(profits)
                ax.plot(cumulative_profit, linewidth=2)
                ax.fill_between(range(len(cumulative_profit)), 0, cumulative_profit,
                               alpha=0.3, color='green')
                ax.set_title('Cumulative Profit')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('Cumulative Profit ($)')
                ax.grid(True, alpha=0.3)
                
                # Win/Loss ratio
                ax = axes[1, 0]
                wins = len([p for p in profits if p > 0])
                losses = len([p for p in profits if p <= 0])
                ax.pie([wins, losses], labels=['Wins', 'Losses'], 
                      colors=['green', 'red'], autopct='%1.1f%%')
                ax.set_title('Win/Loss Ratio')
                
                # Profit distribution
                ax = axes[1, 1]
                ax.hist(profits, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.set_title('Profit Distribution')
                ax.set_xlabel('Profit ($)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(f"{save_path}_trade_analysis.png", dpi=300, bbox_inches='tight')
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                charts['trade_analysis'] = base64.b64encode(buffer.read()).decode()
                plt.close()
            
            return charts
            
        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}")
            return {}
    
    def compare_to_benchmark(self, benchmark_symbol: str = 'BTC/USDT') -> Dict:
        """Compare performance to benchmark"""
        try:
            # This would typically fetch benchmark data from an external source
            # For now, returning a placeholder
            
            if len(self.equity_curve) < 2:
                return {}
            
            # Calculate relative performance
            strategy_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
            
            # Placeholder benchmark return (would be fetched from real data)
            benchmark_return = 0.5  # 50% return
            
            alpha = strategy_return - benchmark_return
            
            # Calculate correlation (placeholder)
            correlation = 0.7
            
            return {
                'benchmark': benchmark_symbol,
                'strategy_return': strategy_return,
                'benchmark_return': benchmark_return,
                'alpha': alpha,
                'correlation': correlation,
                'information_ratio': alpha / 0.15 if alpha > 0 else 0  # Placeholder tracking error
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmark: {str(e)}")
            return {}
    
    def _filter_trades_by_period(self, period: str) -> List[Dict]:
        """Filter trades by time period"""
        if period == 'all' or not self.trades_history:
            return self.trades_history
        
        now = datetime.now()
        
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        elif period == 'quarter':
            start_date = now - timedelta(days=90)
        elif period == 'year':
            start_date = now - timedelta(days=365)
        else:
            return self.trades_history
        
        return [t for t in self.trades_history if t.get('timestamp', now) >= start_date]
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns for heatmap"""
        try:
            if len(self.timestamps) < 2:
                return pd.DataFrame()
            
            # Create DataFrame with equity and timestamps
            df = pd.DataFrame({
                'equity': self.equity_curve,
                'timestamp': self.timestamps
            })
            
            # Calculate daily returns
            df['returns'] = df['equity'].pct_change() * 100
            
            # Extract year and month
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            
            # Calculate monthly returns
            monthly = df.groupby(['year', 'month'])['returns'].sum()
            
            # Pivot to create heatmap format
            monthly_pivot = monthly.reset_index().pivot(
                index='month', columns='year', values='returns'
            )
            
            # Add month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pivot.index = [month_names[i-1] for i in monthly_pivot.index]
            
            return monthly_pivot
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {str(e)}")
            return pd.DataFrame()
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object"""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, monthly_return=0, daily_return=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, current_drawdown=0, var_95=0, cvar_95=0,
            win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
            best_trade=0, worst_trade=0, avg_trade_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            total_volume=0, avg_position_size=0,
            expectancy=0, kelly_percentage=0, recovery_factor=0, payoff_ratio=0
        )
    
    def _report_to_dataframe(self, report: Dict) -> pd.DataFrame:
        """Convert report to DataFrame format"""
        try:
            # Flatten metrics
            metrics_df = pd.DataFrame([report['metrics']])
            
            # Add summary info
            for key, value in report['summary'].items():
                if key != 'generated_at':
                    metrics_df[key] = value
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error converting report to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    async def _save_trade_to_db(self, trade: Dict):
        """Save trade to database"""
        try:
            query = """
                INSERT INTO trades (
                    symbol, side, entry_time, exit_time, entry_price, exit_price,
                    size, profit, return_pct, trade_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, (
                trade.get('symbol'),
                trade.get('side'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('size'),
                trade.get('profit'),
                trade.get('return', 0),
                trade.get('type', 'manual'),
                json.dumps(trade.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {str(e)}")
    
    async def save_snapshot(self):
        """Save daily performance snapshot"""
        try:
            metrics = self.calculate_metrics('today')
            
            snapshot = {
                'date': datetime.now().date(),
                'equity': self.current_capital,
                'metrics': metrics.to_dict(),
                'positions': len(self.positions),
                'trades_today': len(self._filter_trades_by_period('today'))
            }
            
            self.daily_snapshots.append(snapshot)
            
            # Save to database
            query = """
                INSERT INTO performance_snapshots (
                    date, equity, total_return, sharpe_ratio, max_drawdown,
                    win_rate, total_trades, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, (
                snapshot['date'],
                snapshot['equity'],
                metrics.total_return,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.win_rate,
                metrics.total_trades,
                json.dumps(snapshot)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("Saved daily performance snapshot")
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
    
    def export_trades(self, format: str = 'csv', filepath: Optional[str] = None) -> Optional[str]:
        """Export trades history"""
        try:
            if not self.trades_history:
                logger.warning("No trades to export")
                return None
            
            df = pd.DataFrame(self.trades_history)
            
            if format == 'csv':
                if filepath:
                    df.to_csv(filepath, index=False)
                    return filepath
                else:
                    return df.to_csv(index=False)
            
            elif format == 'json':
                if filepath:
                    df.to_json(filepath, orient='records', indent=2)
                    return filepath
                else:
                    return df.to_json(orient='records', indent=2)
            
            elif format == 'excel':
                if not filepath:
                    filepath = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Trades', index=False)
                    
                    # Add metrics sheet
                    metrics_df = pd.DataFrame([self.calculate_metrics().to_dict()])
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                return filepath
            
        except Exception as e:
            logger.error(f"Error exporting trades: {str(e)}")
            return None