"""
P&L Calculator - Calculates profit and loss
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta
from utils.logger import logger as  logging
from zoneinfo import ZoneInfo

class PNLCalculator:
    """
    P&L Calculator - Calculates profit and loss metrics
    
    Features:
    - Realized P&L
    - Unrealized P&L
    - Total P&L
    - Daily P&L
    - Cumulative P&L
    - P&L by symbol/strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.include_commissions = config.get("include_commissions", True)
        self.include_slippage = config.get("include_slippage", True)
        
        logging.info(f"✅ PNLCalculator initialized")
    
    def calculate_realized_pnl(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate realized P&L from completed trades
        """
        if not trades:
            return {
                "total": 0.0,
                "by_symbol": {},
                "by_strategy": {},
                "by_month": {},
                "winning_trades": 0,
                "losing_trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0
            }
        
        realized_trades = [t for t in trades if t.get('exit_time') is not None]
        
        total_pnl = 0.0
        by_symbol = {}
        by_strategy = {}
        by_month = {}
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        for trade in realized_trades:
            pnl = trade.get('pnl', 0)
            
            # Adjust for commissions and slippage
            if self.include_commissions:
                pnl -= trade.get('commission', 0)
            if self.include_slippage:
                pnl -= trade.get('slippage_cost', 0)
            
            total_pnl += pnl
            
            # By symbol
            symbol = trade.get('symbol', 'unknown')
            if symbol not in by_symbol:
                by_symbol[symbol] = 0.0
            by_symbol[symbol] += pnl
            
            # By strategy
            strategy = trade.get('strategy', 'unknown')
            if strategy not in by_strategy:
                by_strategy[strategy] = 0.0
            by_strategy[strategy] += pnl
            
            # By month
            exit_time = datetime.fromisoformat(trade['exit_time'])
            month_key = exit_time.strftime('%Y-%m')
            if month_key not in by_month:
                by_month[month_key] = 0.0
            by_month[month_key] += pnl
            
            # Win/loss tracking
            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
            elif pnl < 0:
                losing_trades += 1
                gross_loss += abs(pnl)
        
        return {
            "total": float(total_pnl),
            "by_symbol": {k: float(v) for k, v in by_symbol.items()},
            "by_strategy": {k: float(v) for k, v in by_strategy.items()},
            "by_month": {k: float(v) for k, v in by_month.items()},
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "net_profit": float(total_pnl),
            "num_trades": len(realized_trades)
        }
    
    def calculate_unrealized_pnl(self, positions: List[Dict[str, Any]], 
                                 current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate unrealized P&L from open positions
        """
        if not positions:
            return {
                "total": 0.0,
                "by_symbol": {},
                "winning_positions": 0,
                "losing_positions": 0
            }
        
        total_unrealized = 0.0
        by_symbol = {}
        winning_positions = 0
        losing_positions = 0
        
        for position in positions:
            symbol = position['symbol']
            quantity = position['quantity']
            avg_price = position['avg_price']
            
            current_price = current_prices.get(symbol, avg_price)
            
            if position.get('side', 'LONG') == 'LONG':
                unrealized = (current_price - avg_price) * quantity
            else:  # SHORT
                unrealized = (avg_price - current_price) * quantity
            
            total_unrealized += unrealized
            by_symbol[symbol] = unrealized
            
            if unrealized > 0:
                winning_positions += 1
            elif unrealized < 0:
                losing_positions += 1
        
        return {
            "total": float(total_unrealized),
            "by_symbol": {k: float(v) for k, v in by_symbol.items()},
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "num_positions": len(positions)
        }
    
    def calculate_total_pnl(self, realized: Dict[str, Any], 
                           unrealized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total P&L (realized + unrealized)
        """
        total = realized.get('total', 0) + unrealized.get('total', 0)
        
        return {
            "total": float(total),
            "realized": realized,
            "unrealized": unrealized,
            "realized_pct": realized.get('total', 0) / total if total != 0 else 0,
            "unrealized_pct": unrealized.get('total', 0) / total if total != 0 else 0
        }
    
    

    def calculate_daily_pnl(self, trades: List[Dict[str, Any]], 
                       days: int = 30) -> Dict[str, Any]:
        end_date = datetime.now(ZoneInfo("UTC"))
        start_date = end_date - timedelta(days=days)
        
        daily_pnl = {}
        
        for trade in trades:
            if 'exit_time' not in trade:
                continue
            
            exit_time = datetime.fromisoformat(trade['exit_time'])
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=ZoneInfo("UTC"))
            else:
                exit_time = exit_time.astimezone(ZoneInfo("UTC"))
            
            if exit_time < start_date:
                continue
            
            date_key = exit_time.strftime('%Y-%m-%d')
            daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + trade.get('pnl', 0)
        
        # Sort after the loop
        sorted_days = sorted(daily_pnl.items())
        
        if sorted_days:
            pnl_values = [v for _, v in sorted_days]
            avg_daily = np.mean(pnl_values)
            std_daily = np.std(pnl_values)
            best_day = max(pnl_values)
            worst_day = min(pnl_values)
        else:
            avg_daily = 0.0
            std_daily = 0.0
            best_day = 0.0
            worst_day = 0.0
        
        return {
            "daily_pnl": dict(sorted_days),
            "average_daily": float(avg_daily),
            "std_daily": float(std_daily),
            "best_day": float(best_day),
            "worst_day": float(worst_day),
            "num_days": len(daily_pnl),
            "period_days": days
        }

    def calculate_cumulative_pnl(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate cumulative P&L over time
        """
        if not trades:
            return []
        
        # Sort by exit time
        closed_trades = [t for t in trades if t.get('exit_time') is not None]
        closed_trades.sort(key=lambda x: x['exit_time'])
        
        cumulative = []
        running_total = 0.0
        
        for trade in closed_trades:
            running_total += trade.get('pnl', 0)
            cumulative.append({
                'date': trade['exit_time'],
                'pnl': trade.get('pnl', 0),
                'cumulative_pnl': running_total
            })
        
        return cumulative
    
    def get_summary(self, trades: List[Dict[str, Any]], 
                   positions: List[Dict[str, Any]] = None,
                   current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get comprehensive P&L summary
        """
        realized = self.calculate_realized_pnl(trades)
        
        if positions and current_prices:
            unrealized = self.calculate_unrealized_pnl(positions, current_prices)
            total = self.calculate_total_pnl(realized, unrealized)
        else:
            unrealized = {"total": 0, "by_symbol": {}}
            total = {"total": realized['total'], "realized": realized, "unrealized": unrealized}
        
        daily = self.calculate_daily_pnl(trades)
        cumulative = self.calculate_cumulative_pnl(trades)
        
        return {
            "total_pnl": total['total'],
            "realized_pnl": realized['total'],
            "unrealized_pnl": unrealized['total'],
            "realized": realized,
            "unrealized": unrealized,
            "daily": daily,
            "cumulative": cumulative,
            "summary": {
                "total_trades": realized['num_trades'],
                "winning_trades": realized['winning_trades'],
                "losing_trades": realized['losing_trades'],
                "win_rate": realized['winning_trades'] / realized['num_trades'] if realized['num_trades'] > 0 else 0,
                "gross_profit": realized['gross_profit'],
                "gross_loss": realized['gross_loss'],
                "net_profit": realized['net_profit'],
                "profit_factor": realized['gross_profit'] / realized['gross_loss'] if realized['gross_loss'] > 0 else float('inf')
            }
        }