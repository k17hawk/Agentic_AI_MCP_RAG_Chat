"""
Simulation Engine - Historical backtesting engine
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
from utils.logger import logger as  logging

class SimulationEngine:
    """
    Simulation Engine - Historical backtesting engine
    
    Features:
    - Historical price simulation
    - Realistic slippage and commissions
    - Multiple position management
    - Performance metrics calculation
    - Equity curve tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Trading parameters
        self.initial_capital = config.get("initial_capital", 100000.0)
        self.commission = config.get("commission", 0.001)  # 0.1%
        self.slippage = config.get("slippage", 0.001)  # 0.1%
        
        # Position limits
        self.max_positions = config.get("max_positions", 10)
        self.max_position_size = config.get("max_position_size", 0.25)  # 25% of capital
        
        # Storage
        self.data_dir = config.get("data_dir", "data/backtests")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        logging.info(f"✅ SimulationEngine initialized")
    
    def run_backtest(self, strategy_func: Callable, 
                    data: pd.DataFrame,
                    symbol: str,
                    params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a single backtest
        """
        params = params or {}
        
        # Initialize state
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Track performance
        daily_returns = []
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            prev_price = data['Close'].iloc[i-1]
            timestamp = data.index[i]
            
            # Get signal from strategy
            signal_data = {
                'price': current_price,
                'prev_price': prev_price,
                'position': position,
                'capital': capital,
                'date': timestamp,
                'indicators': {col: data[col].iloc[i] for col in data.columns if col != 'Close'}
            }
            
            signal = strategy_func(signal_data, params)
            
            # Execute trades based on signal
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                position_value = capital * 0.95  # Use 95% of capital
                shares = int(position_value / current_price)
                
                if shares > 0:
                    # Apply slippage
                    execution_price = current_price * (1 + self.slippage)
                    cost = shares * execution_price
                    commission_cost = cost * self.commission
                    total_cost = cost + commission_cost
                    
                    if total_cost <= capital:
                        position = shares
                        entry_price = execution_price
                        capital -= total_cost
                        
                        trades.append({
                            'type': 'BUY',
                            'date': timestamp,
                            'price': execution_price,
                            'shares': shares,
                            'cost': total_cost,
                            'commission': commission_cost
                        })
            
            elif signal == -1 and position > 0:  # Sell signal
                # Close position
                execution_price = current_price * (1 - self.slippage)
                proceeds = position * execution_price
                commission_cost = proceeds * self.commission
                net_proceeds = proceeds - commission_cost
                
                # Calculate P&L
                pnl = net_proceeds - (position * entry_price)
                pnl_pct = (pnl / (position * entry_price)) * 100
                
                capital += net_proceeds
                
                trades.append({
                    'type': 'SELL',
                    'date': timestamp,
                    'price': execution_price,
                    'shares': -position,
                    'proceeds': net_proceeds,
                    'commission': commission_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                position = 0
                entry_price = 0
            
            # Calculate equity
            equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append({
                'date': timestamp,
                'equity': equity,
                'position': position
            })
            
            # Calculate daily return
            if len(equity_curve) > 1:
                daily_return = (equity - equity_curve[-2]['equity']) / equity_curve[-2]['equity']
                daily_returns.append(daily_return)
        
        # Close any remaining position at the end
        if position > 0:
            final_price = data['Close'].iloc[-1]
            execution_price = final_price * (1 - self.slippage)
            proceeds = position * execution_price
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost
            
            pnl = net_proceeds - (position * entry_price)
            pnl_pct = (pnl / (position * entry_price)) * 100
            
            capital += net_proceeds
            
            trades.append({
                'type': 'SELL (FINAL)',
                'date': data.index[-1],
                'price': execution_price,
                'shares': -position,
                'proceeds': net_proceeds,
                'commission': commission_cost,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, daily_returns)
        
        result = {
            'symbol': symbol,
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'num_trades': len([t for t in trades if t['type'] in ['SELL', 'SELL (FINAL)']]),
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results[f"{symbol}_{datetime.now().timestamp()}"] = result
        
        return result
    
    def _calculate_metrics(self, equity_curve: List[Dict], 
                          trades: List[Dict],
                          daily_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate performance metrics
        """
        if not daily_returns:
            return {}
        
        returns = np.array(daily_returns)
        
        # Win/Loss analysis
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        equity_values = [e['equity'] for e in equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (running_max - equity_values) / running_max
        max_drawdown = np.max(drawdown) * 100
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': len([t for t in trades if t['type'] in ['SELL', 'SELL (FINAL)']]),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_daily_return': np.mean(returns) * 100,
            'std_daily_return': np.std(returns) * 100
        }
    
    def optimize_parameters(self, strategy_func: Callable,
                           data: pd.DataFrame,
                           symbol: str,
                           param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        """
        best_result = None
        best_return = -float('inf')
        
        # Generate all parameter combinations
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Run backtest
            result = self.run_backtest(strategy_func, data, symbol, params)
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_result = result
        
        return {
            'best_params': best_result['params'],
            'best_return': best_return,
            'best_result': best_result
        }
    
    def compare_strategies(self, strategies: Dict[str, Callable],
                          data: pd.DataFrame,
                          symbol: str) -> Dict[str, Any]:
        """
        Compare multiple strategies
        """
        results = {}
        
        for name, strategy in strategies.items():
            result = self.run_backtest(strategy, data, symbol)
            results[name] = {
                'total_return': result['total_return'],
                'sharpe': result['metrics'].get('sharpe_ratio', 0),
                'max_drawdown': result['metrics'].get('max_drawdown', 0),
                'win_rate': result['metrics'].get('win_rate', 0),
                'num_trades': result['num_trades']
            }
        
        # Find best by different metrics
        best_return = max(results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
        best_win_rate = max(results.items(), key=lambda x: x[1]['win_rate'])
        
        return {
            'results': results,
            'best_by_return': best_return,
            'best_by_sharpe': best_sharpe,
            'best_by_win_rate': best_win_rate
        }
    
    def save_results(self, result_id: str = None):
        """
        Save backtest results to disk
        """
        if result_id:
            results_to_save = {result_id: self.results.get(result_id)}
        else:
            results_to_save = self.results
        
        filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            logging.info(f"💾 Saved backtest results to {filepath}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")