"""
Forward Tester - Paper trading for strategy validation
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import json
import os
from utils.logger import logger as  logging

class ForwardTester:
    """
    Forward Tester - Paper trading for strategy validation
    
    Features:
    - Real-time paper trading
    - Performance tracking
    - Comparison with backtest
    - Signal validation
    - Live metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Trading parameters
        self.initial_capital = config.get("initial_capital", 100000.0)
        self.commission = config.get("commission", 0.001)
        
        # Test duration
        self.test_days = config.get("test_days", 30)
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=self.test_days)
        
        # State
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signals = []
        
        # Strategy
        self.strategy = None
        self.strategy_params = None
        
        # Data cache
        self.data_cache = {}
        
        # Storage
        self.data_dir = config.get("data_dir", "data/forward")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Results
        self.results = {}
        
        logging.info(f"✅ ForwardTester initialized")
    
    def set_strategy(self, strategy_func: Callable, params: Dict[str, Any] = None):
        """
        Set the strategy to test
        """
        self.strategy = strategy_func
        self.strategy_params = params or {}
        logging.info(f"📈 Strategy set for forward testing")
    
    async def run(self, symbols: List[str], 
                 data_provider: Callable,
                 update_interval: int = 60) -> Dict[str, Any]:
        """
        Run forward test
        """
        logging.info(f"🚀 Starting forward test for {len(symbols)} symbols")
        
        start_time = datetime.now()
        
        while datetime.now() < self.end_date:
            for symbol in symbols:
                # Get latest data
                data = await self._get_latest_data(symbol, data_provider)
                
                if data is None:
                    continue
                
                # Generate signal
                signal_data = {
                    'symbol': symbol,
                    'price': data['close'],
                    'position': self.positions.get(symbol, 0),
                    'capital': self.capital,
                    'timestamp': datetime.now(),
                    'data': data
                }
                
                signal = self.strategy(signal_data, self.strategy_params)
                
                if signal != 0:
                    self.signals.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': signal,
                        'price': data['close']
                    })
                
                # Execute signal
                await self._execute_signal(symbol, signal, data['close'])
            
            # Update equity curve
            total_equity = self._calculate_equity(data_provider)
            self.equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'equity': total_equity
            })
            
            # Wait for next update
            await asyncio.sleep(update_interval)
        
        # Calculate results
        results = self._calculate_results(start_time)
        
        self.results = results
        
        return results
    
    async def _get_latest_data(self, symbol: str, data_provider: Callable) -> Optional[Dict]:
        """
        Get latest market data
        """
        try:
            data = await data_provider(symbol)
            
            # Cache data
            if symbol not in self.data_cache:
                self.data_cache[symbol] = []
            
            self.data_cache[symbol].append({
                'timestamp': datetime.now().isoformat(),
                **data
            })
            
            return data
            
        except Exception as e:
            logging.error(f"Error getting data for {symbol}: {e}")
            return None
    
    async def _execute_signal(self, symbol: str, signal: int, price: float):
        """
        Execute trading signal
        """
        current_position = self.positions.get(symbol, 0)
        
        if signal == 1 and current_position == 0:  # Buy
            # Calculate position size (use 20% of capital per position)
            position_value = self.capital * 0.2
            shares = int(position_value / price)
            
            if shares > 0:
                cost = shares * price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
                
                if total_cost <= self.capital:
                    self.positions[symbol] = shares
                    self.capital -= total_cost
                    
                    self.trades.append({
                        'type': 'BUY',
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'price': price,
                        'shares': shares,
                        'cost': total_cost,
                        'commission': commission_cost
                    })
                    
                    logging.info(f"✅ BUY {shares} {symbol} @ ${price:.2f}")
        
        elif signal == -1 and current_position > 0:  # Sell
            shares = current_position
            proceeds = shares * price
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost
            
            # Calculate P&L
            entry_trade = next(
                (t for t in reversed(self.trades) 
                 if t['symbol'] == symbol and t['type'] == 'BUY'),
                None
            )
            
            if entry_trade:
                pnl = net_proceeds - entry_trade['cost']
                pnl_pct = (pnl / entry_trade['cost']) * 100
            else:
                pnl = 0
                pnl_pct = 0
            
            self.capital += net_proceeds
            del self.positions[symbol]
            
            self.trades.append({
                'type': 'SELL',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'price': price,
                'shares': -shares,
                'proceeds': net_proceeds,
                'commission': commission_cost,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            
            logging.info(f"✅ SELL {shares} {symbol} @ ${price:.2f} (P&L: ${pnl:.2f})")
    
    def _calculate_equity(self, data_provider: Callable) -> float:
        """
        Calculate total equity
        """
        position_value = 0
        
        for symbol, shares in self.positions.items():
            # Would need current price here
            # For now, use last known price from cache
            if symbol in self.data_cache and self.data_cache[symbol]:
                last_price = self.data_cache[symbol][-1].get('close', 0)
                position_value += shares * last_price
        
        return self.capital + position_value
    
    def _calculate_results(self, start_time: datetime) -> Dict[str, Any]:
        """
        Calculate forward test results
        """
        duration = (datetime.now() - start_time).total_seconds() / 3600  # hours
        
        # Trade analysis
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        total_pnl = sum(t.get('pnl', 0) for t in sell_trades)
        total_return = (self._calculate_equity(None) - self.initial_capital) / self.initial_capital * 100
        
        # Equity curve analysis
        if len(self.equity_curve) > 1:
            equity_values = [e['equity'] for e in self.equity_curve]
            running_max = np.maximum.accumulate(equity_values)
            drawdown = (running_max - equity_values) / running_max
            max_drawdown = np.max(drawdown) * 100
        else:
            max_drawdown = 0
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_hours': duration,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_equity': self._calculate_equity(None),
            'total_return': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(sell_trades),
            'num_signals': len(self.signals),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'positions_remaining': len(self.positions),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'signals': self.signals
        }
    
    def compare_with_backtest(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare forward test results with backtest
        """
        comparison = {
            'metric': ['Total Return', 'Win Rate', 'Max Drawdown', 'Num Trades'],
            'backtest': [
                backtest_result.get('total_return', 0),
                backtest_result.get('metrics', {}).get('win_rate', 0) * 100,
                backtest_result.get('metrics', {}).get('max_drawdown', 0),
                backtest_result.get('num_trades', 0)
            ],
            'forward': [
                self.results.get('total_return', 0),
                self.results.get('win_rate', 0) * 100,
                self.results.get('max_drawdown', 0),
                self.results.get('num_trades', 0)
            ],
            'difference': [
                self.results.get('total_return', 0) - backtest_result.get('total_return', 0),
                (self.results.get('win_rate', 0) - backtest_result.get('metrics', {}).get('win_rate', 0)) * 100,
                self.results.get('max_drawdown', 0) - backtest_result.get('metrics', {}).get('max_drawdown', 0),
                self.results.get('num_trades', 0) - backtest_result.get('num_trades', 0)
            ]
        }
        
        # Calculate decay factor
        if backtest_result.get('total_return', 0) != 0:
            decay = self.results.get('total_return', 0) / backtest_result.get('total_return', 0)
        else:
            decay = 1
        
        comparison['decay_factor'] = decay
        
        return comparison
    
    def save_results(self, filename: str = None):
        """
        Save forward test results to disk
        """
        if filename is None:
            filename = f"forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logging.info(f"💾 Saved forward test results to {filepath}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")