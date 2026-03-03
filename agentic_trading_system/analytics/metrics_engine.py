"""
Metrics Engine - Main orchestrator for all performance metrics
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.logger import logger as  logging
from agents.base_agent import BaseAgent, AgentMessage

# Import all metrics
from analytics.performance_metrics.pnl_calculator import PNLCalculator
from analytics.performance_metrics.sharpe_ratio import SharpeRatio
from analytics.performance_metrics.sortino_ratio import SortinoRatio
from analytics.performance_metrics.calmar_ratio import CalmarRatio
from analytics.performance_metrics.win_rate import WinRate
from analytics.performance_metrics.profit_factor import ProfitFactor
from analytics.performance_metrics.max_drawdown import MaxDrawdown
from analytics.performance_metrics.recovery_factor import RecoveryFactor

class MetricsEngine(BaseAgent):
    """
    Metrics Engine - Main orchestrator for all performance metrics
    
    Responsibilities:
    - Calculate comprehensive performance metrics
    - Generate performance reports
    - Track metrics over time
    - Compare strategies
    - Alert on metric thresholds
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Performance metrics calculation and analysis",
            config=config
        )
        
        # Initialize all metric calculators
        self.pnl = PNLCalculator(config.get("pnl_config", {}))
        self.sharpe = SharpeRatio(config.get("sharpe_config", {}))
        self.sortino = SortinoRatio(config.get("sortino_config", {}))
        self.calmar = CalmarRatio(config.get("calmar_config", {}))
        self.win_rate = WinRate(config.get("win_rate_config", {}))
        self.profit_factor = ProfitFactor(config.get("profit_factor_config", {}))
        self.max_drawdown = MaxDrawdown(config.get("max_drawdown_config", {}))
        self.recovery = RecoveryFactor(config.get("recovery_config", {}))
        
        # Metrics storage
        self.metrics_history = []
        self.daily_metrics = {}
        self.thresholds = config.get("thresholds", {
            "min_sharpe": 1.0,
            "min_win_rate": 0.5,
            "min_profit_factor": 1.5,
            "max_drawdown": 20.0
        })
        
        logging.info(f"✅ MetricsEngine initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process metrics-related requests
        """
        msg_type = message.message_type
        
        if msg_type == "calculate_metrics":
            # Calculate comprehensive metrics
            data = message.content
            trades = data.get("trades", [])
            equity_curve = data.get("equity_curve", [])
            positions = data.get("positions", [])
            current_prices = data.get("current_prices", {})
            
            metrics = await self.calculate_all_metrics(
                trades, equity_curve, positions, current_prices
            )
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="metrics_result",
                content=metrics
            )
        
        elif msg_type == "compare_strategies":
            # Compare multiple strategies
            strategies = message.content.get("strategies", {})
            comparison = await self.compare_strategies(strategies)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="comparison_result",
                content=comparison
            )
        
        elif msg_type == "get_daily_metrics":
            # Get metrics for a specific date
            date = message.content.get("date")
            metrics = self.daily_metrics.get(date, {})
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="daily_metrics",
                content={"date": date, "metrics": metrics}
            )
        
        elif msg_type == "check_thresholds":
            # Check if metrics exceed thresholds
            metrics = message.content.get("metrics", {})
            alerts = self.check_thresholds(metrics)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="threshold_alerts",
                content={"alerts": alerts}
            )
        
        return None
    
    async def calculate_all_metrics(self, trades: List[Dict], 
                                   equity_curve: List[float],
                                   positions: List[Dict] = None,
                                   current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate all performance metrics
        """
        logging.info("📊 Calculating comprehensive performance metrics")
        
        # P&L metrics
        pnl_metrics = self.pnl.get_summary(trades, positions, current_prices)
        
        # Return-based metrics (need equity curve or returns)
        returns = self._get_returns_from_equity(equity_curve) if equity_curve else []
        
        sharpe_metrics = self.sharpe.calculate(returns) if returns else {}
        sortino_metrics = self.sortino.calculate(returns) if returns else {}
        
        # Drawdown metrics
        dd_metrics = self.max_drawdown.calculate(equity_curve) if equity_curve else {}
        dd_from_returns = self.max_drawdown.calculate_from_returns(returns) if returns else {}
        
        # Win rate metrics
        win_rate_metrics = self.win_rate.calculate(trades)
        win_rate_by_strategy = self.win_rate.calculate_by_strategy(trades)
        
        # Profit factor
        pf_metrics = self.profit_factor.calculate(trades)
        pf_by_strategy = self.profit_factor.calculate_by_strategy(trades)
        
        # Calmar ratio
        calmar_metrics = self.calmar.calculate(
            returns, 
            dd_metrics.get('max_drawdown_pct', 0) / 100
        ) if returns else {}
        
        # Recovery metrics
        recovery_metrics = self.recovery.get_recovery_summary(trades, equity_curve) if equity_curve else {}
        
        # Combine all metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_trades": len(trades),
                "open_positions": len(positions) if positions else 0,
                "period_start": trades[0]['entry_time'] if trades else None,
                "period_end": trades[-1]['exit_time'] if trades and 'exit_time' in trades[-1] else None
            },
            "pnl": pnl_metrics,
            "risk_adjusted": {
                "sharpe_ratio": sharpe_metrics,
                "sortino_ratio": sortino_metrics,
                "calmar_ratio": calmar_metrics
            },
            "drawdown": {
                "max_drawdown": dd_metrics,
                "drawdown_from_returns": dd_from_returns
            },
            "trading_stats": {
                "win_rate": win_rate_metrics,
                "profit_factor": pf_metrics,
                "win_rate_by_strategy": win_rate_by_strategy,
                "profit_factor_by_strategy": pf_by_strategy
            },
            "recovery": recovery_metrics
        }
        
        # Store in history
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Store daily metrics
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_metrics[today] = metrics
        
        return metrics
    
    async def compare_strategies(self, strategies: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple strategies
        """
        comparison = {}
        
        for name, data in strategies.items():
            trades = data.get("trades", [])
            equity_curve = data.get("equity_curve", [])
            
            # Calculate key metrics
            returns = self._get_returns_from_equity(equity_curve) if equity_curve else []
            
            sharpe = self.sharpe.calculate(returns) if returns else {}
            sortino = self.sortino.calculate(returns) if returns else {}
            win_rate = self.win_rate.calculate(trades)
            pf = self.profit_factor.calculate(trades)
            dd = self.max_drawdown.calculate(equity_curve) if equity_curve else {}
            
            comparison[name] = {
                "sharpe_ratio": sharpe.get('annualized_sharpe', 0),
                "sortino_ratio": sortino.get('annualized_sortino', 0),
                "win_rate": win_rate.get('win_rate', 0),
                "profit_factor": pf.get('profit_factor', 0),
                "max_drawdown": dd.get('max_drawdown_pct', 0),
                "total_trades": win_rate.get('total_trades', 0)
            }
        
        # Rank strategies
        ranked_by_sharpe = sorted(
            comparison.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        ranked_by_win_rate = sorted(
            comparison.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        return {
            "strategies": comparison,
            "rankings": {
                "by_sharpe": [s[0] for s in ranked_by_sharpe],
                "by_win_rate": [s[0] for s in ranked_by_win_rate],
                "by_profit_factor": sorted(
                    comparison.items(),
                    key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != float('inf') else 999,
                    reverse=True
                )
            },
            "best_overall": ranked_by_sharpe[0][0] if ranked_by_sharpe else None
        }
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check if metrics exceed thresholds
        """
        alerts = []
        
        # Check Sharpe ratio
        sharpe = metrics.get('risk_adjusted', {}).get('sharpe_ratio', {}).get('annualized_sharpe', 0)
        if sharpe < self.thresholds['min_sharpe']:
            alerts.append({
                "metric": "sharpe_ratio",
                "value": sharpe,
                "threshold": self.thresholds['min_sharpe'],
                "severity": "warning",
                "message": f"Sharpe ratio ({sharpe:.2f}) below threshold ({self.thresholds['min_sharpe']})"
            })
        
        # Check win rate
        win_rate = metrics.get('trading_stats', {}).get('win_rate', {}).get('win_rate', 0)
        if win_rate < self.thresholds['min_win_rate']:
            alerts.append({
                "metric": "win_rate",
                "value": win_rate,
                "threshold": self.thresholds['min_win_rate'],
                "severity": "warning",
                "message": f"Win rate ({win_rate:.1%}) below threshold ({self.thresholds['min_win_rate']:.1%})"
            })
        
        # Check profit factor
        pf = metrics.get('trading_stats', {}).get('profit_factor', {}).get('profit_factor', 0)
        if pf != float('inf') and pf < self.thresholds['min_profit_factor']:
            alerts.append({
                "metric": "profit_factor",
                "value": pf,
                "threshold": self.thresholds['min_profit_factor'],
                "severity": "warning",
                "message": f"Profit factor ({pf:.2f}) below threshold ({self.thresholds['min_profit_factor']})"
            })
        
        # Check max drawdown
        dd = metrics.get('drawdown', {}).get('max_drawdown', {}).get('max_drawdown_pct', 0)
        if dd > self.thresholds['max_drawdown']:
            alerts.append({
                "metric": "max_drawdown",
                "value": dd,
                "threshold": self.thresholds['max_drawdown'],
                "severity": "critical",
                "message": f"Max drawdown ({dd:.1f}%) exceeds threshold ({self.thresholds['max_drawdown']}%)"
            })
        
        return alerts
    
    def _get_returns_from_equity(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate returns from equity curve
        """
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)
        
        return returns
    
    def generate_report(self, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        if metrics is None and self.metrics_history:
            metrics = self.metrics_history[-1]['metrics']
        elif metrics is None:
            return {"error": "No metrics available"}
        
        # Extract key metrics for summary
        summary = {
            "report_date": datetime.now().isoformat(),
            "period": metrics['summary'],
            "performance_summary": {
                "total_pnl": metrics['pnl'].get('total_pnl', 0),
                "sharpe_ratio": metrics['risk_adjusted']['sharpe_ratio'].get('annualized_sharpe', 0),
                "max_drawdown": metrics['drawdown']['max_drawdown'].get('max_drawdown_pct', 0),
                "win_rate": metrics['trading_stats']['win_rate'].get('win_rate', 0),
                "profit_factor": metrics['trading_stats']['profit_factor'].get('profit_factor', 0)
            },
            "risk_grade": self._calculate_risk_grade(metrics),
            "alerts": self.check_thresholds(metrics)
        }
        
        return {
            "summary": summary,
            "detailed_metrics": metrics
        }
    
    def _calculate_risk_grade(self, metrics: Dict[str, Any]) -> str:
        """
        Calculate overall risk grade (A-F)
        """
        score = 0
        max_score = 100
        
        # Sharpe ratio contribution (40 points)
        sharpe = metrics['risk_adjusted']['sharpe_ratio'].get('annualized_sharpe', 0)
        if sharpe > 2:
            score += 40
        elif sharpe > 1.5:
            score += 35
        elif sharpe > 1:
            score += 30
        elif sharpe > 0.5:
            score += 20
        elif sharpe > 0:
            score += 10
        
        # Max drawdown contribution (30 points)
        dd = metrics['drawdown']['max_drawdown'].get('max_drawdown_pct', 100)
        if dd < 10:
            score += 30
        elif dd < 15:
            score += 25
        elif dd < 20:
            score += 20
        elif dd < 25:
            score += 15
        elif dd < 30:
            score += 10
        else:
            score += 5
        
        # Win rate contribution (30 points)
        wr = metrics['trading_stats']['win_rate'].get('win_rate', 0) * 100
        if wr > 60:
            score += 30
        elif wr > 55:
            score += 25
        elif wr > 50:
            score += 20
        elif wr > 45:
            score += 15
        else:
            score += 10
        
        # Convert to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def get_status(self) -> Dict[str, Any]:
        """Get metrics engine status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "metrics_history_size": len(self.metrics_history),
            "daily_metrics_days": len(self.daily_metrics),
            "thresholds": self.thresholds
        }