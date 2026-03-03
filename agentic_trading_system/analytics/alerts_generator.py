"""
Alerts Generator - Generates alerts based on performance metrics
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from utils.logger import logger as logging

class AlertsGenerator:
    """
    Alerts Generator - Generates alerts based on performance metrics
    
    Features:
    - Threshold-based alerts
    - Anomaly detection
    - Trend alerts
    - Performance degradation alerts
    - Risk alerts
    - Real-time notifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Alert thresholds
        self.thresholds = config.get("thresholds", {
            "max_daily_loss": 0.05,        # 5% max daily loss
            "max_drawdown": 0.20,           # 20% max drawdown
            "min_sharpe": 0.5,               # Minimum Sharpe ratio
            "min_win_rate": 0.4,              # Minimum win rate
            "consecutive_losses": 5,           # Max consecutive losses
            "volatility_spike": 2.0,            # Volatility spike multiplier
            "profit_factor_min": 1.0,            # Minimum profit factor
            "max_position_concentration": 0.25,   # Max 25% in one position
            "min_trading_volume": 100000          # Min daily volume
        })
        
        # Alert severity levels
        self.severity_levels = {
            "critical": 3,
            "high": 2,
            "medium": 1,
            "low": 0
        }
        
        # Alert history
        self.alert_history = deque(maxlen=config.get("max_alerts", 1000))
        
        # Performance tracking
        self.daily_pnl = deque(maxlen=30)
        self.daily_returns = deque(maxlen=252)  # One year of daily returns
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.peak_equity = None
        self.drawdown_start = None
        
        # Anomaly detection
        self.baseline_stats = {
            "mean_return": 0.0,
            "std_return": 0.0,
            "mean_volume": 0.0,
            "last_update": None
        }
        
        # Notification preferences
        self.notify_channels = config.get("notify_channels", ["log", "console"])
        
        logging.info(f"✅ AlertsGenerator initialized")
    
    def check_alerts(self, metrics: Dict[str, Any], 
                    current_equity: float = None,
                    positions: List[Dict] = None,
                    market_data: Dict = None) -> List[Dict[str, Any]]:
        """
        Check all alert conditions
        """
        alerts = []
        
        # Update tracking
        if current_equity:
            self._update_tracking(current_equity, metrics)
        
        # Check each alert type
        alerts.extend(self._check_daily_loss(metrics))
        alerts.extend(self._check_drawdown(metrics))
        alerts.extend(self._check_sharpe(metrics))
        alerts.extend(self._check_win_rate(metrics))
        alerts.extend(self._check_consecutive_losses())
        alerts.extend(self._check_volatility(metrics))
        alerts.extend(self._check_profit_factor(metrics))
        alerts.extend(self._check_anomalies(metrics))
        
        if positions:
            alerts.extend(self._check_position_concentration(positions))
            alerts.extend(self._check_sector_concentration(positions))
        
        if market_data:
            alerts.extend(self._check_market_conditions(market_data))
        
        # Check for unusual patterns
        alerts.extend(self._check_unusual_patterns(metrics))
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            self._notify(alert)
        
        return alerts
    
    def _update_tracking(self, current_equity: float, metrics: Dict[str, Any]):
        """
        Update internal tracking metrics
        """
        # Update peak equity
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.drawdown_start = None
        else:
            # Track drawdown start
            if self.drawdown_start is None:
                self.drawdown_start = datetime.now()
        
        # Track daily P&L
        if 'pnl' in metrics and 'daily' in metrics['pnl']:
            daily_pnl = metrics['pnl']['daily'].get('total', 0)
            self.daily_pnl.append(daily_pnl)
            
            # Track consecutive wins/losses
            if daily_pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            elif daily_pnl < 0:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
        
        # Track daily returns for anomaly detection
        if 'risk_adjusted' in metrics and 'sharpe_ratio' in metrics['risk_adjusted']:
            returns = metrics.get('returns', [])
            if returns:
                self.daily_returns.extend(returns[-1:])
                
                # Update baseline every 30 days
                if len(self.daily_returns) >= 30:
                    self._update_baseline()
    
    def _update_baseline(self):
        """Update baseline statistics for anomaly detection"""
        if len(self.daily_returns) >= 30:
            returns_array = np.array(list(self.daily_returns)[-30:])
            self.baseline_stats = {
                "mean_return": float(np.mean(returns_array)),
                "std_return": float(np.std(returns_array)),
                "mean_volume": float(np.mean([abs(r) for r in returns_array])),
                "last_update": datetime.now().isoformat()
            }
    
    def _check_daily_loss(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for excessive daily loss"""
        alerts = []
        
        daily_pnl = metrics.get('pnl', {}).get('daily', {}).get('total', 0)
        portfolio_value = metrics.get('summary', {}).get('total_value', 1)
        
        if portfolio_value > 0:
            daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
            
            if daily_loss_pct > self.thresholds['max_daily_loss']:
                severity = "high" if daily_loss_pct > self.thresholds['max_daily_loss'] * 1.5 else "medium"
                alerts.append({
                    "id": f"daily_loss_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "daily_loss",
                    "severity": severity,
                    "title": "Excessive Daily Loss",
                    "message": f"Daily loss of {daily_loss_pct*100:.1f}% exceeds threshold of {self.thresholds['max_daily_loss']*100:.1f}%",
                    "value": float(daily_loss_pct * 100),
                    "threshold": self.thresholds['max_daily_loss'] * 100,
                    "metric": "daily_loss_pct",
                    "recommendation": "Review positions and consider reducing risk"
                })
        
        return alerts
    
    def _check_drawdown(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for excessive drawdown"""
        alerts = []
        
        current_drawdown = metrics.get('drawdown', {}).get('max_drawdown', {}).get('max_drawdown_pct', 0) / 100
        
        if current_drawdown > self.thresholds['max_drawdown']:
            severity = "critical" if current_drawdown > self.thresholds['max_drawdown'] * 1.5 else "high"
            
            # Calculate drawdown duration
            duration = "unknown"
            if self.drawdown_start:
                days = (datetime.now() - self.drawdown_start).days
                duration = f"{days} days"
            
            alerts.append({
                "id": f"drawdown_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "type": "drawdown",
                "severity": severity,
                "title": "Excessive Drawdown",
                "message": f"Current drawdown of {current_drawdown*100:.1f}% exceeds threshold of {self.thresholds['max_drawdown']*100:.1f}%",
                "value": float(current_drawdown * 100),
                "threshold": self.thresholds['max_drawdown'] * 100,
                "duration": duration,
                "metric": "drawdown_pct",
                "recommendation": "Consider hedging or reducing position sizes"
            })
        
        return alerts
    
    def _check_sharpe(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check Sharpe ratio"""
        alerts = []
        
        sharpe = metrics.get('risk_adjusted', {}).get('sharpe_ratio', {}).get('annualized_sharpe', 0)
        
        if sharpe < self.thresholds['min_sharpe']:
            alerts.append({
                "id": f"sharpe_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "type": "sharpe_ratio",
                "severity": "medium",
                "title": "Low Sharpe Ratio",
                "message": f"Sharpe ratio of {sharpe:.2f} is below threshold of {self.thresholds['min_sharpe']}",
                "value": float(sharpe),
                "threshold": self.thresholds['min_sharpe'],
                "metric": "sharpe_ratio",
                "recommendation": "Review risk-adjusted returns and consider strategy adjustments"
            })
        
        return alerts
    
    def _check_win_rate(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check win rate"""
        alerts = []
        
        win_rate = metrics.get('trading_stats', {}).get('win_rate', {}).get('win_rate', 0)
        
        if win_rate < self.thresholds['min_win_rate']:
            alerts.append({
                "id": f"win_rate_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "type": "win_rate",
                "severity": "medium",
                "title": "Low Win Rate",
                "message": f"Win rate of {win_rate*100:.1f}% is below threshold of {self.thresholds['min_win_rate']*100:.1f}%",
                "value": float(win_rate * 100),
                "threshold": self.thresholds['min_win_rate'] * 100,
                "metric": "win_rate",
                "recommendation": "Review entry criteria and signal quality"
            })
        
        return alerts
    
    def _check_consecutive_losses(self) -> List[Dict[str, Any]]:
        """Check for too many consecutive losses"""
        alerts = []
        
        if self.consecutive_losses >= self.thresholds['consecutive_losses']:
            alerts.append({
                "id": f"consecutive_losses_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "type": "consecutive_losses",
                "severity": "high",
                "title": "Consecutive Losses",
                "message": f"{self.consecutive_losses} consecutive losses detected",
                "value": self.consecutive_losses,
                "threshold": self.thresholds['consecutive_losses'],
                "metric": "consecutive_losses",
                "recommendation": "Pause trading and review strategy"
            })
        
        return alerts
    
    def _check_volatility(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for volatility spikes"""
        alerts = []
        
        current_vol = metrics.get('risk_adjusted', {}).get('sharpe_ratio', {}).get('std_return', 0)
        
        if self.baseline_stats['std_return'] > 0:
            vol_ratio = current_vol / self.baseline_stats['std_return']
            
            if vol_ratio > self.thresholds['volatility_spike']:
                alerts.append({
                    "id": f"volatility_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "volatility_spike",
                    "severity": "high",
                    "title": "Volatility Spike",
                    "message": f"Volatility spike detected: {vol_ratio:.1f}x normal levels",
                    "value": float(vol_ratio),
                    "threshold": self.thresholds['volatility_spike'],
                    "metric": "volatility_ratio",
                    "recommendation": "Reduce position sizes and widen stops"
                })
        
        return alerts
    
    def _check_profit_factor(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check profit factor"""
        alerts = []
        
        profit_factor = metrics.get('trading_stats', {}).get('profit_factor', {}).get('profit_factor', 0)
        
        if profit_factor != float('inf') and profit_factor < self.thresholds['profit_factor_min']:
            alerts.append({
                "id": f"profit_factor_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "type": "profit_factor",
                "severity": "high",
                "title": "Low Profit Factor",
                "message": f"Profit factor of {profit_factor:.2f} is below minimum of {self.thresholds['profit_factor_min']}",
                "value": float(profit_factor),
                "threshold": self.thresholds['profit_factor_min'],
                "metric": "profit_factor",
                "recommendation": "System is not profitable - review all components"
            })
        
        return alerts
    
    def _check_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for anomalous patterns"""
        alerts = []
        
        # Check for extreme returns (3 sigma events)
        if self.baseline_stats['std_return'] > 0:
            returns = metrics.get('returns', [])
            if returns:
                latest_return = returns[-1]
                z_score = abs(latest_return - self.baseline_stats['mean_return']) / self.baseline_stats['std_return']
                
                if z_score > 3:
                    alerts.append({
                        "id": f"anomaly_{datetime.now().timestamp()}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "anomaly",
                        "severity": "medium",
                        "title": "Abnormal Return Detected",
                        "message": f"Return of {latest_return*100:.2f}% is {z_score:.1f} standard deviations from mean",
                        "value": float(latest_return * 100),
                        "z_score": float(z_score),
                        "metric": "return_anomaly",
                        "recommendation": "Verify data quality and check for errors"
                    })
        
        return alerts
    
    def _check_position_concentration(self, positions: List[Dict]) -> List[Dict[str, Any]]:
        """Check for position concentration risk"""
        alerts = []
        
        if not positions:
            return alerts
        
        total_value = sum(p.get('market_value', p.get('value', 0)) for p in positions)
        
        if total_value > 0:
            for position in positions:
                symbol = position.get('symbol', 'unknown')
                value = position.get('market_value', position.get('value', 0))
                concentration = value / total_value
                
                if concentration > self.thresholds['max_position_concentration']:
                    alerts.append({
                        "id": f"concentration_{symbol}_{datetime.now().timestamp()}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "position_concentration",
                        "severity": "high",
                        "title": f"High Concentration in {symbol}",
                        "message": f"Position in {symbol} represents {concentration*100:.1f}% of portfolio",
                        "value": float(concentration * 100),
                        "threshold": self.thresholds['max_position_concentration'] * 100,
                        "symbol": symbol,
                        "metric": "position_concentration",
                        "recommendation": "Consider reducing position or adding hedges"
                    })
        
        return alerts
    
    def _check_sector_concentration(self, positions: List[Dict]) -> List[Dict[str, Any]]:
        """Check for sector concentration risk"""
        alerts = []
        
        if not positions:
            return alerts
        
        # Group by sector
        sector_values = {}
        total_value = 0
        
        for position in positions:
            sector = position.get('sector', 'Unknown')
            value = position.get('market_value', position.get('value', 0))
            sector_values[sector] = sector_values.get(sector, 0) + value
            total_value += value
        
        if total_value > 0:
            for sector, value in sector_values.items():
                concentration = value / total_value
                
                if concentration > 0.4:  # 40% sector concentration
                    alerts.append({
                        "id": f"sector_concentration_{sector}_{datetime.now().timestamp()}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "sector_concentration",
                        "severity": "medium",
                        "title": f"High Sector Concentration in {sector}",
                        "message": f"{sector} sector represents {concentration*100:.1f}% of portfolio",
                        "value": float(concentration * 100),
                        "threshold": 40,
                        "sector": sector,
                        "metric": "sector_concentration",
                        "recommendation": "Diversify across sectors to reduce risk"
                    })
        
        return alerts
    
    def _check_market_conditions(self, market_data: Dict) -> List[Dict[str, Any]]:
        """Check for unusual market conditions"""
        alerts = []
        
        # Check VIX if available
        vix = market_data.get('vix')
        if vix:
            if vix > 30:
                alerts.append({
                    "id": f"vix_high_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "market_volatility",
                    "severity": "medium",
                    "title": "High Market Volatility",
                    "message": f"VIX at {vix:.1f} indicates elevated market fear",
                    "value": float(vix),
                    "threshold": 30,
                    "metric": "vix",
                    "recommendation": "Reduce position sizes and use wider stops"
                })
            elif vix < 12:
                alerts.append({
                    "id": f"vix_low_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "market_complacency",
                    "severity": "low",
                    "title": "Low Market Volatility",
                    "message": f"VIX at {vix:.1f} indicates market complacency",
                    "value": float(vix),
                    "threshold": 12,
                    "metric": "vix",
                    "recommendation": "Be cautious of complacency - consider mean reversion strategies"
                })
        
        # Check trading volume
        volume = market_data.get('volume')
        avg_volume = market_data.get('avg_volume')
        
        if volume and avg_volume and avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio < 0.5:
                alerts.append({
                    "id": f"low_volume_{datetime.now().timestamp()}",
                    "timestamp": datetime.now().isoformat(),
                    "type": "low_volume",
                    "severity": "low",
                    "title": "Low Trading Volume",
                    "message": f"Current volume is {volume_ratio*100:.0f}% of average",
                    "value": float(volume_ratio),
                    "threshold": 0.5,
                    "metric": "volume_ratio",
                    "recommendation": "Be careful with large orders - may experience slippage"
                })
        
        return alerts
    
    def _check_unusual_patterns(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for unusual trading patterns"""
        alerts = []
        
        # Check for sudden change in trade frequency
        trades = metrics.get('trades', [])
        if len(trades) > 20:
            recent_trades = trades[-10:]
            older_trades = trades[-20:-10]
            
            if len(older_trades) > 0:
                recent_rate = len([t for t in recent_trades if t.get('exit_time')]) / 10
                older_rate = len([t for t in older_trades if t.get('exit_time')]) / 10
                
                if recent_rate > older_rate * 3 and older_rate > 0:
                    alerts.append({
                        "id": f"trade_frequency_{datetime.now().timestamp()}",
                        "timestamp": datetime.now().isoformat(),
                        "type": "trade_frequency",
                        "severity": "low",
                        "title": "Increased Trading Frequency",
                        "message": f"Trade frequency has increased {recent_rate/older_rate:.1f}x",
                        "value": float(recent_rate / older_rate),
                        "metric": "trade_frequency_ratio",
                        "recommendation": "Verify strategy is not overtrading"
                    })
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 100, severity: str = None) -> List[Dict]:
        """Get recent alerts, optionally filtered by severity"""
        alerts = list(self.alert_history)[-limit:]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        alerts = list(self.alert_history)
        
        if not alerts:
            return {"message": "No alerts in history"}
        
        # Count by severity
        severity_counts = {
            "critical": len([a for a in alerts if a['severity'] == 'critical']),
            "high": len([a for a in alerts if a['severity'] == 'high']),
            "medium": len([a for a in alerts if a['severity'] == 'medium']),
            "low": len([a for a in alerts if a['severity'] == 'low'])
        }
        
        # Count by type
        type_counts = {}
        for alert in alerts:
            alert_type = alert['type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Most recent by severity
        latest = {}
        for severity in ['critical', 'high', 'medium', 'low']:
            sev_alerts = [a for a in alerts if a['severity'] == severity]
            if sev_alerts:
                latest[severity] = sev_alerts[-1]
        
        return {
            "total_alerts": len(alerts),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "latest_by_severity": latest,
            "time_period": {
                "oldest": alerts[0]['timestamp'] if alerts else None,
                "newest": alerts[-1]['timestamp'] if alerts else None
            }
        }
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alert_history.clear()
        logging.info("🧹 Cleared all alerts")
    
    def _notify(self, alert: Dict[str, Any]):
        """
        Send notification based on alert severity
        """
        # Log alert
        log_message = f"[{alert['severity'].upper()}] {alert['title']}: {alert['message']}"
        
        if alert['severity'] == 'critical':
            logging.critical(log_message)
        elif alert['severity'] == 'high':
            logging.error(log_message)
        elif alert['severity'] == 'medium':
            logging.warning(log_message)
        else:
            logging.info(log_message)
        
        # In production, this would send to various channels
        # For now, just log
        if 'console' in self.notify_channels:
            print(f"\n🔔 ALERT: {log_message}\n")
    
    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        """Update alert thresholds"""
        self.thresholds.update(new_thresholds)
        logging.info(f"📊 Updated alert thresholds: {new_thresholds}")