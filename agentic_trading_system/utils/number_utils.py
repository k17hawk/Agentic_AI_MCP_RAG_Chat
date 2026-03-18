"""
Number Utils - Financial and mathematical utility functions
"""
import math
from typing import List, Optional, Union,Dict
import numpy as np

def round_price(price: float, tick_size: float = 0.01) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size

def round_quantity(quantity: float, lot_size: int = 1) -> int:
    """Round quantity to nearest lot size"""
    return int(round(quantity / lot_size) * lot_size)

def calculate_percentage(part: float, whole: float, 
                        decimals: int = 2) -> float:
    """Calculate percentage"""
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimals)

def calculate_change(current: float, previous: float) -> Dict[str, float]:
    """Calculate absolute and percentage change"""
    absolute = current - previous
    if previous == 0:
        percentage = 0.0
    else:
        percentage = (absolute / previous) * 100
    
    return {
        "absolute": round(absolute, 2),
        "percentage": round(percentage, 2)
    }

def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average"""
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def exponential_moving_average(values: List[float], period: int) -> List[float]:
    """Calculate exponential moving average"""
    if len(values) < period:
        return []
    
    alpha = 2 / (period + 1)
    ema = []
    
    # SMA for first value
    ema.append(sum(values[:period]) / period)
    
    for i in range(period, len(values)):
        ema.append(alpha * values[i] + (1 - alpha) * ema[-1])
    
    return ema

def standard_deviation(values: List[float], sample: bool = True) -> float:
    """Calculate standard deviation"""
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1 if sample else n)
    return math.sqrt(variance)

def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02,
                periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = [r - risk_free_rate/periods_per_year for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    std_excess = standard_deviation(excess_returns)
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    return sharpe * math.sqrt(periods_per_year)

def sortino_ratio(returns: List[float], risk_free_rate: float = 0.02,
                  periods_per_year: int = 252, target_return: float = 0) -> float:
    """Calculate Sortino ratio"""
    if len(returns) < 2:
        return 0.0
    
    # Calculate downside deviation
    target = target_return / periods_per_year
    downside_returns = [min(0, r - target) for r in returns]
    downside_dev = standard_deviation(downside_returns)
    
    if downside_dev == 0:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - risk_free_rate/periods_per_year
    
    sortino = excess_return / downside_dev
    return sortino * math.sqrt(periods_per_year)

def max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """Calculate maximum drawdown"""
    if len(equity_curve) < 2:
        return {"drawdown": 0.0, "peak_index": 0, "trough_index": 0}
    
    peak = equity_curve[0]
    peak_index = 0
    max_dd = 0
    max_dd_peak = 0
    max_dd_trough = 0
    max_dd_peak_index = 0
    max_dd_trough_index = 0
    
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_index = i
        
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_peak = peak
            max_dd_trough = value
            max_dd_peak_index = peak_index
            max_dd_trough_index = i
    
    return {
        "drawdown": max_dd * 100,
        "peak_value": max_dd_peak,
        "trough_value": max_dd_trough,
        "peak_index": max_dd_peak_index,
        "trough_index": max_dd_trough_index
    }

def calculate_beta(stock_returns: List[float], 
                   market_returns: List[float]) -> float:
    """Calculate beta (systematic risk)"""
    if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
        return 1.0
    
    # Calculate covariance and variance
    stock_mean = sum(stock_returns) / len(stock_returns)
    market_mean = sum(market_returns) / len(market_returns)
    
    covariance = sum((s - stock_mean) * (m - market_mean) 
                    for s, m in zip(stock_returns, market_returns))
    variance = sum((m - market_mean) ** 2 for m in market_returns)
    
    if variance == 0:
        return 1.0
    
    return covariance / variance

def calculate_alpha(stock_returns: List[float], market_returns: List[float],
                   risk_free_rate: float = 0.02) -> float:
    """Calculate alpha (excess return)"""
    beta = calculate_beta(stock_returns, market_returns)
    
    stock_mean = sum(stock_returns) / len(stock_returns)
    market_mean = sum(market_returns) / len(market_returns)
    
    expected_return = risk_free_rate + beta * (market_mean - risk_free_rate)
    alpha = stock_mean - expected_return
    
    return alpha * 252  # Annualized

def calculate_var(returns: List[float], confidence: float = 0.95,
                 portfolio_value: float = 1.0) -> float:
    """Calculate Value at Risk"""
    if len(returns) < 2:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var_return = sorted_returns[index]
    
    return portfolio_value * abs(var_return)

def calculate_cvar(returns: List[float], confidence: float = 0.95,
                  portfolio_value: float = 1.0) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if len(returns) < 2:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    tail_returns = sorted_returns[:index]
    
    if not tail_returns:
        tail_returns = [sorted_returns[0]]
    
    cvar_return = sum(tail_returns) / len(tail_returns)
    return portfolio_value * abs(cvar_return)

def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) * 
                           sum((yi - mean_y) ** 2 for yi in y))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0
    
    gains = []
    losses = []
    
    for i in range(1, period + 1):
        change = prices[-i] - prices[-i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi