"""
Sliding Window Analyzer - 60-day rolling window analysis
This implements your core requirement for 2-month analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

class SlidingWindowAnalyzer:
    """
    60-day sliding window analysis for price movements
    Calculates rolling statistics, detects anomalies, and identifies trends
    
    This is the CORE of your 60-day requirement!
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.window_days = config.get("lookback_days", 60)  # YOUR 60-DAY REQUIREMENT
        self.min_window = config.get("min_window", 30)  # Minimum data points needed
        
        # Window configurations for different analyses
        self.windows = {
            "short": 5,     # 5-day momentum
            "medium": 20,   # 20-day trend
            "long": 60,     # 60-day baseline (YOUR REQUIREMENT)
            "yearly": 252   # 1-year context (trading days)
        }
        
        logger.info(f"📊 SlidingWindowAnalyzer initialized with {self.window_days}-day window")
    
    def analyze(self, data: pd.DataFrame, window_days: int = None) -> Dict:
        """
        Perform comprehensive sliding window analysis
        
        Args:
            data: DataFrame with OHLCV data
            window_days: Analysis window (defaults to 60)
            
        Returns:
            Dictionary with all sliding window metrics
        """
        if window_days is None:
            window_days = self.window_days
        
        if len(data) < self.min_window:
            return {
                "error": f"Insufficient data: {len(data)} < {self.min_window}",
                "sufficient_data": False
            }
        
        results = {
            "sufficient_data": True,
            "window_days": window_days,
            "current_price": float(data['Close'].iloc[-1]),
            "windows": {},
            "rolling_stats": self._rolling_statistics(data, window_days),
            "anomalies": self._detect_anomalies(data, window_days),
            "trend": self._calculate_trend(data, window_days),
            "momentum": self._calculate_momentum(data),
            "volatility_regime": self._volatility_regime(data, window_days),
            "position": self._calculate_position(data)
        }
        
        # Analyze each window
        for window_name, window_size in self.windows.items():
            if len(data) >= window_size:
                window_result = self._analyze_window(data, window_size)
                results["windows"][window_name] = window_result
        
        return results
    
    def _analyze_window(self, data: pd.DataFrame, window: int) -> Dict:
        """
        Analyze a specific time window
        
        Args:
            data: Full DataFrame
            window: Window size in days
            
        Returns:
            Statistics for this window
        """
        window_data = data.iloc[-window:]
        
        # Basic statistics
        mean_price = window_data['Close'].mean()
        std_price = window_data['Close'].std()
        min_price = window_data['Close'].min()
        max_price = window_data['Close'].max()
        
        # Returns within window
        returns = window_data['Close'].pct_change().dropna() * 100
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Current position within window
        current_price = data['Close'].iloc[-1]
        if max_price > min_price:
            position_pct = (current_price - min_price) / (max_price - min_price) * 100
        else:
            position_pct = 50.0
        
        # Z-score within window
        if std_price > 0:
            z_score = (current_price - mean_price) / std_price
        else:
            z_score = 0.0
        
        # Volume analysis for this window
        avg_volume = window_data['Volume'].mean()
        current_volume = window_data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            "window_days": window,
            "mean_price": float(mean_price),
            "std_price": float(std_price),
            "min_price": float(min_price),
            "max_price": float(max_price),
            "price_range": float(max_price - min_price),
            "range_pct": float((max_price - min_price) / mean_price * 100) if mean_price > 0 else 0,
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "position_pct": float(position_pct),
            "z_score": float(z_score),
            "is_high": position_pct > 80,
            "is_low": position_pct < 20,
            "volume_ratio": float(volume_ratio),
            "data_points": len(window_data)
        }
    
    def _rolling_statistics(self, data: pd.DataFrame, window: int) -> Dict:
        """
        Calculate rolling statistics over the window
        
        Args:
            data: Full DataFrame
            window: Window size
            
        Returns:
            Rolling statistics
        """
        if len(data) < window:
            return {}
        
        # Get rolling window
        rolling = data.iloc[-window:]
        
        # Calculate statistics
        close_prices = rolling['Close']
        
        stats = {
            "mean": float(close_prices.mean()),
            "median": float(close_prices.median()),
            "std": float(close_prices.std()),
            "min": float(close_prices.min()),
            "max": float(close_prices.max()),
            "percentile_25": float(close_prices.quantile(0.25)),
            "percentile_75": float(close_prices.quantile(0.75)),
            "iqr": float(close_prices.quantile(0.75) - close_prices.quantile(0.25)),
            "range": float(close_prices.max() - close_prices.min()),
            "range_pct": float((close_prices.max() - close_prices.min()) / close_prices.mean() * 100) if close_prices.mean() > 0 else 0
        }
        
        # Skewness and kurtosis (measure of distribution shape)
        if len(rolling) > 3:
            stats["skewness"] = float(close_prices.skew())
            stats["kurtosis"] = float(close_prices.kurtosis())
        
        return stats
    
    def _detect_anomalies(self, data: pd.DataFrame, window: int) -> List[Dict]:
        """
        Detect anomalous price movements using the sliding window
        
        Args:
            data: Full DataFrame
            window: Window size for baseline
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(data) < window:
            return anomalies
        
        # Use window for baseline
        baseline = data.iloc[-window:]
        returns = baseline['Close'].pct_change().dropna() * 100
        
        # Calculate statistics
        if len(returns) < 10:
            return anomalies
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return anomalies
        
        # Check last 10 days for anomalies
        recent = data.iloc[-10:]
        for i in range(1, len(recent)):
            daily_return = (recent['Close'].iloc[i] / recent['Close'].iloc[i-1] - 1) * 100
            
            # Z-score of this return
            z_score = (daily_return - mean_return) / std_return
            
            # Check if anomalous (|z| > 2)
            if abs(z_score) > 2:
                anomalies.append({
                    "date": recent.index[i].strftime("%Y-%m-%d") if hasattr(recent.index[i], 'strftime') else str(i),
                    "return": float(daily_return),
                    "z_score": float(z_score),
                    "type": "positive" if daily_return > 0 else "negative",
                    "severity": "high" if abs(z_score) > 3 else "medium",
                    "position": i
                })
        
        return anomalies
    
    def _calculate_trend(self, data: pd.DataFrame, window: int) -> Dict:
        """
        Calculate trend strength and direction
        
        Args:
            data: Full DataFrame
            window: Window size
            
        Returns:
            Trend analysis
        """
        if len(data) < window:
            return {"strength": 0, "direction": "unknown", "confidence": 0}
        
        recent = data.iloc[-window:]
        x = np.arange(len(recent))
        y = recent['Close'].values
        
        # Linear regression
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            # R-squared (trend consistency)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Trend direction
            if slope > 0.01:
                direction = "up"
            elif slope < -0.01:
                direction = "down"
            else:
                direction = "sideways"
            
            # Trend strength (0-100)
            strength = r_squared * 100
            
            # Price change over window
            price_change_pct = (y[-1] - y[0]) / y[0] * 100 if y[0] != 0 else 0
            
            # Confidence in trend
            if strength > 70 and abs(price_change_pct) > 5:
                confidence = "high"
            elif strength > 40 and abs(price_change_pct) > 2:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "direction": direction,
                "strength": float(strength),
                "slope": float(slope),
                "r_squared": float(r_squared),
                "price_change_pct": float(price_change_pct),
                "confidence": confidence,
                "is_strong_trend": strength > 60,
                "is_weak_trend": strength < 30
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {"strength": 0, "direction": "unknown", "confidence": "low"}
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Dict:
        """
        Calculate momentum signals across different periods
        
        Args:
            data: Full DataFrame
            
        Returns:
            Momentum analysis
        """
        if len(data) < 20:
            return {}
        
        close = data['Close']
        current = close.iloc[-1]
        
        # Rate of change over different periods
        roc = {}
        periods = [1, 5, 10, 20, 60]
        
        for period in periods:
            if len(data) >= period + 1:
                past = close.iloc[-period-1]
                roc[f"roc_{period}"] = float((current - past) / past * 100) if past != 0 else 0
            else:
                roc[f"roc_{period}"] = None
        
        # Momentum score (0-100)
        scores = []
        for period, value in roc.items():
            if value is not None and value > 0:
                scores.append(1)
            elif value is not None:
                scores.append(0)
        
        momentum_score = (sum(scores) / len(scores) * 100) if scores else 50
        
        # Check if momentum is accelerating
        if roc.get("roc_5") and roc.get("roc_20"):
            accelerating = roc["roc_5"] > roc["roc_20"]
        else:
            accelerating = None
        
        return {
            "rate_of_change": roc,
            "momentum_score": float(momentum_score),
            "accelerating": accelerating,
            "bullish_momentum": momentum_score > 60,
            "bearish_momentum": momentum_score < 40
        }
    
    def _volatility_regime(self, data: pd.DataFrame, window: int) -> Dict:
        """
        Determine volatility regime using sliding window
        
        Args:
            data: Full DataFrame
            window: Window size
            
        Returns:
            Volatility regime analysis
        """
        if len(data) < window:
            return {"regime": "unknown", "level": 0}
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna() * 100
        
        if len(returns) < 20:
            return {"regime": "unknown", "level": 0}
        
        # Current volatility (last 20 days)
        current_vol = returns.iloc[-20:].std()
        
        # Historical volatility (full window)
        historical_vol = returns.iloc[-window:].std()
        
        # Volatility percentile (where current ranks historically)
        if historical_vol > 0:
            vol_percentile = (current_vol / historical_vol) * 100
        else:
            vol_percentile = 50
        
        # Determine regime
        if vol_percentile > 80:
            regime = "very_high"
            level = 4
            description = "Extremely high volatility - reduce position sizes"
        elif vol_percentile > 60:
            regime = "high"
            level = 3
            description = "High volatility - be cautious"
        elif vol_percentile > 40:
            regime = "normal"
            level = 2
            description = "Normal volatility - standard position sizing"
        elif vol_percentile > 20:
            regime = "low"
            level = 1
            description = "Low volatility - can increase positions"
        else:
            regime = "very_low"
            level = 0
            description = "Very low volatility - potential for expansion"
        
        return {
            "regime": regime,
            "level": level,
            "description": description,
            "current_volatility": float(current_vol),
            "historical_volatility": float(historical_vol),
            "vol_percentile": float(vol_percentile),
            "volatility_ratio": float(current_vol / historical_vol) if historical_vol > 0 else 1.0
        }
    
    def _calculate_position(self, data: pd.DataFrame) -> Dict:
        """
        Calculate position relative to key moving averages
        
        Args:
            data: Full DataFrame
            
        Returns:
            Position analysis
        """
        if len(data) < 20:
            return {}
        
        current = data['Close'].iloc[-1]
        
        # Moving averages
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
        ma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else None
        
        # Distance from averages
        position = {
            "current_price": float(current),
            "above_ma20": current > ma20 if ma20 else None,
            "above_ma50": current > ma50 if ma50 else None,
            "above_ma200": current > ma200 if ma200 else None
        }
        
        if ma20:
            position["distance_from_ma20"] = float((current - ma20) / ma20 * 100)
        if ma50:
            position["distance_from_ma50"] = float((current - ma50) / ma50 * 100)
        if ma200:
            position["distance_from_ma200"] = float((current - ma200) / ma200 * 100)
        
        # Golden/Death cross detection
        if ma50 and ma200:
            position["golden_cross"] = ma50 > ma200 and current > ma50
            position["death_cross"] = ma50 < ma200 and current < ma50
        
        return position
    
    def get_moving_averages(self, data: pd.DataFrame) -> Dict:
        """
        Get all moving averages
        
        Args:
            data: Full DataFrame
            
        Returns:
            Dictionary with all MAs
        """
        mas = {}
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(data) >= period:
                mas[f"ma_{period}"] = float(data['Close'].rolling(period).mean().iloc[-1])
            else:
                mas[f"ma_{period}"] = None
        
        return mas
    
    def get_support_resistance(self, data: pd.DataFrame, window: int = 60) -> Dict:
        """
        Find support and resistance levels using sliding window
        
        Args:
            data: Full DataFrame
            window: Window size
            
        Returns:
            Support and resistance levels
        """
        if len(data) < window:
            return {}
        
        window_data = data.iloc[-window:]
        current = data['Close'].iloc[-1]
        
        # Find local maxima and minima
        highs = window_data['High'].values
        lows = window_data['Low'].values
        
        # Simple method: use percentiles
        resistance_levels = [
            float(np.percentile(highs, 90)),
            float(np.percentile(highs, 75)),
            float(np.percentile(highs, 50))
        ]
        
        support_levels = [
            float(np.percentile(lows, 10)),
            float(np.percentile(lows, 25)),
            float(np.percentile(lows, 50))
        ]
        
        # Find nearest resistance above current price
        nearest_resistance = min([r for r in resistance_levels if r > current], default=None)
        if nearest_resistance:
            resistance_distance = (nearest_resistance - current) / current * 100
        else:
            resistance_distance = None
        
        # Find nearest support below current price
        nearest_support = max([s for s in support_levels if s < current], default=None)
        if nearest_support:
            support_distance = (current - nearest_support) / current * 100
        else:
            support_distance = None
        
        return {
            "current_price": float(current),
            "resistance_levels": resistance_levels,
            "support_levels": support_levels,
            "nearest_resistance": float(nearest_resistance) if nearest_resistance else None,
            "resistance_distance": float(resistance_distance) if resistance_distance else None,
            "nearest_support": float(nearest_support) if nearest_support else None,
            "support_distance": float(support_distance) if support_distance else None
        }
    
    def get_window_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get a human-readable summary of the 60-day window analysis
        
        Args:
            data: Full DataFrame
            
        Returns:
            Summary dictionary
        """
        analysis = self.analyze(data)
        
        summary = {
            "window_days": self.window_days,
            "current_price": analysis.get("current_price"),
            "trend": analysis.get("trend", {}).get("direction", "unknown"),
            "trend_strength": analysis.get("trend", {}).get("confidence", "low"),
            "volatility": analysis.get("volatility_regime", {}).get("regime", "unknown"),
            "momentum": "bullish" if analysis.get("momentum", {}).get("bullish_momentum") else "bearish" if analysis.get("momentum", {}).get("bearish_momentum") else "neutral",
            "anomalies_detected": len(analysis.get("anomalies", [])),
            "position": "high" if analysis.get("position", {}).get("above_ma20") and analysis.get("position", {}).get("above_ma50") else "low" if not analysis.get("position", {}).get("above_ma20") else "medium"
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Test the sliding window analyzer
    print("Testing SlidingWindowAnalyzer...")
    
    # Get sample data
    ticker = "AAPL"
    data = yf.Ticker(ticker).history(period="6mo")
    
    # Create analyzer
    analyzer = SlidingWindowAnalyzer({"lookback_days": 60})
    
    # Run analysis
    results = analyzer.analyze(data)
    
    # Print summary
    print(f"\n📊 60-Day Window Analysis for {ticker}")
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Trend: {results['trend']['direction']} (strength: {results['trend']['strength']:.1f}%)")
    print(f"Volatility Regime: {results['volatility_regime']['regime']}")
    print(f"Anomalies Detected: {len(results['anomalies'])}")
    
    # Print window details
    print("\n📈 Window Analysis:")
    for window_name, window_data in results['windows'].items():
        print(f"  {window_name}: Z-score={window_data['z_score']:.2f}, "
              f"Position={window_data['position_pct']:.1f}%, "
              f"Volume Ratio={window_data['volume_ratio']:.2f}x")
    
    # Get summary
    summary = analyzer.get_window_summary(data)
    print(f"\n📋 Summary: {summary}")