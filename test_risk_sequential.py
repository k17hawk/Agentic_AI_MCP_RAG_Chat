#!/usr/bin/env python3
"""
Comprehensive Sequential Test: Risk Module
Tests position sizing, stop losses, VaR, portfolio risk, and RiskManager integration
Uses all imported risk components for complete testing
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Import all risk components
from agentic_trading_system.risk.position_sizing.kelly_criterion import KellyCriterion
from agentic_trading_system.risk.position_sizing.half_kelly import HalfKelly
from agentic_trading_system.risk.position_sizing.fixed_fraction import FixedFraction
from agentic_trading_system.risk.position_sizing.volatility_adjusted import VolatilityAdjusted
from agentic_trading_system.risk.stop_loss_optimizer.atr_stop import ATRStop
from agentic_trading_system.risk.stop_loss_optimizer.volatility_stop import VolatilityStop
from agentic_trading_system.risk.stop_loss_optimizer.trailing_stop import TrailingStop
from agentic_trading_system.risk.stop_loss_optimizer.time_stop import TimeStop
from agentic_trading_system.risk.portfolio_risk.var_calculator import VaRCalculator
from agentic_trading_system.risk.portfolio_risk.expected_shortfall import ExpectedShortfall
from agentic_trading_system.risk.portfolio_risk.diversification_score import DiversificationScore
from agentic_trading_system.risk.portfolio_risk.stress_tester import StressTester
from agentic_trading_system.risk.risk_scorer import RiskScorer
from agentic_trading_system.risk.market_regime_risk import MarketRegimeRisk
from agentic_trading_system.risk.risk_manager import RiskManager
from agentic_trading_system.agents.base_agent import AgentMessage
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class ComprehensiveRiskTest:
    """
    Comprehensive sequential test for Risk module
    Tests individual components AND full RiskManager integration
    """
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
        self.analysis_results = self.load_analysis_results()
        self.results = []
        self.errors = []
        self.initial_capital = 100000
        self.component_tests = []
        self.integration_tests = []
        
        # Initialize RiskManager with complete configuration
        self.risk_manager = RiskManager(
            name="TestRiskManager",
            config={
                "initial_capital": self.initial_capital,
                "kelly_config": {
                    "default_win_rate": 0.55,
                    "default_avg_win": 0.10,
                    "default_avg_loss": 0.05,
                    "max_fraction": 0.25,
                    "min_fraction": 0.01
                },
                "half_kelly_config": {
                    "kelly_multiplier": 0.5,
                    "max_fraction": 0.15,
                    "min_fraction": 0.005
                },
                "fixed_fraction_config": {
                    "default_fraction": 0.02,
                    "max_fraction": 0.10,
                    "min_fraction": 0.005,
                    "scale_with_confidence": True,
                    "confidence_scale_factor": 0.5
                },
                "vol_adjusted_config": {
                    "base_fraction": 0.02,
                    "target_volatility": 0.20,
                    "max_scale": 2.0,
                    "min_scale": 0.25
                },
                "atr_stop_config": {
                    "default_multiplier": 2.0,
                    "min_multiplier": 1.0,
                    "max_multiplier": 4.0,
                    "atr_period": 14
                },
                "vol_stop_config": {
                    "default_multiplier": 2.0,
                    "lookback_period": 20,
                    "min_multiplier": 1.0,
                    "max_multiplier": 4.0
                },
                "trailing_stop_config": {
                    "default_trail_percent": 5.0,
                    "atr_multiplier": 2.0,
                    "activation_percent": 10.0
                },
                "time_stop_config": {
                    "default_max_hold_hours": 24,
                    "short_term_hours": 4,
                    "swing_hours": 72,
                    "position_hours": 168
                },
                "var_config": {
                    "confidence_level": 0.95,
                    "time_horizon": 1,
                    "historical_periods": 252,
                    "method": "historical"
                },
                "es_config": {
                    "confidence_level": 0.95,
                    "historical_periods": 252
                },
                "diversification_config": {
                    "ideal_num_assets": 20,
                    "max_sector_exposure": 0.30,
                    "weights": {
                        "num_assets": 0.3,
                        "correlation": 0.3,
                        "sector_concentration": 0.2,
                        "position_concentration": 0.2
                    }
                },
                "stress_config": {
                    "scenarios": {
                        "market_crash_2008": {"equity": -0.37, "bonds": 0.05},
                        "flash_crash_2010": {"equity": -0.09, "bonds": 0.02},
                        "covid_crash_2020": {"equity": -0.34, "bonds": 0.10},
                        "interest_rate_shock": {"equity": -0.15, "bonds": -0.05},
                        "volatility_spike": {"equity": -0.10, "bonds": 0.03}
                    }
                },
                "regime_risk_config": {
                    "risk_multipliers": {
                        "strong_bull_trending": 1.2,
                        "bull_trending": 1.1,
                        "bull_ranging": 1.0,
                        "neutral_ranging": 1.0,
                        "bear_ranging": 0.9,
                        "bear_trending": 0.8,
                        "strong_bear_trending": 0.7,
                        "high_volatility": 0.6,
                        "extreme_volatility_ranging": 0.5,
                        "transition": 0.7,
                        "panic": 0.4
                    },
                    "stop_multipliers": {
                        "strong_bull_trending": 2.5,
                        "bull_trending": 2.0,
                        "bull_ranging": 1.8,
                        "neutral_ranging": 1.5,
                        "bear_ranging": 1.3,
                        "bear_trending": 1.2,
                        "strong_bear_trending": 1.0,
                        "high_volatility": 1.5,
                        "extreme_volatility_ranging": 1.8,
                        "transition": 1.2,
                        "panic": 1.0
                    },
                    "max_position_sizes": {
                        "strong_bull_trending": 0.25,
                        "bull_trending": 0.20,
                        "bull_ranging": 0.15,
                        "neutral_ranging": 0.12,
                        "bear_ranging": 0.10,
                        "bear_trending": 0.08,
                        "strong_bear_trending": 0.05,
                        "high_volatility": 0.06,
                        "extreme_volatility_ranging": 0.04,
                        "transition": 0.07,
                        "panic": 0.03
                    }
                },
                "scorer_config": {
                    "trade_weights": {
                        "position_size": 0.20,
                        "stop_distance": 0.15,
                        "volatility": 0.15,
                        "market_regime": 0.15,
                        "liquidity": 0.10,
                        "correlation": 0.10,
                        "leverage": 0.10,
                        "time_horizon": 0.05
                    },
                    "low_risk_threshold": 0.3,
                    "medium_risk_threshold": 0.6,
                    "high_risk_threshold": 0.8
                },
                "queue_config": {
                    "max_size": 500,
                    "item_ttl_seconds": 300
                }
            }
        )
        
        # Set initial market regime
        self.current_regime = "bear_trending_high_vol"
        
        logging.info("✅ RiskManager initialized with complete configuration")
    
    def load_analysis_results(self):
        """Load previous analysis results"""
        try:
            with open("data/analysis_results.json", "r") as f:
                data = json.load(f)
                return data.get("results", [])
        except FileNotFoundError:
            print("⚠️ No analysis results found. Using mock data.")
            return [
                {"symbol": "AAPL", "weighted": {"final_score": 0.62, "action": "WATCH", "confidence": 0.75}},
                {"symbol": "MSFT", "weighted": {"final_score": 0.66, "action": "WATCH", "confidence": 0.74}},
                {"symbol": "GOOGL", "weighted": {"final_score": 0.65, "action": "WATCH", "confidence": 0.74}},
                {"symbol": "TSLA", "weighted": {"final_score": 0.53, "action": "SELL", "confidence": 0.78}},
                {"symbol": "NVDA", "weighted": {"final_score": 0.62, "action": "WATCH", "confidence": 0.75}},
                {"symbol": "META", "weighted": {"final_score": 0.63, "action": "WATCH", "confidence": 0.75}}
            ]
    
    # ==================== COMPONENT TESTS ====================
    
    async def test_position_sizing_components(self, symbol: str, price: float, 
                                             volatility: float, confidence: float) -> Dict:
        """
        Test individual position sizing components
        """
        print(f"\n   📏 Testing Position Sizing Components for {symbol}:")
        print("   " + "-" * 40)
        
        capital = self.initial_capital
        
        # Kelly Criterion
        kelly = KellyCriterion({})
        kelly_result = kelly.calculate(
            win_rate=0.55,
            avg_win=0.10,
            avg_loss=0.05
        )
        kelly_position = capital * kelly_result['recommended_fraction']
        
        # Half Kelly
        half_kelly = HalfKelly({})
        half_result = half_kelly.calculate(
            win_rate=0.55,
            avg_win=0.10,
            avg_loss=0.05
        )
        half_position = capital * half_result['recommended_fraction']
        
        # Fixed Fraction
        fixed = FixedFraction({})
        fixed_result = fixed.calculate(capital=capital, confidence=confidence)
        fixed_position = fixed_result['position_value']
        
        # Volatility Adjusted
        vol_adjusted = VolatilityAdjusted({})
        vol_result = vol_adjusted.calculate(
            capital=capital,
            volatility=volatility,
            confidence=confidence
        )
        vol_position = vol_result['position_value']
        
        print(f"      • Kelly Criterion: ${kelly_position:,.0f} ({kelly_result['recommended_fraction']*100:.1f}%)")
        print(f"      • Half Kelly: ${half_position:,.0f} ({half_result['recommended_fraction']*100:.1f}%)")
        print(f"      • Fixed Fraction: ${fixed_position:,.0f} ({fixed_result['fraction']*100:.1f}%)")
        print(f"      • Volatility Adjusted: ${vol_position:,.0f} ({vol_result['fraction']*100:.1f}%)")
        
        # Conservative approach (min of all)
        conservative_position = min(kelly_position, half_position, fixed_position, vol_position)
        aggressive_position = max(kelly_position, half_position, fixed_position, vol_position)
        
        result = {
            "kelly": kelly_position,
            "half_kelly": half_position,
            "fixed_fraction": fixed_position,
            "volatility_adjusted": vol_position,
            "conservative": conservative_position,
            "aggressive": aggressive_position,
            "shares": int(conservative_position / price) if price > 0 else 0
        }
        
        self.component_tests.append({
            "test": "position_sizing",
            "symbol": symbol,
            "result": result
        })
        
        return result
    
    async def test_stop_loss_components(self, symbol: str, entry_price: float, 
                                       volatility: float, atr: float) -> Dict:
        """
        Test individual stop loss components
        """
        print(f"\n   🛑 Testing Stop Loss Components for {symbol}:")
        print("   " + "-" * 40)
        
        # ATR Stop
        atr_stop = ATRStop({})
        atr_result = atr_stop.calculate(entry_price, atr, multiplier=2.0)
        
        # Volatility Stop
        vol_stop = VolatilityStop({})
        returns = [random.uniform(-0.02, 0.02) for _ in range(20)]
        vol_result = vol_stop.calculate(entry_price, returns, multiplier=2.0)
        
        # Trailing Stop
        trailing = TrailingStop({})
        trailing_result = trailing.calculate_percentage(entry_price, entry_price, trail_pct=5.0)
        
        # Time Stop
        time_stop = TimeStop({})
        time_result = time_stop.check_expiry(datetime.now(), max_hold_hours=24)
        
        print(f"      • ATR Stop: ${atr_result['stop_price']:.2f} (${atr_result['stop_distance']:.2f} risk)")
        print(f"      • Volatility Stop: ${vol_result['stop_price']:.2f} (${vol_result['stop_distance']:.2f} risk)")
        print(f"      • Trailing Stop: ${trailing_result['stop_price']:.2f} (trail: {trailing_result['trail_percent']:.1f}%)")
        print(f"      • Time Stop: {time_result['hold_hours']:.1f}h elapsed, expires in {time_result['time_remaining_hours']:.1f}h")
        
        result = {
            "atr_stop": atr_result['stop_price'],
            "volatility_stop": vol_result['stop_price'],
            "trailing_stop": trailing_result['stop_price'],
            "risk_per_share": atr_result['stop_distance'],
            "stop_percent": atr_result['stop_percent']
        }
        
        self.component_tests.append({
            "test": "stop_loss",
            "symbol": symbol,
            "result": result
        })
        
        return result
    
    async def test_portfolio_risk_components(self, positions: List[Dict]) -> Dict:
        """
        Test individual portfolio risk components
        """
        print(f"\n   📊 Testing Portfolio Risk Components:")
        print("   " + "-" * 40)
        
        # Create mock returns
        returns = [random.uniform(-0.02, 0.02) for _ in range(252)]
        portfolio_value = sum(p.get('value', 0) for p in positions)
        
        # VaR Calculator
        var_calc = VaRCalculator({})
        var_result = var_calc.historical_var(returns, portfolio_value)
        
        # Expected Shortfall
        es_calc = ExpectedShortfall({})
        es_result = es_calc.calculate(returns, portfolio_value)
        
        # Diversification Score
        div_calc = DiversificationScore({})
        div_result = div_calc.calculate(positions)
        
        # Stress Tester
        stress = StressTester({})
        stress_result = stress.test_portfolio(positions)
        
        print(f"      • Portfolio Value: ${portfolio_value:,.2f}")
        print(f"      • VaR (95%): ${var_result['var_amount']:,.2f} ({var_result['var_percent']:.2f}%)")
        print(f"      • CVaR (95%): ${es_result['es_amount']:,.2f} ({es_result['es_percent']:.2f}%)")
        print(f"      • Diversification Score: {div_result['score']:.1f}/100")
        print(f"      • Worst Case Scenario: {stress_result['worst_case']['scenario']} - {stress_result['worst_case']['loss']:,.2f}")
        
        result = {
            "var": var_result,
            "cvar": es_result,
            "diversification": div_result,
            "stress_test": stress_result
        }
        
        self.component_tests.append({
            "test": "portfolio_risk",
            "result": result
        })
        
        return result
    
    async def test_risk_scorer_component(self, trade_params: Dict) -> Dict:
        """
        Test risk scorer component
        """
        print(f"\n   🎯 Testing Risk Scorer Component:")
        print("   " + "-" * 40)
        
        scorer = RiskScorer({})
        score_result = scorer.score_trade(trade_params)
        
        print(f"      • Risk Score: {score_result['risk_score']:.2f}")
        print(f"      • Risk Level: {score_result['risk_level']}")
        print(f"      • Should Trade: {score_result['should_trade']}")
        
        for component, value in score_result['components'].items():
            if isinstance(value, (int, float)):
                print(f"      • {component.replace('_', ' ').title()}: {value:.2f}")
        
        self.component_tests.append({
            "test": "risk_scorer",
            "result": score_result
        })
        
        return score_result
    
    async def test_market_regime_component(self, regime: str, trade_params: Dict) -> Dict:
        """
        Test market regime risk component
        """
        print(f"\n   🌍 Testing Market Regime Component:")
        print("   " + "-" * 40)
        
        regime_risk = MarketRegimeRisk({})
        
        # Adjust based on regime
        adjusted = regime_risk.adjust_risk(trade_params, regime)
        
        print(f"      • Regime: {regime}")
        print(f"      • Risk Appetite: {regime_risk.get_risk_appetite(regime)}")
        print(f"      • Should Trade: {regime_risk.should_trade(regime)}")
        print(f"      • Position Multiplier: {adjusted['risk_multiplier']:.2f}")
        print(f"      • Stop Multiplier: {adjusted['stop_multiplier']:.2f}")
        print(f"      • Adjusted Position: ${adjusted['position_size']:,.0f}")
        print(f"      • Adjusted Stop: {adjusted['stop_percent']:.1f}%")
        
        self.component_tests.append({
            "test": "market_regime",
            "regime": regime,
            "result": adjusted
        })
        
        return adjusted
    
    # ==================== INTEGRATION TESTS ====================
    
    async def test_risk_manager_calculate(self, symbol: str, analysis: Dict, 
                                          current_price: float, volatility: float) -> Dict:
        """
        Test RiskManager.calculate_risk() method
        """
        print(f"\n   🏦 Testing RiskManager.calculate_risk() for {symbol}:")
        print("   " + "-" * 40)
        
        analysis_message = {
            "symbol": symbol,
            "action": analysis['weighted']['action'],
            "final_score": analysis['weighted']['final_score'],
            "confidence": analysis['weighted']['confidence'],
            "price": current_price,
            "volatility": volatility,
            "regime": self.current_regime
        }
        
        result = await self.risk_manager.calculate_risk(
            analysis_message,
            requester="Test"
        )
        
        print(f"      • Message Type: {result.message_type}")
        print(f"      • Symbol: {result.content.get('ticker', 'N/A')}")
        print(f"      • Position Size: ${result.content.get('position_size', 0):,.2f}")
        print(f"      • Position Fraction: {result.content.get('position_fraction', 0)*100:.2f}%")
        print(f"      • Stop Price: ${result.content.get('stop_price', 0):.2f}")
        print(f"      • Stop Percent: {result.content.get('stop_percent', 0):.2f}%")
        print(f"      • Risk Score: {result.content.get('risk_score', 0):.2f}")
        print(f"      • Risk Level: {result.content.get('risk_level', 'UNKNOWN')}")
        
        integration_result = result.content
        
        self.integration_tests.append({
            "test": "risk_manager_calculate",
            "symbol": symbol,
            "result": integration_result
        })
        
        return integration_result
    
    async def test_risk_manager_portfolio_risk(self):
        """
        Test RiskManager.calculate_portfolio_risk() method
        """
        print(f"\n   🏦 Testing RiskManager.calculate_portfolio_risk():")
        print("   " + "-" * 40)
        
        # Build positions from results that passed risk check
        for r in self.results:
            if r and r.get('should_trade', False):
                await self.risk_manager._update_portfolio({
                    "action": "buy",
                    "symbol": r['symbol'],
                    "shares": r.get('recommended_shares', 0),
                    "price": r.get('price', 0),
                    "value": r.get('recommended_position', 0),
                    "stop_loss": r.get('recommended_stop', 0),
                    "take_profit": r.get('recommended_stop', 0) * 1.5,
                    "sector": "Technology"
                })
        
        risk_metrics = await self.risk_manager.calculate_portfolio_risk()
        
        print(f"      • Total Value: ${risk_metrics.get('total_value', 0):,.2f}")
        print(f"      • Cash: ${risk_metrics.get('cash', 0):,.2f}")
        print(f"      • Invested: ${risk_metrics.get('invested', 0):,.2f}")
        print(f"      • Num Positions: {risk_metrics.get('num_positions', 0)}")
        
        if 'var' in risk_metrics:
            var = risk_metrics['var']
            print(f"      • VaR (95%): ${var.get('var_amount', 0):,.2f} ({var.get('var_percent', 0):.2f}%)")
        
        if 'diversification' in risk_metrics:
            div = risk_metrics['diversification']
            print(f"      • Diversification Score: {div.get('score', 0):.1f}/100")
        
        if 'portfolio_risk_score' in risk_metrics:
            ps = risk_metrics['portfolio_risk_score']
            print(f"      • Portfolio Risk Score: {ps.get('risk_score', 0):.2f}")
            print(f"      • Risk Level: {ps.get('risk_level', 'UNKNOWN')}")
        
        self.integration_tests.append({
            "test": "portfolio_risk",
            "result": risk_metrics
        })
        
        return risk_metrics
    
    async def test_risk_manager_queue(self):
        """
        Test RiskManager approved queue
        """
        print(f"\n   📋 Testing RiskManager Approved Queue:")
        print("   " + "-" * 40)
        
        queue_stats = self.risk_manager.approved_queue.get_stats()
        
        print(f"      • Queue Size: {queue_stats['current_size']}")
        print(f"      • Total Added: {queue_stats['total_added']}")
        print(f"      • Total Removed: {queue_stats['total_removed']}")
        print(f"      • Total Expired: {queue_stats['total_expired']}")
        
        next_trade = await self.risk_manager.approved_queue.get_next()
        
        if next_trade:
            print(f"      • Next Trade: {next_trade.get('ticker')} - {next_trade.get('action')} - ${next_trade.get('position_size', 0):,.2f}")
        else:
            print(f"      • No pending trades in queue")
        
        self.integration_tests.append({
            "test": "queue",
            "result": queue_stats
        })
        
        return queue_stats
    
    async def test_risk_manager_status(self):
        """
        Test RiskManager.get_status() method
        """
        print(f"\n   📊 Testing RiskManager.get_status():")
        print("   " + "-" * 40)
        
        status = self.risk_manager.get_status()
        
        print(f"      • Portfolio Capital: ${status['portfolio']['capital']:,.2f}")
        print(f"      • Portfolio Cash: ${status['portfolio']['cash']:,.2f}")
        print(f"      • Positions: {status['portfolio']['positions']}")
        print(f"      • Invested %: {status['portfolio']['invested_pct']:.1f}%")
        print(f"      • Current Regime: {status['current_regime']}")
        print(f"      • Risk Appetite: {status['risk_appetite']}")
        print(f"      • Queue Size: {status['queue']['current_size']}")
        print(f"      • Total Trades Processed: {status['queue']['total_processed']}")
        
        self.integration_tests.append({
            "test": "status",
            "result": status
        })
        
        return status
    
    async def test_risk_manager_regime_update(self):
        """
        Test RiskManager updating market regime
        """
        print(f"\n   🌍 Testing RiskManager.update_regime():")
        print("   " + "-" * 40)
        
        test_regimes = ["panic", "bull_trending", "high_volatility", "strong_bear_trending"]
        
        for regime in test_regimes:
            await self.risk_manager.process(
                AgentMessage(
                    sender="Test",
                    receiver="RiskManager",
                    message_type="update_regime",
                    content={"regime": regime}
                )
            )
            
            status = self.risk_manager.get_status()
            print(f"      • {regime}: Risk Appetite = {status['risk_appetite']}")
        
        await self.risk_manager.process(
            AgentMessage(
                sender="Test",
                receiver="RiskManager",
                message_type="update_regime",
                content={"regime": self.current_regime}
            )
        )
        
        self.integration_tests.append({
            "test": "regime_update",
            "result": {"regimes_tested": test_regimes}
        })
        
        return status
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    async def analyze_symbol_comprehensive(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """
        Run comprehensive risk analysis combining component and integration tests
        """
        print(f"\n" + "="*70)
        print(f"🛡️  COMPREHENSIVE RISK ANALYSIS FOR {symbol}")
        print("="*70)
        
        try:
            # Fetch current data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            data = ticker.history(period="6mo")
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
            
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                atr = (data['High'] - data['Low']).mean()
            else:
                volatility = 0.25
                atr = current_price * 0.02
            
            action = analysis.get('weighted', {}).get('action', 'WATCH')
            confidence = analysis.get('weighted', {}).get('confidence', 0.5)
            
            print(f"   📈 Current Price: ${current_price:.2f}")
            print(f"   📊 Volatility: {volatility*100:.1f}%")
            print(f"   🎯 Action: {action}")
            print(f"   🎲 Confidence: {confidence:.2f}")
            
            # ===== COMPONENT TESTS =====
            print("\n🔧 RUNNING COMPONENT TESTS...")
            
            # Test position sizing components
            position_sizing = await self.test_position_sizing_components(
                symbol, current_price, volatility, confidence
            )
            
            # Test stop loss components
            stop_loss = await self.test_stop_loss_components(
                symbol, current_price, volatility, atr
            )
            
            # Test risk scorer
            risk_score = await self.test_risk_scorer_component({
                "position_fraction": position_sizing['conservative'] / self.initial_capital,
                "stop_percent": stop_loss['stop_percent'],
                "volatility": volatility,
                "regime": self.current_regime,
                "volume": info.get('volume', 0),
                "avg_volume": info.get('averageVolume', 0),
                "avg_portfolio_correlation": 0.6,
                "leverage": 1.0,
                "time_horizon_days": 5
            })
            
            # Test market regime
            regime_adjusted = await self.test_market_regime_component(
                self.current_regime,
                {
                    "position_size": position_sizing['conservative'],
                    "position_fraction": position_sizing['conservative'] / self.initial_capital,
                    "stop_distance": stop_loss['risk_per_share'],
                    "stop_percent": stop_loss['stop_percent'],
                    "risk_per_trade": position_sizing['conservative'] * (stop_loss['stop_percent'] / 100)
                }
            )
            
            # ===== INTEGRATION TESTS =====
            print("\n🔗 RUNNING INTEGRATION TESTS...")
            
            # Test RiskManager calculate
            risk_manager_result = await self.test_risk_manager_calculate(
                symbol, analysis, current_price, volatility
            )
            
            # Store combined results
            result = {
                "symbol": symbol,
                "price": current_price,
                "volatility": volatility,
                "action": action,
                "confidence": confidence,
                "component_tests": {
                    "position_sizing": position_sizing,
                    "stop_loss": stop_loss,
                    "risk_score": risk_score,
                    "regime_adjusted": regime_adjusted
                },
                "integration_tests": {
                    "risk_manager": risk_manager_result
                },
                "recommended_position": risk_manager_result.get('position_size', position_sizing['conservative']),
                "recommended_shares": int(risk_manager_result.get('position_size', position_sizing['conservative']) / current_price) if current_price > 0 else 0,
                "recommended_stop": risk_manager_result.get('stop_price', stop_loss['atr_stop']),
                "risk_score": risk_manager_result.get('risk_score', risk_score['risk_score']),
                "risk_level": risk_manager_result.get('risk_level', risk_score['risk_level']),
                "should_trade": risk_manager_result.get('risk_score', 1) <= 0.7 and risk_manager_result.get('risk_level') != 'HIGH'
            }
            
            self.results.append(result)
            
            print(f"\n   🎯 FINAL RECOMMENDATION FOR {symbol}:")
            print(f"      • Should Trade: {'✅ YES' if result['should_trade'] else '❌ NO'}")
            print(f"      • Position: ${result['recommended_position']:,.2f} ({result['recommended_shares']} shares)")
            print(f"      • Stop Loss: ${result['recommended_stop']:.2f}")
            print(f"      • Risk Score: {result['risk_score']:.2f}")
            print(f"      • Risk Level: {result['risk_level']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.errors.append({"symbol": symbol, "error": str(e)})
            return None
    
    async def run_comprehensive(self):
        """
        Run comprehensive risk analysis for all symbols
        """
        print("\n" + "="*70)
        print("🚀 COMPREHENSIVE RISK MODULE TEST")
        print("="*70)
        
        # Set initial market regime
        await self.risk_manager.process(
            AgentMessage(
                sender="Test",
                receiver="RiskManager",
                message_type="update_regime",
                content={"regime": self.current_regime}
            )
        )
        
        # Show loaded analysis results
        print("\n📋 Loaded Analysis Results:")
        for a in self.analysis_results:
            print(f"   • {a['symbol']}: {a['weighted']['action']} (score: {a['weighted']['final_score']:.2f})")
        
        # Analyze each symbol comprehensively
        for analysis in self.analysis_results:
            await self.analyze_symbol_comprehensive(analysis['symbol'], analysis)
            await asyncio.sleep(0.5)
        
        # Test portfolio-level risk with RiskManager
        await self.test_risk_manager_portfolio_risk()
        
        # Test queue management
        await self.test_risk_manager_queue()
        
        # Test RiskManager status
        await self.test_risk_manager_status()
        
        # Test regime update
        await self.test_risk_manager_regime_update()
        
        # Test portfolio risk components
        positions = []
        for r in self.results:
            if r and r['should_trade']:
                positions.append({
                    "symbol": r['symbol'],
                    "value": r['recommended_position'],
                    "weight": r['recommended_position'] / self.initial_capital,
                    "sector": "Technology",
                    "volatility": r['volatility']
                })
        
        if positions:
            await self.test_portfolio_risk_components(positions)
        
        # Print final summary
        self.print_comprehensive_summary()
        
        # Save results
        self.save_comprehensive_results()
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary of all tests"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE RISK ANALYSIS SUMMARY")
        print("="*70)
        
        if not self.results:
            print("   No results to display")
            return
        
        # Component test summary
        print("\n🔧 COMPONENT TESTS SUMMARY:")
        component_types = {}
        for test in self.component_tests:
            test_type = test['test']
            component_types[test_type] = component_types.get(test_type, 0) + 1
        
        for test_type, count in component_types.items():
            print(f"   • {test_type}: {count} tests completed")
        
        # Integration test summary
        print("\n🔗 INTEGRATION TESTS SUMMARY:")
        for test in self.integration_tests:
            print(f"   • {test['test']}: {'✓' if test['result'] else '✗'}")
        
        # Trading recommendations
        print(f"\n📈 TRADING RECOMMENDATIONS:")
        print(f"\n{'Symbol':<8} {'Action':<10} {'Should Trade':<13} {'Position':<12} {'Shares':<8} {'Stop':<10} {'Risk':<10}")
        print("-" * 80)
        
        for r in self.results:
            symbol = r['symbol']
            action = r['action']
            should_trade = "✅ YES" if r['should_trade'] else "❌ NO"
            position = f"${r['recommended_position']:,.0f}"
            shares = r['recommended_shares']
            stop = f"${r['recommended_stop']:.2f}"
            risk = r['risk_level']
            
            print(f"{symbol:<8} {action:<10} {should_trade:<13} {position:<12} {shares:<8} {stop:<10} {risk:<10}")
        
        # Statistics
        print("\n" + "="*70)
        print("📊 STATISTICS")
        print("="*70)
        
        total_portfolio = sum(r['recommended_position'] for r in self.results if r['should_trade'])
        tradeable = sum(1 for r in self.results if r['should_trade'])
        
        print(f"   • Total Portfolio Value: ${total_portfolio:,.2f}")
        print(f"   • Average Position: ${total_portfolio/tradeable:,.2f}" if tradeable > 0 else "   • Average Position: N/A")
        print(f"   • Tradeable Signals: {tradeable}/{len(self.results)}")
        print(f"   • Average Risk Score: {sum(r['risk_score'] for r in self.results)/len(self.results):.2f}")
        
        # Component test counts
        print(f"   • Component Tests: {len(self.component_tests)}")
        print(f"   • Integration Tests: {len(self.integration_tests)}")
        print(f"   • Errors: {len(self.errors)}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        buys = [r for r in self.results if r['should_trade'] and r['action'] != 'SELL']
        sells = [r for r in self.results if r['action'] == 'SELL']
        watches = [r for r in self.results if not r['should_trade'] and r['action'] != 'SELL']
        
        if buys:
            print(f"   ✅ BUY: {', '.join([r['symbol'] for r in buys])}")
        if watches:
            print(f"   👀 WATCH: {', '.join([r['symbol'] for r in watches])}")
        if sells:
            print(f"   ❌ SELL: {', '.join([r['symbol'] for r in sells])}")
        
        # Risk warnings
        high_risk = [r for r in self.results if r['risk_level'] in ['HIGH', 'VERY_HIGH']]
        if high_risk:
            print(f"\n⚠️ HIGH RISK POSITIONS:")
            for r in high_risk:
                print(f"   • {r['symbol']}: Risk Score {r['risk_score']:.2f}")
    
    def save_comprehensive_results(self):
        """Save comprehensive results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(self.results),
            "component_tests": self.component_tests,
            "integration_tests": self.integration_tests,
            "errors": self.errors,
            "results": self.results,
            "summary": {
                "total_portfolio": sum(r['recommended_position'] for r in self.results if r['should_trade']),
                "tradeable_signals": sum(1 for r in self.results if r['should_trade']),
                "avg_risk_score": sum(r['risk_score'] for r in self.results) / len(self.results) if self.results else 0,
                "component_tests_count": len(self.component_tests),
                "integration_tests_count": len(self.integration_tests)
            }
        }
        
        # Create directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        with open("data/comprehensive_risk_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to: data/comprehensive_risk_results.json")

async def main():
    """Main entry point"""
    tester = ComprehensiveRiskTest()
    await tester.run_comprehensive()

if __name__ == "__main__":
    asyncio.run(main())