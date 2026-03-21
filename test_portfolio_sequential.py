#!/usr/bin/env python3
"""
Sequential Test: Portfolio Module with Full PortfolioOptimizer Integration
Tests allocation, rebalancing, and portfolio optimization
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

# Import portfolio components
from agentic_trading_system.portfolio.efficient_frontier import EfficientFrontier
from agentic_trading_system.portfolio.black_litterman import BlackLitterman
from agentic_trading_system.portfolio.risk_parity import RiskParity
from agentic_trading_system.portfolio.hierarchical_risk_parity import HierarchicalRiskParity
from agentic_trading_system.portfolio.allocation_engine import AllocationEngine
from agentic_trading_system.portfolio.rebalancing_signals import RebalancingSignals
from agentic_trading_system.portfolio.recommendation_generator import RecommendationGenerator
from agentic_trading_system.portfolio.portfolio_optimizer import PortfolioOptimizer
from agentic_trading_system.agents.base_agent import AgentMessage
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class PortfolioSequentialTest:
    """
    Sequential test for Portfolio module with full PortfolioOptimizer integration
    """
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
        self.risk_results = self.load_risk_results()
        self.results = []
        self.errors = []
        self.initial_capital = 100000
        
        # Initialize PortfolioOptimizer with complete configuration
        self.portfolio_optimizer = PortfolioOptimizer(
            name="TestPortfolioOptimizer",
            config={
                "initial_value": self.initial_capital,
                "initial_cash": self.initial_capital,
                "ef_config": {
                    "risk_free_rate": 0.02,
                    "max_weight": 0.25,
                    "min_weight": 0.01,
                    "max_iterations": 1000
                },
                "bl_config": {
                    "risk_aversion": 2.5,
                    "tau": 0.05,
                    "risk_free_rate": 0.02,
                    "max_weight": 0.25,
                    "min_weight": 0.01
                },
                "rp_config": {
                    "max_weight": 0.30,
                    "min_weight": 0.02,
                    "tolerance": 1e-6
                },
                "hrp_config": {
                    "linkage_method": "ward",
                    "correlation_method": "pearson",
                    "max_weight": 0.30,
                    "min_weight": 0.01
                },
                "allocation_config": {
                    "max_turnover": 0.20,
                    "tax_aware": True,
                    "tax_rates": {
                        "short_term": 0.35,
                        "long_term": 0.15
                    }
                },
                "rebalancing_config": {
                    "absolute_threshold": 0.05,
                    "relative_threshold": 0.20,
                    "calendar_days": 90,
                    "use_volatility_bands": True,
                    "volatility_multiplier": 2.0
                },
                "recommendation_config": {
                    "buy_threshold": 0.02,
                    "sell_threshold": 0.02
                },
                "constraints": {
                    "max_position": 0.25,
                    "min_position": 0.01,
                    "max_sector": 0.30,
                    "max_turnover": 0.20
                }
            }
        )
        
        logging.info("✅ PortfolioOptimizer initialized with complete configuration")
    
    def _extract_risk_score(self, entry):
        """Extract numeric risk score from a risk result entry."""
        risk_score = entry.get("risk_score")
        if isinstance(risk_score, dict):
            for key in ("value", "score"):
                if key in risk_score and isinstance(risk_score[key], (int, float)):
                    return float(risk_score[key])
            for v in risk_score.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        elif isinstance(risk_score, (int, float)):
            return float(risk_score)
        else:
            return 0.0

    def load_risk_results(self):
        """Load previous risk results with robust flattening."""
        MOCK_DATA = [
            {"symbol": "AAPL", "recommended_position": 1664, "recommended_shares": 6, "price": 247.99, "risk_score": 0.38, "action": "WATCH"},
            {"symbol": "MSFT", "recommended_position": 1341, "recommended_shares": 3, "price": 381.87, "risk_score": 0.40, "action": "WATCH"},
            {"symbol": "GOOGL", "recommended_position": 1327, "recommended_shares": 4, "price": 301.00, "risk_score": 0.40, "action": "WATCH"},
            {"symbol": "TSLA", "recommended_position": 843, "recommended_shares": 2, "price": 367.96, "risk_score": 0.46, "action": "SELL"},
            {"symbol": "NVDA", "recommended_position": 983, "recommended_shares": 5, "price": 172.70, "risk_score": 0.43, "action": "WATCH"},
            {"symbol": "META", "recommended_position": 1026, "recommended_shares": 1, "price": 593.66, "risk_score": 0.43, "action": "WATCH"},
        ]

        def _normalise(raw):
            if isinstance(raw, dict):
                normalized = []
                for sym, val in raw.items():
                    if isinstance(val, dict):
                        if "symbol" not in val:
                            val["symbol"] = sym
                        normalized.append(val)
                return [r for r in normalized if isinstance(r, dict) and "symbol" in r]
            elif isinstance(raw, list):
                return [r for r in raw if isinstance(r, dict) and "symbol" in r]
            return []

        try:
            with open("data/risk_results.json", "r") as fh:
                content = fh.read().strip()
                if not content:
                    raise ValueError("Empty file")
                data = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: risk_results.json unreadable ({exc}). Using mock data.")
            return MOCK_DATA

        if isinstance(data, dict):
            for key in ("results", "data", "risk_results"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        results = _normalise(data)
        if not results:
            print("Warning: risk_results.json had no valid entries. Using mock data.")
            return MOCK_DATA
        return results

    async def test_efficient_frontier(self):
        """Test 1: Efficient Frontier Optimization (Markowitz)"""
        print(f"\n   📈 Testing Efficient Frontier (Markowitz):")
        print("   " + "-" * 40)
        
        symbols = [r['symbol'] for r in self.risk_results if r.get('action') != 'SELL']
        np.random.seed(42)
        
        returns = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            mu = 0.0005 + i * 0.0001
            sigma = 0.01 + i * 0.001
            returns[symbol] = np.random.normal(mu, sigma, 252)
        
        corr_matrix = np.eye(len(symbols)) * 0.7 + 0.3
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                if i != j:
                    corr_matrix[i, j] = 0.5
        
        L = np.linalg.cholesky(corr_matrix)
        correlated_returns = returns.values @ L.T
        returns = pd.DataFrame(correlated_returns, columns=symbols)
        
        ef = EfficientFrontier({})
        max_sharpe = ef.optimize_max_sharpe(returns)
        min_vol = ef.optimize_min_volatility(returns)
        
        print(f"      • Max Sharpe Portfolio:")
        print(f"         Expected Return: {max_sharpe['expected_return']*100:.2f}%")
        print(f"         Volatility: {max_sharpe['volatility']*100:.2f}%")
        print(f"         Sharpe Ratio: {max_sharpe['sharpe_ratio']:.2f}")
        
        print(f"      • Min Volatility Portfolio:")
        print(f"         Expected Return: {min_vol['expected_return']*100:.2f}%")
        print(f"         Volatility: {min_vol['volatility']*100:.2f}%")
        print(f"         Sharpe Ratio: {min_vol['sharpe_ratio']:.2f}")
        
        top_weights = sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"      • Top Weights (Max Sharpe):")
        for symbol, weight in top_weights:
            print(f"         {symbol}: {weight*100:.1f}%")
        
        return {"max_sharpe": max_sharpe, "min_volatility": min_vol}
    
    async def test_risk_parity(self):
        """Test 2: Risk Parity Allocation"""
        print(f"\n   ⚖️ Testing Risk Parity Allocation:")
        print("   " + "-" * 40)
        
        symbols = [r['symbol'] for r in self.risk_results if r.get('action') != 'SELL']
        np.random.seed(42)
        
        returns = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            mu = 0.0005 + i * 0.0001
            sigma = 0.01 + i * 0.001
            returns[symbol] = np.random.normal(mu, sigma, 252)
        
        rp = RiskParity({})
        result = rp.optimize(returns)
        
        print(f"      • Risk Parity Weights:")
        top_weights = sorted(result['weights'].items(), key=lambda x: x[1], reverse=True)[:5]
        for symbol, weight in top_weights:
            print(f"         {symbol}: {weight*100:.1f}%")
        
        print(f"      • Expected Return: {result['expected_return']*100:.2f}%")
        print(f"      • Volatility: {result['volatility']*100:.2f}%")
        print(f"      • Risk Concentration: {result['risk_concentration']:.3f}")
        
        return result
    
    async def test_hierarchical_risk_parity(self):
        """
        Test 3: Hierarchical Risk Parity (HRP) using the actual HRP class
        """
        print(f"\n   🏛️ Testing Hierarchical Risk Parity (HRP) – Real Class:")
        print("   " + "-" * 40)
        
        # Use only active symbols (exclude SELL)
        symbols = [r['symbol'] for r in self.risk_results if r.get('action') != 'SELL']
        np.random.seed(42)
        
        # Generate realistic returns with correlations
        returns = pd.DataFrame()
        for i, symbol in enumerate(symbols):
            mu = 0.0005 + i * 0.0001
            sigma = 0.01 + i * 0.001
            returns[symbol] = np.random.normal(mu, sigma, 252)
        
        # Add correlation structure
        n = len(symbols)
        corr_matrix = 0.5 * np.ones((n, n)) + 0.5 * np.eye(n)
        L = np.linalg.cholesky(corr_matrix)
        returns = pd.DataFrame(returns.values @ L.T, columns=symbols)
        
        # Instantiate HRP with config
        hrp = HierarchicalRiskParity({
            "linkage_method": "ward",
            "max_weight": 0.30,
            "min_weight": 0.01
        })
        
        try:
            result = hrp.optimize(returns)
            print(f"      • HRP Weights:")
            top_weights = sorted(result['weights'].items(), key=lambda x: x[1], reverse=True)[:5]
            for symbol, weight in top_weights:
                print(f"         {symbol}: {weight*100:.1f}%")
            print(f"      • Expected Return: {result['expected_return']*100:.2f}%")
            print(f"      • Volatility: {result['volatility']*100:.2f}%")
            print(f"      • Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        except Exception as e:
            print(f"      ❌ HRP optimization failed: {e}")
            result = {"error": str(e)}
        
        return result
    
    async def test_black_litterman(self):
        """
        Test 4: Black‑Litterman model with views
        """
        print(f"\n   🎯 Testing Black‑Litterman Model:")
        print("   " + "-" * 40)
        
        # Use only active symbols
        symbols = [r['symbol'] for r in self.risk_results if r.get('action') != 'SELL']
        np.random.seed(42)
        
        # Generate price data (needed for returns)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        prices = pd.DataFrame(index=dates)
        for symbol in symbols:
            # Random walk starting from the price from risk_results
            base_price = next(r['price'] for r in self.risk_results if r['symbol'] == symbol)
            returns = np.random.normal(0.0005, 0.02, 251)
            price_series = [base_price]
            for r in returns:
                price_series.append(price_series[-1] * (1 + r))
            prices[symbol] = price_series
        
        # Mock market caps (in billions)
        market_caps = {
            "AAPL": 2800e9,
            "MSFT": 2500e9,
            "GOOGL": 1700e9,
            "NVDA": 1200e9,
            "META": 900e9
        }
        
        # Define views
        views = [
            {
                "assets": ["AAPL"],
                "type": "absolute",
                "return": 0.12,      # expect 12% return
                "confidence": 0.8
            },
            {
                "assets": ["MSFT", "GOOGL"],
                "type": "relative",
                "return": 0.05,      # MSFT outperforms GOOGL by 5%
                "confidence": 0.6
            }
        ]
        
        bl = BlackLitterman({
            "risk_aversion": 2.5,
            "tau": 0.05,
            "risk_free_rate": 0.02,
            "max_weight": 0.25,
            "min_weight": 0.01
        })
        
        try:
            result = bl.optimize(prices, market_caps, views)
            print(f"      • Black‑Litterman Weights:")
            top_weights = sorted(result['weights'].items(), key=lambda x: x[1], reverse=True)[:5]
            for symbol, weight in top_weights:
                print(f"         {symbol}: {weight*100:.1f}%")
            print(f"      • Expected Return: {result['expected_return']*100:.2f}%")
            print(f"      • Volatility: {result['volatility']*100:.2f}%")
            print(f"      • Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"      • Implied Returns (selected):")
            implied = result['implied_returns']
            for sym in symbols[:3]:
                print(f"         {sym}: {implied.get(sym, 0)*100:.2f}%")
            print(f"      • Posterior Returns (selected):")
            post = result['posterior_returns']
            for sym in symbols[:3]:
                print(f"         {sym}: {post.get(sym, 0)*100:.2f}%")
        except Exception as e:
            print(f"      ❌ Black‑Litterman optimization failed: {e}")
            result = {"error": str(e)}
        
        return result
    
    async def test_allocation_engine(self):
        """Test 5: Allocation Engine - Convert risk results to allocations"""
        print(f"\n   📊 Testing Allocation Engine:")
        print("   " + "-" * 40)
        
        # Current portfolio (empty)
        current_portfolio = {
            "positions": [],
            "total_value": self.initial_capital,
            "cash": self.initial_capital
        }
        
        # Build target allocation from risk results
        target_allocation = {}
        for r in self.risk_results:
            if r.get('action', 'WATCH') != 'SELL':
                target_allocation[r['symbol']] = r['recommended_position'] / self.initial_capital
        
        total_target = sum(target_allocation.values())
        if total_target > 0:
            target_allocation = {k: v/total_target for k, v in target_allocation.items()}
        
        print(f"      • Target Allocation:")
        for symbol, weight in list(target_allocation.items())[:5]:
            print(f"         {symbol}: {weight*100:.1f}%")
        
        # Create mock price data
        prices = pd.DataFrame()
        for r in self.risk_results:
            prices[r['symbol']] = [r.get('price', 100)]
        
        engine = AllocationEngine({})
        allocation = engine.allocate(
            current_portfolio,
            target_allocation,
            prices,
            cash=self.initial_capital
        )
        
        print(f"\n      • Recommended Trades:")
        for trade in allocation['trades'][:5]:
            print(f"         {trade['action']} {trade['symbol']}: ${trade['value']:,.2f} "
                  f"({trade['current_weight']*100:.1f}% → {trade['target_weight']*100:.1f}%)")
        
        print(f"      • Turnover: {allocation['turnover']*100:.1f}%")
        print(f"      • Net Cash Required: ${allocation['net_cash_required']:,.2f}")
        
        return allocation
    
    async def test_rebalancing_signals(self):
        """Test 6: Rebalancing Signals"""
        print(f"\n   🔄 Testing Rebalancing Signals:")
        print("   " + "-" * 40)
        
        current_allocation = {}
        for r in self.risk_results:
            if r.get('action', 'WATCH') != 'SELL':
                current_allocation[r['symbol']] = r['recommended_position'] / self.initial_capital
        
        target_allocation = {k: 1/len(current_allocation) for k in current_allocation.keys()}
        
        rebalancer = RebalancingSignals({})
        signal = rebalancer.check_rebalance(
            current_allocation,
            target_allocation,
            self.initial_capital,
            volatility=0.20
        )
        
        print(f"      • Needs Rebalance: {signal['needs_rebalance']}")
        print(f"      • Urgency: {signal['urgency']}")
        print(f"      • Max Drift: {signal['max_drift']*100:.2f}%")
        print(f"      • Avg Drift: {signal['avg_drift']*100:.2f}%")
        
        if signal['reasons']:
            print(f"      • Reasons: {signal['reasons'][0]}")
        
        if signal.get('drifting_assets'):
            print(f"      • Drifting Assets:")
            for asset in signal['drifting_assets'][:3]:
                print(f"         {asset['symbol']}: {asset['current']*100:.1f}% vs {asset['target']*100:.1f}%")
        
        return signal
    
    async def test_recommendation_generator(self, allocation: Dict, rebalance_signal: Dict):
        """Test 7: Recommendation Generator"""
        print(f"\n   📝 Testing Recommendation Generator:")
        print("   " + "-" * 40)
        
        generator = RecommendationGenerator({})
        recommendations = generator.generate_recommendations(
            allocation,
            rebalance_signal,
            market_outlook={"risk_level": "HIGH", "outlook": "bearish"}
        )
        
        print(f"      • Priority: {recommendations['priority']}")
        print(f"      • Summary: {recommendations['summary']}")
        
        if recommendations['trades']:
            print(f"      • Trade Recommendations:")
            for trade in recommendations['trades'][:5]:
                print(f"         {trade['action']} {trade['symbol']}: ${trade['value']:,.0f} - {trade['reason']}")
        
        if recommendations['risk_alerts']:
            print(f"      • Risk Alerts: {len(recommendations['risk_alerts'])}")
            for alert in recommendations['risk_alerts'][:2]:
                print(f"         {alert['message']}")
        
        if recommendations['performance_insights']:
            print(f"      • Insights: {len(recommendations['performance_insights'])}")
            for insight in recommendations['performance_insights'][:2]:
                print(f"         {insight['message']}")
        
        return recommendations
    
    async def test_portfolio_optimizer_optimize(self):
        """Test 8: PortfolioOptimizer.optimize() method"""
        print(f"\n   🏦 Testing PortfolioOptimizer.optimize():")
        print("   " + "-" * 40)
        
        symbols = [r['symbol'] for r in self.risk_results if r.get('action') != 'SELL']
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = pd.DataFrame(index=dates)
        
        for symbol in symbols:
            base_price = next(r['price'] for r in self.risk_results if r['symbol'] == symbol)
            returns = np.random.normal(0.0005, 0.02, 99)
            price_series = [base_price]
            for r in returns:
                price_series.append(price_series[-1] * (1 + r))
            prices[symbol] = price_series
        
        methods = ["max_sharpe", "min_volatility", "risk_parity", "hierarchical_risk_parity"]
        results = {}
        for method in methods:
            print(f"\n      • Method: {method}")
            result = await self.portfolio_optimizer.optimize(prices, method)
            
            print(f"         Expected Return: {result.get('expected_return', 0)*100:.2f}%")
            print(f"         Volatility: {result.get('volatility', 0)*100:.2f}%")
            print(f"         Sharpe: {result.get('sharpe_ratio', 0):.2f}")
            
            weights = result.get('weights', {})
            top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"         Top Weights: {', '.join([f'{s}: {w*100:.1f}%' for s, w in top_weights])}")
            
            results[method] = result
        
        return results
    
    async def test_portfolio_optimizer_check_rebalance(self):
        """Test 9: PortfolioOptimizer.check_rebalance() method"""
        print(f"\n   🔄 Testing PortfolioOptimizer.check_rebalance():")
        print("   " + "-" * 40)
        
        self.portfolio_optimizer.portfolio["target_allocation"] = {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "GOOGL": 0.20,
            "NVDA": 0.20,
            "META": 0.20
        }
        
        self.portfolio_optimizer.portfolio["positions"] = [
            {"symbol": "AAPL", "value": 25000, "shares": 100},
            {"symbol": "MSFT", "value": 15000, "shares": 40},
            {"symbol": "GOOGL", "value": 20000, "shares": 66},
            {"symbol": "NVDA", "value": 18000, "shares": 104},
            {"symbol": "META", "value": 12000, "shares": 20}
        ]
        self.portfolio_optimizer.portfolio["total_value"] = 90000
        self.portfolio_optimizer.portfolio["cash"] = 10000
        
        signal = await self.portfolio_optimizer.check_rebalance()
        
        print(f"      • Needs Rebalance: {signal['needs_rebalance']}")
        print(f"      • Urgency: {signal['urgency']}")
        print(f"      • Max Drift: {signal['max_drift']*100:.2f}%")
        
        if signal.get('drifting_assets'):
            print(f"      • Drifting Assets:")
            for asset in signal['drifting_assets'][:3]:
                print(f"         {asset['symbol']}: {asset['current']*100:.1f}% vs {asset['target']*100:.1f}%")
        
        return signal
    
    async def test_portfolio_optimizer_generate_recommendations(self):
        """Test 10: PortfolioOptimizer.generate_recommendations() method"""
        print(f"\n   📝 Testing PortfolioOptimizer.generate_recommendations():")
        print("   " + "-" * 40)
        
        market_outlook = {
            "risk_level": "HIGH",
            "outlook": "bearish",
            "warning": "Elevated market volatility"
        }
        
        recommendations = await self.portfolio_optimizer.generate_recommendations(market_outlook)
        
        print(f"      • Recommendations Summary:")
        print(f"         {recommendations.get('summary', 'No summary')}")
        
        if recommendations.get('trades'):
            print(f"      • Trades: {len(recommendations['trades'])}")
            for trade in recommendations['trades'][:3]:
                print(f"         {trade['action']} {trade['symbol']}: ${trade['value']:,.0f}")
        
        if recommendations.get('risk_alerts'):
            print(f"      • Risk Alerts: {len(recommendations['risk_alerts'])}")
        
        return recommendations
    
    async def test_portfolio_optimizer_get_status(self):
        """Test 11: PortfolioOptimizer.get_status() method"""
        print(f"\n   📊 Testing PortfolioOptimizer.get_status():")
        print("   " + "-" * 40)
        
        status = self.portfolio_optimizer.get_status()
        
        print(f"      • Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
        print(f"      • Cash: ${status['portfolio']['cash']:,.2f}")
        print(f"      • Invested: ${status['portfolio']['invested']:,.2f}")
        print(f"      • Positions: {status['portfolio']['num_positions']}")
        print(f"      • Has Target Allocation: {status['portfolio']['has_target']}")
        print(f"      • Current Regime: {status['current_regime']}")
        print(f"      • Constraints: {status['constraints']}")
        
        return status
    
    async def test_portfolio_optimizer_update_portfolio(self):
        """Test 12: PortfolioOptimizer.update_portfolio() method"""
        print(f"\n   🔄 Testing PortfolioOptimizer.update_portfolio():")
        print("   " + "-" * 40)
        
        await self.portfolio_optimizer._update_portfolio({
            "action": "add_position",
            "position": {
                "symbol": "TEST",
                "shares": 100,
                "price": 50,
                "value": 5000,
                "cost_basis": 5000,
                "sector": "Technology",
                "entry_time": datetime.now().isoformat()
            }
        })
        
        await self.portfolio_optimizer._update_portfolio({
            "action": "update_prices",
            "prices": {"AAPL": 260, "MSFT": 400, "GOOGL": 310, "TEST": 55}
        })
        
        status = self.portfolio_optimizer.get_status()
        
        print(f"      • Updated Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
        print(f"      • Updated Positions: {status['portfolio']['num_positions']}")
        print(f"      • Cash: ${status['portfolio']['cash']:,.2f}")
        
        return status
    
    async def run(self):
        """Run all portfolio tests"""
        print("\n" + "="*70)
        print("🚀 PORTFOLIO MODULE SEQUENTIAL TEST")
        print("="*70)
        
        print("\n📋 Loaded Risk Results:")
        for r in self.risk_results:
            action = r.get('action', 'WATCH')
            print(f"   • {r['symbol']}: ${r['recommended_position']:,.0f} ({r['recommended_shares']} shares) - {action}")
        
        ef_results = await self.test_efficient_frontier()
        rp_results = await self.test_risk_parity()
        hrp_results = await self.test_hierarchical_risk_parity()
        bl_results = await self.test_black_litterman()
        
        allocation = await self.test_allocation_engine()
        rebalance_signal = await self.test_rebalancing_signals()
        recommendations = await self.test_recommendation_generator(allocation, rebalance_signal)
        optimize_results = await self.test_portfolio_optimizer_optimize()
        check_rebalance = await self.test_portfolio_optimizer_check_rebalance()
        gen_recommendations = await self.test_portfolio_optimizer_generate_recommendations()
        status = await self.test_portfolio_optimizer_get_status()
        update_status = await self.test_portfolio_optimizer_update_portfolio()
        
        self.print_summary()
        
        self.save_results({
            "efficient_frontier": ef_results,
            "risk_parity": rp_results,
            "hierarchical_risk_parity": hrp_results,
            "black_litterman": bl_results,
            "allocation": allocation,
            "rebalance_signal": rebalance_signal,
            "recommendations": recommendations,
            "optimize_results": optimize_results,
            "check_rebalance": check_rebalance,
            "gen_recommendations": gen_recommendations,
            "status": status,
            "update_status": update_status
        })
    
    def print_summary(self):
        """Print summary of portfolio analysis"""
        print("\n" + "="*70)
        print("📊 PORTFOLIO MODULE SUMMARY")
        print("="*70)

        active = [r for r in self.risk_results
                  if isinstance(r, dict) and r.get("action") != "SELL"]

        total_risk_position = sum(r.get("recommended_position", 0) for r in active)
        print(f"\n💰 Capital Utilization:")
        print(f"   • Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   • Total Proposed Positions: ${total_risk_position:,.2f}")
        util_pct = (total_risk_position / self.initial_capital * 100) if self.initial_capital else 0
        print(f"   • Capital Utilization: {util_pct:.1f}%")
        print(f"   • Remaining Cash: ${self.initial_capital - total_risk_position:,.2f}")

        print(f"\n🎯 Trade Recommendations:")
        for r in active:
            print(f"   • BUY {r.get('recommended_shares', 0)} {r['symbol']} @ ${r.get('price', 0):.2f} = ${r.get('recommended_position', 0):,.2f}")

        symbols = [r["symbol"] for r in active]
        print(f"\n📊 Portfolio Diversification:")
        print(f"   • Number of Assets: {len(symbols)}")
        print(f"   • Assets: {', '.join(symbols)}")

        print(f"\n⚠️ Risk Summary:")
        if active:
            risk_scores = [self._extract_risk_score(r) for r in active]
            avg_risk = sum(risk_scores) / len(risk_scores)
            print(f"   • Average Risk Score: {avg_risk:.2f}")
            print(f"   • Risk Level: {'LOW' if avg_risk < 0.5 else 'MEDIUM' if avg_risk < 0.7 else 'HIGH'}")
        else:
            print(f"   • No active positions to assess.")

        if total_risk_position > 0:
            print(f"\n🔄 Rebalancing Recommendation:")
            print(f"   • Consider rebalancing monthly or when drift > 5%")
            print(f"   • Current allocation is {'well diversified' if len(symbols) >= 5 else 'concentrated'}")

    def save_results(self, results):
        """Save results to file"""
        active = [r for r in self.risk_results
                  if isinstance(r, dict) and r.get("action") != "SELL"]
        total_investment = sum(r.get("recommended_position", 0) for r in active)
        num_positions = len(active)

        output = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "risk_results": self.risk_results,
            "portfolio_results": results,
            "summary": {
                "total_investment": total_investment,
                "capital_utilization": (total_investment / self.initial_capital * 100) if self.initial_capital else 0,
                "remaining_cash": self.initial_capital - total_investment,
                "num_positions": num_positions,
                "avg_risk_score": (
                    sum(self._extract_risk_score(r) for r in active) / num_positions
                    if num_positions else 0
                ),
            },
        }

        Path("data").mkdir(exist_ok=True)

        with open("data/portfolio_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n💾 Results saved to: data/portfolio_results.json")

async def main():
    """Main entry point"""
    tester = PortfolioSequentialTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())