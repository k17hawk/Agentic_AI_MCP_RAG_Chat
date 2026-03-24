#!/usr/bin/env python3
"""
Sequential Test: Execution Module
Tests order types, broker connectivity, settlement, and position tracking
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import random

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.execution.execution_engine import ExecutionEngine
from agentic_trading_system.execution.order_manager import OrderManager
from agentic_trading_system.execution.order_types.market_order import MarketOrder
from agentic_trading_system.execution.order_types.limit_order import LimitOrder
from agentic_trading_system.execution.order_types.stop_order import StopOrder
from agentic_trading_system.execution.order_types.trailing_stop_order import TrailingStopOrder
from agentic_trading_system.execution.broker_connectors.paper_trading import PaperTrading
from agentic_trading_system.execution.broker_connectors.mock_broker import MockBroker
from agentic_trading_system.execution.fills_manager import FillsManager
from agentic_trading_system.execution.open_positions import OpenPositions
from agentic_trading_system.execution.settlement import Settlement
from agentic_trading_system.agents.base_agent import AgentMessage
from agentic_trading_system.utils.logger import logging



class ExecutionSequentialTest:
    """
    Sequential test for Execution module
    """
    
    def __init__(self):
        self.initial_capital = 100000
        self.execution_engine = None
        self.results = []
        self.errors = []
        
        logging.info("✅ Execution test initialized")
    
    async def setup(self):
        """Initialize execution engine with paper trading"""
        
        # Create ExecutionEngine with paper trading mode
        self.execution_engine = ExecutionEngine(
            name="TestExecutionEngine",
            config={
                "mode": "paper",  # Paper trading mode
                "initial_capital": self.initial_capital,
                "paper_config": {
                    "initial_capital": self.initial_capital,
                    "fill_delay_seconds": 1,
                    "slippage_model": "percentage",
                    "percentage_slippage": 0.001
                },
                "order_config": {
                    "max_history": 1000
                },
                "routing_config": {
                    "default_strategy": "lowest_cost"
                }
            }
        )
        
        # Set market open
        self.execution_engine.is_market_open = True
        
        logging.info("✅ ExecutionEngine initialized (paper trading mode)")
    
    async def test_market_order(self):
        """
        Test 1: Market Order
        """
        print(f"\n   📈 Testing Market Order:")
        print("   " + "-" * 40)
        
        # Create market order
        order = {
            "symbol": "AAPL",
            "quantity": 10,
            "side": "BUY",
            "order_type": "MARKET",
            "time_in_force": "DAY"
        }
        
        # Submit order
        result = await self.execution_engine.execute_trade(order, "Test")
        
        print(f"      • Order ID: {result.content.get('order_id')}")
        print(f"      • Symbol: {result.content.get('symbol')}")
        print(f"      • Quantity: {result.content.get('quantity')}")
        print(f"      • Status: {result.content.get('result', {}).get('status', 'unknown')}")
        
        if result.content.get('result', {}).get('filled_price'):
            print(f"      • Filled Price: ${result.content['result']['filled_price']:.2f}")
        
        return result.content
    
    async def test_limit_order(self):
        """
        Test 2: Limit Order
        """
        print(f"\n   📈 Testing Limit Order:")
        print("   " + "-" * 40)
        
        # Get current price
        ticker = yf.Ticker("MSFT")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Create limit order (buy below market)
        limit_price = current_price * 0.98  # 2% below market
        
        order = {
            "symbol": "MSFT",
            "quantity": 5,
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": limit_price,
            "time_in_force": "DAY"
        }
        
        # Submit order
        result = await self.execution_engine.execute_trade(order, "Test")
        
        print(f"      • Current Price: ${current_price:.2f}")
        print(f"      • Limit Price: ${limit_price:.2f}")
        print(f"      • Order ID: {result.content.get('order_id')}")
        print(f"      • Status: {result.content.get('result', {}).get('status', 'unknown')}")
        
        return result.content
    
    async def test_stop_order(self):
        """
        Test 3: Stop Order (Stop Loss)
        """
        print(f"\n   📈 Testing Stop Order:")
        print("   " + "-" * 40)
        
        # First, buy some shares to have a position
        buy_order = {
            "symbol": "NVDA",
            "quantity": 5,
            "side": "BUY",
            "order_type": "MARKET"
        }
        
        buy_result = await self.execution_engine.execute_trade(buy_order, "Test")
        
        # Get current price
        ticker = yf.Ticker("NVDA")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Create stop loss order (sell if price drops 5%)
        stop_price = current_price * 0.95
        
        stop_order = {
            "symbol": "NVDA",
            "quantity": 5,
            "side": "SELL",
            "order_type": "STOP",
            "stop_price": stop_price,
            "stop_type": "STOP_LOSS",
            "time_in_force": "GTC"
        }
        
        # Submit stop order
        result = await self.execution_engine.execute_trade(stop_order, "Test")
        
        print(f"      • Current Price: ${current_price:.2f}")
        print(f"      • Stop Price: ${stop_price:.2f}")
        print(f"      • Order ID: {result.content.get('order_id')}")
        print(f"      • Status: {result.content.get('result', {}).get('status', 'unknown')}")
        
        return result.content
    
    async def test_trailing_stop(self):
        """
        Test 4: Trailing Stop Order
        """
        print(f"\n   📈 Testing Trailing Stop Order:")
        print("   " + "-" * 40)
        
        # Get current price
        ticker = yf.Ticker("GOOGL")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Create trailing stop order
        trailing_order = {
            "symbol": "GOOGL",
            "quantity": 3,
            "side": "SELL",
            "order_type": "TRAILING_STOP",
            "trail_type": "PERCENT",
            "trail_value": 5.0,  # 5% trail
            "activation_percent": 2.0,  # Activate after 2% profit
            "entry_price": current_price
        }
        
        # Submit trailing stop order
        result = await self.execution_engine.execute_trade(trailing_order, "Test")
        
        print(f"      • Current Price: ${current_price:.2f}")
        print(f"      • Trail Type: PERCENT")
        print(f"      • Trail Value: 5.0%")
        print(f"      • Order ID: {result.content.get('order_id')}")
        print(f"      • Status: {result.content.get('result', {}).get('status', 'unknown')}")
        
        return result.content
    
    async def test_cancel_order(self):
        """
        Test 5: Cancel Order
        """
        print(f"\n   📈 Testing Cancel Order:")
        print("   " + "-" * 40)
        
        # Create a limit order that won't fill immediately
        ticker = yf.Ticker("META")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        limit_price = current_price * 0.90  # 10% below market (unlikely to fill)
        
        order = {
            "symbol": "META",
            "quantity": 2,
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": limit_price,
            "time_in_force": "DAY"
        }
        
        # Submit order
        result = await self.execution_engine.execute_trade(order, "Test")
        order_id = result.content.get('order_id')
        
        print(f"      • Order ID: {order_id}")
        print(f"      • Status: {result.content.get('result', {}).get('status', 'unknown')}")
        
        # Cancel the order
        cancel_result = await self.execution_engine.cancel_order(order_id)
        
        print(f"      • Cancel Success: {cancel_result.get('success', False)}")
        
        return cancel_result
    
    async def test_get_positions(self):
        """
        Test 6: Get Positions
        """
        print(f"\n   📈 Testing Get Positions:")
        print("   " + "-" * 40)
        
        positions = await self.execution_engine.get_positions()
        
        print(f"      • Total Positions: {positions.get('total_positions', 0)}")
        
        for pos in positions.get('positions', [])[:5]:
            print(f"      • {pos['symbol']}: {pos['shares']} shares @ ${pos['avg_price']:.2f} "
                  f"(P&L: ${pos.get('unrealized_pl', 0):+.2f})")
        
        return positions
    
    async def test_account_summary(self):
        """
        Test 7: Account Summary
        """
        print(f"\n   📈 Testing Account Summary:")
        print("   " + "-" * 40)
        
        summary = await self.execution_engine.get_account_summary()
        
        print(f"      • Mode: {summary.get('mode', 'unknown')}")
        print(f"      • Total Value: ${summary.get('broker', {}).get('portfolio_value', 0):,.2f}")
        print(f"      • Cash: ${summary.get('cash', {}).get('cash_balance', 0):,.2f}")
        print(f"      • Invested: ${summary.get('invested', 0):,.2f}")
        print(f"      • Positions: {summary.get('positions', {}).get('total_positions', 0)}")
        
        return summary
    
    async def test_fills_manager(self):
        """
        Test 8: Fills Manager
        """
        print(f"\n   📈 Testing Fills Manager:")
        print("   " + "-" * 40)
        
        # Get recent fills
        fills = self.execution_engine.fills_manager.get_recent_fills(limit=10)
        
        print(f"      • Total Fills: {len(fills)}")
        
        for fill in fills[:5]:
            print(f"      • {fill.get('symbol')}: {fill.get('quantity')} shares @ ${fill.get('price', 0):.2f}")
        
        # Get stats
        stats = self.execution_engine.fills_manager.get_stats()
        print(f"      • Total Volume: {stats.get('total_volume', 0)} shares")
        print(f"      • Total Value: ${stats.get('total_value', 0):,.2f}")
        
        return fills
    
    async def test_settlement(self):
        """
        Test 9: Settlement (Cash Management)
        """
        print(f"\n   📈 Testing Settlement:")
        print("   " + "-" * 40)
        
        settlement = self.execution_engine.settlement
        
        # Process a trade
        trade_result = settlement.process_trade({
            "order_id": "TEST_123",
            "symbol": "AAPL",
            "quantity": 10,
            "price": 250.00,
            "side": "BUY"
        })
        
        print(f"      • Trade Processed: ${trade_result.get('trade_value', 0):,.2f}")
        print(f"      • Cash Balance: ${settlement.cash_balance:,.2f}")
        
        # Get cash summary
        cash_summary = settlement.get_cash_summary()
        print(f"      • Cash Summary: ${cash_summary.get('cash_balance', 0):,.2f}")
        print(f"      • Available Cash: ${cash_summary.get('available_cash', 0):,.2f}")
        
        # Get settlement schedule
        schedule = settlement.get_settlement_schedule()
        print(f"      • Pending Settlements: {schedule.get('total_pending', 0)}")
        
        return settlement
    
    async def test_stop_monitoring(self):
        """
        Test 10: Stop Monitoring (Simulate price movement)
        """
        print(f"\n   📈 Testing Stop Monitoring:")
        print("   " + "-" * 40)
        
        # Get positions
        positions = await self.execution_engine.get_positions()
        
        if positions.get('positions'):
            print(f"      • Active Positions: {len(positions['positions'])}")
            
            # Simulate price drop to trigger stop
            current_prices = {}
            for pos in positions['positions'][:2]:
                current_prices[pos['symbol']] = pos['avg_price'] * 0.94  # 6% drop
                print(f"      • Simulated price for {pos['symbol']}: ${current_prices[pos['symbol']]:.2f}")
            
            # Check stop orders
            triggered = self.execution_engine.order_manager.check_stop_orders(current_prices)
            
            print(f"      • Stops Triggered: {len(triggered)}")
            
            for t in triggered:
                print(f"         - {t.get('stop_order_id')}: triggered at ${t.get('trigger_price', 0):.2f}")
        
        return {"positions_monitored": len(positions.get('positions', []))}
    
    async def test_order_history(self):
        """
        Test 11: Order History
        """
        print(f"\n   📈 Testing Order History:")
        print("   " + "-" * 40)
        
        summary = self.execution_engine.order_manager.get_order_summary()
        
        print(f"      • Total Orders: {summary.get('total_orders', 0)}")
        print(f"      • Active Orders: {summary.get('active_orders', 0)}")
        print(f"      • Status Breakdown: {summary.get('status_breakdown', {})}")
        
        return summary
    
    async def test_paper_trading_simulation(self):
        """
        Test 12: Paper Trading Simulation
        """
        print(f"\n   📈 Testing Paper Trading Simulation:")
        print("   " + "-" * 40)
        
        # Get account status
        account = await self.execution_engine.broker.get_account()
        
        print(f"      • Account ID: {account.get('account_id', 'N/A')}")
        print(f"      • Cash: ${account.get('cash', 0):,.2f}")
        print(f"      • Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"      • Is Paper: {account.get('is_paper', False)}")
        
        return account
    
    async def test_full_trade_cycle(self):
        """
        Test 13: Full Trade Cycle (Simulate from approval to execution)
        """
        print(f"\n   📈 Testing Full Trade Cycle:")
        print("   " + "-" * 40)
        
        # Simulate an approved trade from HITL
        approved_trade = {
            "symbol": "AMZN",
            "action": "BUY",
            "quantity": 2,
            "price": 180.00,
            "shares": 2,
            "position_value": 360.00,
            "stop_loss": 171.00,
            "take_profit": 198.00,
            "confidence": 0.72,
            "risk_score": 0.35,
            "approved": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Execute the trade
        result = await self.execution_engine.process(
            AgentMessage(
                sender="HITL",
                receiver="ExecutionEngine",
                message_type="execute_trade",
                content=approved_trade
            )
        )
        
        print(f"      • Trade Executed: {result.message_type}")
        print(f"      • Symbol: {result.content.get('symbol', 'N/A')}")
        print(f"      • Quantity: {result.content.get('quantity', 0)}")
        
        if result.content.get('result', {}).get('success'):
            print(f"      • Status: EXECUTED")
            print(f"      • Fill Price: ${result.content['result'].get('filled_price', 0):.2f}")
        else:
            print(f"      • Status: {result.content.get('result', {}).get('status', 'FAILED')}")
        
        return result.content
    
    async def run(self):
        """
        Run all execution tests
        """
        print("\n" + "="*70)
        print("🚀 EXECUTION MODULE SEQUENTIAL TEST")
        print("="*70)
        
        # Setup
        await self.setup()
        
        # Test 1: Market Order
        await self.test_market_order()
        
        # Test 2: Limit Order
        await self.test_limit_order()
        
        # Test 3: Stop Order
        await self.test_stop_order()
        
        # Test 4: Trailing Stop
        await self.test_trailing_stop()
        
        # Test 5: Cancel Order
        await self.test_cancel_order()
        
        # Test 6: Get Positions
        await self.test_get_positions()
        
        # Test 7: Account Summary
        await self.test_account_summary()
        
        # Test 8: Fills Manager
        await self.test_fills_manager()
        
        # Test 9: Settlement
        await self.test_settlement()
        
        # Test 10: Stop Monitoring
        await self.test_stop_monitoring()
        
        # Test 11: Order History
        await self.test_order_history()
        
        # Test 12: Paper Trading Simulation
        await self.test_paper_trading_simulation()
        
        # Test 13: Full Trade Cycle
        await self.test_full_trade_cycle()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print summary of execution tests"""
        print("\n" + "="*70)
        print("📊 EXECUTION MODULE SUMMARY")
        print("="*70)
        
        # Get final status
        status = self.execution_engine.get_status()
        
        print(f"\n📈 Execution Engine Status:")
        print(f"   • Mode: {status.get('mode', 'unknown')}")
        print(f"   • Market Open: {status.get('market_open', False)}")
        print(f"   • Orders Today: {status.get('daily_stats', {}).get('orders_submitted', 0)}")
        print(f"   • Filled Today: {status.get('daily_stats', {}).get('orders_filled', 0)}")
        
        print(f"\n📊 Positions:")
        print(f"   • Open Positions: {status.get('positions_count', 0)}")
        print(f"   • Cash Balance: ${status.get('cash_balance', 0):,.2f}")
        
        print(f"\n📋 Order Summary:")
        order_summary = status.get('order_summary', {})
        print(f"   • Total Orders: {order_summary.get('total_orders', 0)}")
        print(f"   • Active Orders: {order_summary.get('active_orders', 0)}")
        
        print(f"\n💰 Settlement:")
        print(f"   • Cash Balance: ${status.get('cash_balance', 0):,.2f}")
    
    def save_results(self):
        """Save results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "execution_engine_status": self.execution_engine.get_status(),
            "positions": self.execution_engine.positions.get_all_positions(),
            "recent_fills": self.execution_engine.fills_manager.get_recent_fills(10),
            "cash_summary": self.execution_engine.settlement.get_cash_summary()
        }
        
        Path("data").mkdir(exist_ok=True)
        
        with open("data/execution_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: data/execution_results.json")

async def main():
    """Main entry point"""
    tester = ExecutionSequentialTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())