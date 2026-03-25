#!/usr/bin/env python3
"""
Complete Execution Module Test - Fixed with proper ExecutionEngine integration
"""
import asyncio
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime
import json
import uuid
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Import all execution components
from agentic_trading_system.execution.execution_engine import ExecutionEngine
from agentic_trading_system.execution.order_manager import OrderManager
from agentic_trading_system.execution.order_types.market_order import MarketOrder
from agentic_trading_system.execution.order_types.limit_order import LimitOrder
from agentic_trading_system.execution.order_types.stop_order import StopOrder
from agentic_trading_system.execution.order_types.trailing_stop_order import TrailingStopOrder
from agentic_trading_system.execution.broker_connectors.alpaca_client import AlpacaClient
from agentic_trading_system.execution.broker_connectors.ibkr_client import IBKRClient
from agentic_trading_system.execution.broker_connectors.paper_trading import PaperTrading
from agentic_trading_system.execution.broker_connectors.mock_broker import MockBroker
from agentic_trading_system.execution.routing.smart_order_routing import SmartOrderRouting
from agentic_trading_system.execution.routing.venue_analyzer import VenueAnalyzer
from agentic_trading_system.execution.fills_manager import FillsManager
from agentic_trading_system.execution.open_positions import OpenPositions
from agentic_trading_system.execution.settlement import Settlement
from agentic_trading_system.agents.base_agent import AgentMessage
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CompleteExecutionTest:
    """
    Complete sequential test for all Execution module components
    """
    
    def __init__(self):
        self.initial_capital = 100000.0
        self.execution_engine = None
        self.order_manager = None
        self.market_order = None
        self.limit_order = None
        self.stop_order = None
        self.trailing_stop = None
        self.paper_trading = None
        self.mock_broker = None
        self.alpaca_client = None
        self.ibkr_client = None
        self.smart_routing = None
        self.venue_analyzer = None
        self.fills_manager = None
        self.open_positions = None
        self.settlement = None
        
        self.results = []
        self.errors = []
        
        logging.info("✅ Complete Execution Test initialized")
    
    async def setup_all_components(self):
        """Initialize all execution components"""
        print("\n" + "="*70)
        print("🔧 INITIALIZING ALL EXECUTION COMPONENTS")
        print("="*70)
        
        # 1. Order Types
        print("\n📝 Initializing Order Types...")
        self.market_order = MarketOrder({"max_slippage": 0.02})
        self.limit_order = LimitOrder({"default_time_in_force": "DAY"})
        self.stop_order = StopOrder({"default_time_in_force": "GTC"})
        self.trailing_stop = TrailingStopOrder({
            "default_trail_percent": 2.0,
            "trail_type": "PERCENT",
            "activation_percent": 1.0
        })
        print("   ✅ Order Types initialized")
        
        # 2. Broker Connectors
        print("\n🔌 Initializing Broker Connectors...")
        self.paper_trading = PaperTrading({
            "initial_capital": self.initial_capital,
            "fill_delay_seconds": 0.5,
            "slippage_model": "percentage",
            "percentage_slippage": 0.001
        })
        
        self.mock_broker = MockBroker({
            "initial_cash": self.initial_capital,
            "always_succeed": True,
            "fail_rate": 0.0
        })
        
        self.alpaca_client = AlpacaClient({
            "api_key": None,
            "api_secret": None,
            "base_url": "https://paper-api.alpaca.markets"
        })
        
        self.ibkr_client = IBKRClient({
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1
        })
        
        print("   ✅ Broker Connectors initialized")
        
        # 3. Routing Components
        print("\n🔄 Initializing Routing Components...")
        self.venue_analyzer = VenueAnalyzer({
            "venues": {
                "NYSE": {
                    "name": "New York Stock Exchange",
                    "type": "exchange",
                    "latency_ms": 5,
                    "maker_fee": -0.0001,
                    "taker_fee": 0.0002,
                    "liquidity_score": 0.95,
                    "max_order_size": 100000
                },
                "NASDAQ": {
                    "name": "NASDAQ",
                    "type": "exchange",
                    "latency_ms": 4,
                    "maker_fee": -0.0001,
                    "taker_fee": 0.0002,
                    "liquidity_score": 0.94,
                    "max_order_size": 100000
                },
                "IEX": {
                    "name": "IEX",
                    "type": "exchange",
                    "latency_ms": 10,
                    "maker_fee": 0.0,
                    "taker_fee": 0.0,
                    "liquidity_score": 0.70,
                    "max_order_size": 50000
                },
                "DARK_POOL_A": {
                    "name": "Dark Pool A",
                    "type": "dark_pool",
                    "latency_ms": 15,
                    "maker_fee": -0.00005,
                    "taker_fee": 0.00015,
                    "liquidity_score": 0.60,
                    "min_size": 1000,
                    "max_order_size": 25000
                }
            }
        })
        
        self.smart_routing = SmartOrderRouting({
            "venue_config": {},
            "default_strategy": "lowest_cost",
            "max_splits": 5,
            "min_split_size": 100
        })
        self.smart_routing.venue_analyzer = self.venue_analyzer
        
        print("   ✅ Routing Components initialized")
        
        # 4. Tracking Components
        print("\n📊 Initializing Tracking Components...")
        self.fills_manager = FillsManager({})
        self.open_positions = OpenPositions({
            "max_positions": 20,
            "max_position_size": 0.25,
            "max_sector_exposure": 0.30
        })
        self.settlement = Settlement({
            "initial_cash": self.initial_capital,
            "settlement_days": 2,
            "margin_enabled": False
        })
        print("   ✅ Tracking Components initialized")
        
        # 5. Order Manager
        print("\n📋 Initializing Order Manager...")
        self.order_manager = OrderManager({
            "market_config": {},
            "limit_config": {},
            "stop_config": {},
            "trailing_config": {},
            "max_history": 1000
        })
        print("   ✅ Order Manager initialized")
        
        # 6. Execution Engine
        print("\n⚙️ Initializing Execution Engine...")
        self.execution_engine = ExecutionEngine(
            name="CompleteExecutionEngine",
            config={
                "mode": "paper",
                "initial_capital": self.initial_capital,
                "paper_config": {
                    "initial_capital": self.initial_capital,
                    "fill_delay_seconds": 0.5,
                    "slippage_model": "percentage",
                    "percentage_slippage": 0.001
                },
                "order_config": {"max_history": 1000},
                "routing_config": {"default_strategy": "lowest_cost"},
                "fills_config": {},
                "positions_config": {},
                "settlement_config": {}
            }
        )
        
        # Set market open
        self.execution_engine.is_market_open = True
        
        # Link order manager to execution engine
        self.order_manager.set_execution_engine(self.execution_engine)
        self.execution_engine.order_manager = self.order_manager
        
        print("   ✅ Execution Engine initialized\n")
    
    async def test_market_order_creation(self):
        """Test 1: Market Order Creation and Validation"""
        print(f"\n   📈 Test 1: Market Order Creation")
        print("   " + "-" * 40)
        
        # Create market order
        order = self.market_order.create(
            symbol="AAPL",
            quantity=10,
            side="BUY",
            time_in_force="DAY",
            client_id=str(uuid.uuid4())
        )
        
        # Validate
        validation = self.market_order.validate(order)
        
        print(f"      • Order ID: {order['order_id']}")
        print(f"      • Symbol: {order['symbol']}")
        print(f"      • Quantity: {order['quantity']}")
        print(f"      • Side: {order['side']}")
        print(f"      • Type: {order['order_type']}")
        print(f"      • Valid: {validation['valid']}")
        
        # Estimate cost
        estimate = self.market_order.estimate_cost(order, 175.50)
        print(f"      • Estimated Price: ${estimate['estimated_price']:.2f}")
        print(f"      • Slippage Cost: ${estimate['slippage_cost']:.2f}")
        
        return order
    
    async def test_limit_order_creation(self):
        """Test 2: Limit Order Creation and Validation"""
        print(f"\n   📈 Test 2: Limit Order Creation")
        print("   " + "-" * 40)
        
        # Get current price for MSFT
        ticker = yf.Ticker("MSFT")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        limit_price = current_price * 0.98  # 2% below market
        
        # Create limit order
        order = self.limit_order.create(
            symbol="MSFT",
            quantity=5,
            side="BUY",
            limit_price=limit_price,
            time_in_force="DAY"
        )
        
        # Validate
        validation = self.limit_order.validate(order)
        
        print(f"      • Order ID: {order['order_id']}")
        print(f"      • Current Price: ${current_price:.2f}")
        print(f"      • Limit Price: ${limit_price:.2f}")
        print(f"      • Valid: {validation['valid']}")
        
        # Check execution condition
        check = self.limit_order.check_execution(order, current_price)
        print(f"      • Can Execute Now: {check['can_execute']}")
        
        return order
    
    async def test_stop_order_creation(self):
        """Test 3: Stop Order Creation and Validation"""
        print(f"\n   📈 Test 3: Stop Order Creation")
        print("   " + "-" * 40)
        
        # Get current price for NVDA
        ticker = yf.Ticker("NVDA")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        stop_price = current_price * 0.95  # 5% stop loss
        
        # Create stop order
        order = self.stop_order.create(
            symbol="NVDA",
            quantity=5,
            side="SELL",
            stop_price=stop_price,
            stop_type="STOP_LOSS",
            time_in_force="GTC"
        )
        
        # Validate
        validation = self.stop_order.validate(order)
        
        print(f"      • Order ID: {order['order_id']}")
        print(f"      • Current Price: ${current_price:.2f}")
        print(f"      • Stop Price: ${stop_price:.2f}")
        print(f"      • Stop Type: {order['stop_type']}")
        print(f"      • Valid: {validation['valid']}")
        
        # Check trigger condition
        check = self.stop_order.check_trigger(order, current_price)
        print(f"      • Triggered: {check['triggered']}")
        
        return order
    
    async def test_trailing_stop_creation(self):
        """Test 4: Trailing Stop Creation"""
        print(f"\n   📈 Test 4: Trailing Stop Creation")
        print("   " + "-" * 40)
        
        # Get current price for GOOGL
        ticker = yf.Ticker("GOOGL")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Create trailing stop
        order = self.trailing_stop.create(
            symbol="GOOGL",
            quantity=3,
            side="SELL",
            entry_price=current_price,
            trail_value=5.0,
            trail_type="PERCENT",
            activation_percent=2.0
        )
        
        print(f"      • Order ID: {order['order_id']}")
        print(f"      • Entry Price: ${order['entry_price']:.2f}")
        print(f"      • Trail Type: {order['trail_type']}")
        print(f"      • Trail Value: {order['trail_value']}")
        print(f"      • Status: {order['status']}")
        
        # Simulate price increase
        simulated_price = current_price * 1.03  # 3% up
        updated = self.trailing_stop.update(order, simulated_price)
        
        print(f"      • Simulated Price: ${simulated_price:.2f}")
        print(f"      • Activated: {updated['activated']}")
        if updated.get('current_stop'):
            print(f"      • Current Stop: ${updated['current_stop']:.2f}")
        
        return order
    
    async def test_broker_connectors(self):
        """Test 5: All Broker Connectors"""
        print(f"\n   📈 Test 5: Broker Connectors")
        print("   " + "-" * 40)
        
        # Test Paper Trading
        print("\n      📝 Paper Trading:")
        account = await self.paper_trading.get_account()
        print(f"         • Account: {account.get('account_id')}")
        print(f"         • Cash: ${account.get('cash', 0):,.2f}")
        print(f"         • Portfolio: ${account.get('portfolio_value', 0):,.2f}")
        
        # Submit paper trade
        order = {
            "symbol": "AAPL",
            "quantity": 5,
            "side": "BUY",
            "order_type": "MARKET"
        }
        result = await self.paper_trading.submit_order(order)
        print(f"         • Trade Executed: {result.get('success')}")
        if result.get('filled_price'):
            print(f"         • Fill Price: ${result['filled_price']:.2f}")
        
        # Test Mock Broker
        print("\n      🧪 Mock Broker:")
        mock_account = await self.mock_broker.get_account()
        print(f"         • Account: {mock_account.get('account_id')}")
        print(f"         • Cash: ${mock_account.get('cash', 0):,.2f}")
        
        # Test Alpaca (if configured)
        print("\n      🔷 Alpaca Client:")
        if self.alpaca_client.is_configured:
            alpaca_account = await self.alpaca_client.get_account()
            print(f"         • Status: {alpaca_account.get('status')}")
            print(f"         • Cash: ${alpaca_account.get('cash', 0):,.2f}")
        else:
            print("         • Not configured (credentials missing)")
        
        # Test IBKR
        print("\n      🔶 IBKR Client:")
        ibkr_account = await self.ibkr_client.get_account()
        print(f"         • Connected: {ibkr_account.get('connected')}")
        print(f"         • Account: {ibkr_account.get('account_id')}")
        
        return {
            "paper_account": account,
            "mock_account": mock_account,
            "alpaca_configured": self.alpaca_client.is_configured,
            "ibkr_connected": ibkr_account.get('connected')
        }
    
    async def test_smart_routing(self):
        """Test 6: Smart Order Routing"""
        print(f"\n   📈 Test 6: Smart Order Routing")
        print("   " + "-" * 40)
        
        # Test order
        order = {
            "symbol": "AAPL",
            "quantity": 1000,
            "side": "BUY",
            "order_type": "MARKET"
        }
        
        # Test different routing strategies
        strategies = ["best_price", "lowest_cost", "fastest", "dark_pool_first", "lit_only"]
        
        results = {}
        for strategy in strategies:
            print(f"\n      🎯 Strategy: {strategy}")
            routing_plan = await self.smart_routing.route_order(order, strategy)
            results[strategy] = routing_plan
            
            print(f"         • Type: {routing_plan.get('type')}")
            
            if routing_plan.get('type') == 'single':
                venue = routing_plan.get('venue')
                print(f"         • Venue: {venue if venue else 'Fallback'}")
                print(f"         • Quantity: {routing_plan.get('quantity')}")
            else:
                print(f"         • Splits: {len(routing_plan.get('splits', []))}")
                for split in routing_plan.get('splits', [])[:3]:
                    print(f"            - {split['venue']}: {split['quantity']} shares")
        
        return results
    
    async def test_venue_analyzer(self):
        """Test 7: Venue Analyzer"""
        print(f"\n   📈 Test 7: Venue Analyzer")
        print("   " + "-" * 40)
        
        # Test order
        order = {
            "symbol": "AAPL",
            "quantity": 500,
            "side": "BUY"
        }
        
        # Get best venue
        best = self.venue_analyzer.get_best_venue(order)
        
        print(f"      • Best Venue: {best['best_venue']}")
        
        # Show all venues with scores
        print(f"\n      📊 Venue Scores:")
        for venue in best['all_venues'][:4]:
            print(f"         • {venue['venue']}: Score={venue['score']:.1f}")
            cost = venue['estimated_cost']
            print(f"           Fees: ${cost['fees']:.2f}, Slippage: ${cost['expected_slippage']:.2f}")
        
        # Get venue stats
        stats = self.venue_analyzer.get_venue_stats()
        print(f"\n      • Best Overall: {stats['best_overall']}")
        
        # Update performance
        self.venue_analyzer.update_performance("NYSE", True, 4.5, 0.001)
        self.venue_analyzer.update_performance("NASDAQ", True, 3.8, 0.0008)
        
        print(f"      • Performance updated for NYSE and NASDAQ")
        
        return best
    
    async def test_fills_manager(self):
        """Test 8: Fills Manager"""
        print(f"\n   📈 Test 8: Fills Manager")
        print("   " + "-" * 40)
        
        # Add some fills
        fill1 = self.fills_manager.add_fill({
            "order_id": "order_001",
            "symbol": "AAPL",
            "quantity": 10,
            "price": 175.50,
            "value": 1755.00,
            "broker_order_id": "broker_001",
            "venue": "NASDAQ"
        })
        
        fill2 = self.fills_manager.add_fill({
            "order_id": "order_001",
            "symbol": "AAPL",
            "quantity": 5,
            "price": 176.00,
            "value": 880.00,
            "broker_order_id": "broker_001"
        })
        
        fill3 = self.fills_manager.add_fill({
            "order_id": "order_002",
            "symbol": "MSFT",
            "quantity": 8,
            "price": 420.00,
            "value": 3360.00,
            "broker_order_id": "broker_002"
        })
        
        print(f"      • Fills Added: 3")
        
        # Get average price for order
        avg_price = self.fills_manager.get_average_fill_price("order_001")
        print(f"      • Order 001 Avg Price: ${avg_price:.2f}")
        
        # Get total filled quantity
        filled_qty = self.fills_manager.get_total_filled_quantity("order_001")
        print(f"      • Order 001 Filled: {filled_qty} shares")
        
        # Check if order is complete
        is_complete = self.fills_manager.is_order_complete("order_001", 15)
        print(f"      • Order 001 Complete: {is_complete}")
        
        # Get recent fills
        recent = self.fills_manager.get_recent_fills(5)
        print(f"      • Recent Fills: {len(recent)}")
        
        # Get stats
        stats = self.fills_manager.get_stats()
        print(f"      • Total Volume: {stats['total_volume']} shares")
        print(f"      • Total Value: ${stats['total_value']:,.2f}")
        
        return stats
    
    async def test_open_positions(self):
        """Test 9: Open Positions Management"""
        print(f"\n   📈 Test 9: Open Positions Management")
        print("   " + "-" * 40)
        
        # Add positions
        self.open_positions.add_position({
            "symbol": "AAPL",
            "shares": 15,
            "avg_price": 175.75,
            "cost_basis": 2636.25,
            "sector": "Technology",
            "entry_date": datetime.now().isoformat()
        })
        
        self.open_positions.add_position({
            "symbol": "MSFT",
            "shares": 8,
            "avg_price": 420.00,
            "cost_basis": 3360.00,
            "sector": "Technology",
            "entry_date": datetime.now().isoformat()
        })
        
        self.open_positions.add_position({
            "symbol": "JPM",
            "shares": 20,
            "avg_price": 150.00,
            "cost_basis": 3000.00,
            "sector": "Financial",
            "entry_date": datetime.now().isoformat()
        })
        
        print(f"      • Positions Added: 3")
        
        # Get all positions
        all_positions = self.open_positions.get_all_positions()
        print(f"      • Total Positions: {len(all_positions)}")
        
        # Get position details
        aapl_pos = self.open_positions.get_position("AAPL")
        print(f"      • AAPL: {aapl_pos['shares']} shares @ ${aapl_pos['avg_price']:.2f}")
        
        # Update prices (simulate)
        self.open_positions.positions["AAPL"]["current_price"] = 180.00
        self.open_positions.positions["AAPL"]["market_value"] = 15 * 180.00
        self.open_positions.positions["AAPL"]["unrealized_pl"] = (15 * 180.00) - 2636.25
        self.open_positions.positions["AAPL"]["unrealized_plpc"] = (63.75 / 2636.25) * 100
        
        print(f"      • AAPL P&L: ${self.open_positions.positions['AAPL']['unrealized_pl']:.2f}")
        
        # Get sector exposure
        sector_exp = self.open_positions.get_sector_exposure()
        print(f"      • Sector Exposure:")
        for sector, pct in sector_exp.items():
            print(f"         - {sector}: {pct:.1f}%")
        
        # Check limits
        limits_check = self.open_positions.check_limits()
        print(f"      • Within Limits: {limits_check['within_limits']}")
        
        # Generate report
        report = self.open_positions.generate_report()
        print(f"      • Total Value: ${report['total_value']:,.2f}")
        print(f"      • Total P&L: ${report['total_unrealized_pl']:+.2f}")
        
        return report
    
    async def test_settlement(self):
        """Test 10: Settlement and Cash Management"""
        print(f"\n   📈 Test 10: Settlement Management")
        print("   " + "-" * 40)
        
        # Process a trade
        trade1 = self.settlement.process_trade({
            "order_id": "trade_001",
            "symbol": "AAPL",
            "quantity": 15,
            "price": 175.75,
            "side": "BUY"
        })
        
        print(f"      • Trade 1: BUY 15 AAPL @ $175.75")
        print(f"         • Trade Value: ${trade1['trade_value']:,.2f}")
        print(f"         • Settlement Date: {trade1['settlement_date']}")
        
        # Process a sell
        trade2 = self.settlement.process_trade({
            "order_id": "trade_002",
            "symbol": "MSFT",
            "quantity": 5,
            "price": 425.00,
            "side": "SELL"
        })
        
        print(f"      • Trade 2: SELL 5 MSFT @ $425.00")
        print(f"         • Trade Value: ${trade2['trade_value']:,.2f}")
        
        # Process a dividend
        dividend = self.settlement.process_dividend("AAPL", 0.24, 15)
        print(f"      • Dividend: ${dividend['amount']:.2f} from AAPL")
        
        # Process a fee
        fee = self.settlement.process_fee(1.50, "Commission")
        print(f"      • Fee: ${fee['amount']:.2f}")
        
        # Get cash summary
        cash_summary = self.settlement.get_cash_summary()
        print(f"\n      💰 Cash Summary:")
        print(f"         • Cash Balance: ${cash_summary['cash_balance']:,.2f}")
        print(f"         • Pending Settlements: {cash_summary['pending_settlements']}")
        print(f"         • Projected Cash: ${cash_summary['projected_cash']:,.2f}")
        print(f"         • Available Cash: ${cash_summary['available_cash']:,.2f}")
        
        # Get settlement schedule
        schedule = self.settlement.get_settlement_schedule()
        print(f"\n      📅 Settlement Schedule:")
        print(f"         • Total Pending: {schedule['total_pending']}")
        print(f"         • Net Settlement: ${schedule['net_settlement']:,.2f}")
        
        # Settle pending
        settled = self.settlement.settle_pending()
        if settled:
            print(f"      • Settled: {len(settled)} transactions")
        
        # Get transaction history
        history = self.settlement.get_transaction_history(limit=10)
        print(f"      • Recent Transactions: {len(history)}")
        
        return cash_summary
    
    async def test_order_manager(self):
        """Test 11: Order Manager"""
        print(f"\n   📈 Test 11: Order Manager")
        print("   " + "-" * 40)
        
        # Create orders
        market_order = self.order_manager.create_order({
            "symbol": "AAPL",
            "quantity": 10,
            "side": "BUY",
            "order_type": "MARKET"
        })
        
        limit_order = self.order_manager.create_order({
            "symbol": "MSFT",
            "quantity": 5,
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": 415.00
        })
        
        stop_order = self.order_manager.create_order({
            "symbol": "NVDA",
            "quantity": 3,
            "side": "SELL",
            "order_type": "STOP",
            "stop_price": 135.00
        })
        
        print(f"      • Orders Created: 3")
        print(f"         - Market: {market_order['order_id']}")
        print(f"         - Limit: {limit_order['order_id']}")
        print(f"         - Stop: {stop_order['order_id']}")
        
        # Get order by ID
        retrieved = self.order_manager.get_order(market_order['order_id'])
        print(f"      • Retrieved Order: {retrieved['order_type']} {retrieved['symbol']}")
        
        # Modify limit order
        modified = self.order_manager.modify_order(limit_order['order_id'], {
            "limit_price": 418.00
        })
        if modified:
            print(f"      • Modified Limit Order: ${modified['limit_price']:.2f}")
        
        # Get active orders
        active = self.order_manager.get_active_orders()
        print(f"      • Active Orders: {len(active)}")
        
        # Get order summary
        summary = self.order_manager.get_order_summary()
        print(f"      • Total Orders: {summary['total_orders']}")
        print(f"      • Status Breakdown: {summary['status_breakdown']}")
        
        return summary
    
    async def test_execution_engine_orders(self):
        """Test 12: Execution Engine Order Submission"""
        print(f"\n   📈 Test 12: Execution Engine Order Submission")
        print("   " + "-" * 40)
        
        # 1. Create and submit market order through OrderManager
        print("\n      🚀 Market Order:")
        market_order = self.order_manager.create_order({
            "symbol": "AAPL",
            "quantity": 10,
            "side": "BUY",
            "order_type": "MARKET"
        })
        
        print(f"         • Created Order ID: {market_order['order_id']}")
        
        # Submit the order
        result1 = await self.order_manager.submit_order(market_order['order_id'])
        print(f"         • Submit Result: {result1.get('success', False)}")
        if result1.get('success'):
            order = self.order_manager.get_order(market_order['order_id'])
            print(f"         • Order Status: {order.get('status')}")
            if order.get('filled_price'):
                print(f"         • Fill Price: ${order['filled_price']:.2f}")
        
        # 2. Create and submit limit order
        print("\n      🎯 Limit Order:")
        ticker = yf.Ticker("MSFT")
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        limit_price = current_price * 0.97
        
        limit_order = self.order_manager.create_order({
            "symbol": "MSFT",
            "quantity": 5,
            "side": "BUY",
            "order_type": "LIMIT",
            "limit_price": limit_price
        })
        
        print(f"         • Current Price: ${current_price:.2f}")
        print(f"         • Limit Price: ${limit_price:.2f}")
        print(f"         • Created Order ID: {limit_order['order_id']}")
        
        # Submit limit order
        result2 = await self.order_manager.submit_order(limit_order['order_id'])
        print(f"         • Submit Result: {result2.get('success', False)}")
        
        # Wait a bit for orders to process
        await asyncio.sleep(2)
        
        # 3. Get positions from execution engine
        print("\n      📊 Current Positions:")
        positions = await self.execution_engine.get_positions()
        print(f"         • Total Positions: {positions.get('total_positions', 0)}")
        for pos in positions.get('positions', []):
            print(f"         • {pos['symbol']}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
        
        # 4. Get account summary
        print("\n      💰 Account Summary:")
        summary = await self.execution_engine.get_account_summary()
        print(f"         • Cash: ${summary.get('cash', {}).get('cash_balance', 0):,.2f}")
        print(f"         • Portfolio Value: ${summary.get('broker', {}).get('portfolio_value', 0):,.2f}")
        print(f"         • Positions: {summary.get('positions', {}).get('total_positions', 0)}")
        
        return {
            "market_order": market_order,
            "limit_order": limit_order,
            "positions": positions,
            "summary": summary
        }
    
    async def test_stop_monitoring_simulation(self):
        """Test 13: Stop Monitoring Simulation"""
        print(f"\n   📈 Test 13: Stop Monitoring Simulation")
        print("   " + "-" * 40)
        
        # Create a stop loss order
        stop_order = self.order_manager.create_order({
            "symbol": "AAPL",
            "quantity": 10,
            "side": "SELL",
            "order_type": "STOP",
            "stop_price": 170.00,
            "stop_type": "STOP_LOSS"
        })
        
        print(f"      • Stop Order Created: {stop_order['order_id']}")
        print(f"      • Stop Price: ${stop_order['stop_price']:.2f}")
        
        # Simulate price movement
        current_prices = {
            "AAPL": 168.50  # Below stop
        }
        
        # Check stops
        triggered = self.order_manager.check_stop_orders(current_prices)
        
        print(f"      • Triggered: {len(triggered)}")
        for t in triggered:
            print(f"         - Stop ID: {t['stop_order_id']}")
            print(f"         - Market Order: {t['market_order_id']}")
            print(f"         - Trigger Price: ${t['trigger_price']:.2f}")
        
        # Create trailing stop
        trailing_order = self.trailing_stop.create(
            symbol="AAPL",
            quantity=10,
            side="SELL",
            entry_price=175.00,
            trail_value=5.0,
            trail_type="PERCENT",
            activation_percent=1.0
        )
        
        print(f"\n      • Trailing Stop Created")
        
        # Simulate price increase
        prices = [175.00, 178.00, 182.00, 185.00, 183.00]
        stop_prices = []
        
        for price in prices:
            updated = self.trailing_stop.update(trailing_order, price)
            if updated.get('current_stop'):
                stop_prices.append((price, updated['current_stop']))
        
        print(f"      • Trailing Stop Movement:")
        for price, stop in stop_prices:
            print(f"         Price: ${price:.2f} → Stop: ${stop:.2f}")
        
        return triggered
    
    async def run_all_tests(self):
        """Run all execution tests"""
        print("\n" + "="*70)
        print("🚀 COMPLETE EXECUTION MODULE SEQUENTIAL TEST")
        print("="*70)
        
        # Initialize all components
        await self.setup_all_components()
        
        # Run all tests
        tests = [
            ("Market Order Creation", self.test_market_order_creation),
            ("Limit Order Creation", self.test_limit_order_creation),
            ("Stop Order Creation", self.test_stop_order_creation),
            ("Trailing Stop Creation", self.test_trailing_stop_creation),
            ("Broker Connectors", self.test_broker_connectors),
            ("Smart Order Routing", self.test_smart_routing),
            ("Venue Analyzer", self.test_venue_analyzer),
            ("Fills Manager", self.test_fills_manager),
            ("Open Positions", self.test_open_positions),
            ("Settlement", self.test_settlement),
            ("Order Manager", self.test_order_manager),
            ("Execution Engine Orders", self.test_execution_engine_orders),
            ("Stop Monitoring", self.test_stop_monitoring_simulation)
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"\n{'='*70}")
                print(f"📋 {test_name}")
                print(f"{'='*70}")
                result = await test_func()
                self.results.append({
                    "test": test_name,
                    "status": "PASSED",
                    "result": result
                })
            except Exception as e:
                error_msg = f"Error in {test_name}: {str(e)}"
                print(f"   ❌ {error_msg}")
                import traceback
                traceback.print_exc()
                self.errors.append(error_msg)
                self.results.append({
                    "test": test_name,
                    "status": "FAILED",
                    "error": str(e)
                })
        
        # Print final summary
        self.print_final_summary()
        
        # Save results
        self.save_results()
    
    def print_final_summary(self):
        """Print final summary of all tests"""
        print("\n" + "="*70)
        print("📊 EXECUTION MODULE FINAL SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in self.results if r["status"] == "PASSED")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        
        print(f"\n📈 Test Results:")
        print(f"   • Total Tests: {len(self.results)}")
        print(f"   • Passed: {passed}")
        print(f"   • Failed: {failed}")
        
        if self.errors:
            print(f"\n⚠️ Errors Encountered:")
            for error in self.errors:
                print(f"   • {error}")
        
        # Get final execution engine status
        if self.execution_engine:
            status = self.execution_engine.get_status()
            print(f"\n⚙️ Execution Engine Final Status:")
            print(f"   • Mode: {status.get('mode', 'unknown')}")
            print(f"   • Market Open: {status.get('market_open', False)}")
            print(f"   • Orders Submitted: {status.get('daily_stats', {}).get('orders_submitted', 0)}")
            print(f"   • Orders Filled: {status.get('daily_stats', {}).get('orders_filled', 0)}")
            print(f"   • Active Orders: {status.get('order_summary', {}).get('active_orders', 0)}")
            print(f"   • Open Positions: {status.get('positions_count', 0)}")
            print(f"   • Cash Balance: ${status.get('cash_balance', 0):,.2f}")
        
        # Get fills manager stats
        if self.fills_manager:
            fills_stats = self.fills_manager.get_stats()
            print(f"\n📝 Fills Manager Stats:")
            print(f"   • Total Fills: {fills_stats.get('total_fills', 0)}")
            print(f"   • Total Volume: {fills_stats.get('total_volume', 0)} shares")
            print(f"   • Total Value: ${fills_stats.get('total_value', 0):,.2f}")
        
        # Get settlement cash summary
        if self.settlement:
            cash_summary = self.settlement.get_cash_summary()
            print(f"\n💰 Final Cash Position:")
            print(f"   • Cash Balance: ${cash_summary.get('cash_balance', 0):,.2f}")
            print(f"   • Total P&L: ${cash_summary.get('total_pnl', 0):,.2f}")
            print(f"   • Available Cash: ${cash_summary.get('available_cash', 0):,.2f}")
    
    def save_results(self):
        """Save all results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.results,
            "errors": self.errors,
            "execution_engine_status": self.execution_engine.get_status() if self.execution_engine else None,
            "fills_stats": self.fills_manager.get_stats() if self.fills_manager else None,
            "positions_report": self.open_positions.generate_report() if self.open_positions else None,
            "cash_summary": self.settlement.get_cash_summary() if self.settlement else None,
            "order_summary": self.order_manager.get_order_summary() if self.order_manager else None
        }
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        # Save to file
        output_file = "data/complete_execution_results.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Complete results saved to: {output_file}")


async def main():
    """Main entry point"""
    tester = CompleteExecutionTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())