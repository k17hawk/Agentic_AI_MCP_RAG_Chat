"""
Execution Engine - Main orchestrator for trade execution
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import execution components
from agentic_trading_system.execution.order_manager import OrderManager
from agentic_trading_system.execution.broker_connectors.alpaca_client import AlpacaClient
from agentic_trading_system.execution.broker_connectors.ibkr_client import IBKRClient
from agentic_trading_system.execution.broker_connectors.paper_trading import PaperTrading
from agentic_trading_system.execution.broker_connectors.mock_broker import MockBroker
from agentic_trading_system.execution.routing.smart_order_routing import SmartOrderRouting
from agentic_trading_system.execution.fills_manager import FillsManager
from agentic_trading_system.execution.open_positions import OpenPositions
from execution.settlement import Settlement

class ExecutionEngine(BaseAgent):
    """
    Execution Engine - Main orchestrator for trade execution
    
    Responsibilities:
    - Receive trade signals
    - Route orders to appropriate venues
    - Manage order lifecycle
    - Track fills and positions
    - Handle settlements
    - Report execution status
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Trade execution engine",
            config=config
        )
        
        # Initialize components
        self.order_manager = OrderManager(config.get("order_config", {}))
        self.order_manager.set_execution_engine(self)
        
        # Initialize broker connector based on mode
        self.mode = config.get("mode", "paper")  # live, paper, mock
        self.broker = self._init_broker(config)
        
        # Initialize routing
        self.routing = SmartOrderRouting(config.get("routing_config", {}))
        
        # Initialize tracking
        self.fills_manager = FillsManager(config.get("fills_config", {}))
        self.positions = OpenPositions(config.get("positions_config", {}))
        self.settlement = Settlement(config.get("settlement_config", {}))
        
        # Execution state
        self.is_market_open = False
        self.daily_stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "total_volume": 0,
            "total_value": 0.0
        }
        
        # Price cache for stop checks
        self.current_prices = {}
        
        # Start background tasks
        self.stop_check_task = None
        self.settlement_task = None
        self._start_background_tasks()
        
        logging.info(f"✅ ExecutionEngine initialized in {self.mode} mode")
    
    def _init_broker(self, config: Dict) -> Any:
        """Initialize broker connector based on mode"""
        if self.mode == "live":
            # Try Alpaca first, fallback to IBKR
            alpaca = AlpacaClient(config.get("alpaca_config", {}))
            if alpaca.is_configured:
                logging.info("🔌 Connected to Alpaca (live trading)")
                return alpaca
            else:
                logging.info("🔌 Connected to IBKR (live trading)")
                return IBKRClient(config.get("ibkr_config", {}))
        
        elif self.mode == "paper":
            logging.info("📝 Using paper trading")
            return PaperTrading(config.get("paper_config", {}))
        
        else:  # mock
            logging.info("🧪 Using mock broker for testing")
            return MockBroker(config.get("mock_config", {}))
    
    def _start_background_tasks(self):
        """Start background tasks"""
        async def stop_check_loop():
            while True:
                await asyncio.sleep(1)  # Check every second
                await self._check_stop_orders()
        
        async def settlement_loop():
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._process_settlements()
        
        self.stop_check_task = asyncio.create_task(stop_check_loop())
        self.settlement_task = asyncio.create_task(settlement_loop())
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process execution requests
        """
        if message.message_type == "execute_trade":
            # Execute a trade
            trade = message.content
            return await self.execute_trade(trade, message.sender)
        
        elif message.message_type == "cancel_order":
            # Cancel an order
            order_id = message.content.get("order_id")
            result = await self.cancel_order(order_id)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="order_cancelled",
                content=result
            )
        
        elif message.message_type == "get_order_status":
            # Get order status
            order_id = message.content.get("order_id")
            order = self.order_manager.get_order(order_id)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="order_status",
                content=order or {"error": "Order not found"}
            )
        
        elif message.message_type == "get_positions":
            # Get current positions
            positions = await self.get_positions()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="positions_report",
                content=positions
            )
        
        elif message.message_type == "get_account_summary":
            # Get account summary
            summary = await self.get_account_summary()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="account_summary",
                content=summary
            )
        
        elif message.message_type == "update_market_status":
            # Update market open/closed status
            self.is_market_open = message.content.get("is_open", False)
            return None
        
        return None
    
    async def execute_trade(self, trade: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Execute a trade
        """
        symbol = trade.get("symbol")
        action = trade.get("action")
        quantity = trade.get("shares", trade.get("quantity", 0))
        
        logging.info(f"⚡ Executing trade: {action} {quantity} {symbol}")
        
        # Convert action to order side
        if action in ["BUY", "STRONG_BUY"]:
            side = "BUY"
        elif action in ["SELL", "STRONG_SELL"]:
            side = "SELL"
        else:
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="execution_failed",
                content={"error": f"Invalid action: {action}"}
            )
        
        # Create order
        order_data = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "order_type": trade.get("order_type", "MARKET"),
            "limit_price": trade.get("limit_price"),
            "stop_price": trade.get("stop_price"),
            "time_in_force": trade.get("time_in_force", "DAY"),
            "client_order_id": trade.get("client_order_id")
        }
        
        order = self.order_manager.create_order(order_data)
        order_id = order["order_id"]
        
        # Route order
        routing_plan = await self.routing.route_order(order)
        
        # Submit to broker
        if routing_plan["type"] == "single":
            # Single venue execution
            result = await self._submit_to_broker(order, routing_plan["venue"])
        else:
            # Split execution
            result = await self._execute_split_order(order, routing_plan["splits"])
        
        # Process settlement
        if result.get("success"):
            settlement_result = self.settlement.process_trade({
                "order_id": order_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": result.get("avg_price", trade.get("price", 0)),
                "side": side
            })
            
            # Update daily stats
            self.daily_stats["orders_submitted"] += 1
            self.daily_stats["total_volume"] += quantity
            self.daily_stats["total_value"] += result.get("total_value", 0)
            
            if result.get("status") == "filled":
                self.daily_stats["orders_filled"] += 1
        else:
            self.daily_stats["orders_rejected"] += 1
        
        return AgentMessage(
            sender=self.name,
            receiver=requester,
            message_type="execution_result",
            content={
                "order_id": order_id,
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "result": result,
                "settlement": settlement_result if result.get("success") else None,
                "routing_plan": routing_plan
            }
        )
    
    async def _submit_to_broker(self, order: Dict[str, Any], venue: str = None) -> Dict[str, Any]:
        """
        Submit order to broker
        """
        # In production, this would route to specific venue
        result = await self.broker.submit_order(order)
        
        if result.get("success"):
            # Update order with broker info
            self.order_manager.update_order_status(
                order["order_id"],
                "FILLED" if result.get("status") == "filled" else "ACCEPTED",
                {
                    "quantity": order["quantity"],
                    "price": result.get("filled_price", 0),
                    "time": datetime.now().isoformat()
                }
            )
            
            # Record fill
            fill = {
                "order_id": order["order_id"],
                "symbol": order["symbol"],
                "quantity": order["quantity"],
                "price": result.get("filled_price", 0),
                "value": order["quantity"] * result.get("filled_price", 0),
                "broker_order_id": result.get("broker_order_id"),
                "venue": venue
            }
            self.fills_manager.add_fill(fill)
            
            # Update position
            if result.get("side") == "BUY":
                self.positions.add_position({
                    "symbol": order["symbol"],
                    "shares": order["quantity"],
                    "avg_price": result.get("filled_price", 0),
                    "cost_basis": order["quantity"] * result.get("filled_price", 0)
                })
            else:  # SELL
                self.positions.remove_position(order["symbol"], order["quantity"])
        
        return result
    
    async def _execute_split_order(self, order: Dict[str, Any], 
                                  splits: List[Dict]) -> Dict[str, Any]:
        """
        Execute a split order across multiple venues
        """
        results = []
        total_quantity = 0
        total_value = 0
        total_filled = 0
        
        for split in splits:
            # Create sub-order for this split
            sub_order = order.copy()
            sub_order["quantity"] = split["quantity"]
            sub_order["parent_order_id"] = order["order_id"]
            
            # Execute
            result = await self._submit_to_broker(sub_order, split["venue"])
            results.append(result)
            
            if result.get("success"):
                total_quantity += split["quantity"]
                total_value += result.get("total_value", 0)
                total_filled += 1
        
        # Calculate average price
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        return {
            "success": total_filled > 0,
            "total_quantity": total_quantity,
            "total_value": total_value,
            "avg_price": avg_price,
            "splits": results,
            "filled_splits": total_filled,
            "total_splits": len(splits)
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order
        """
        success = await self.order_manager.cancel_order(order_id)
        
        if success:
            # Cancel in broker if submitted
            order = self.order_manager.get_order(order_id)
            if order and "broker_order_id" in order:
                await self.broker.cancel_order(order["broker_order_id"])
        
        return {
            "success": success,
            "order_id": order_id
        }
    
    async def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions
        """
        # Update prices
        await self.positions.update_prices()
        
        return self.positions.generate_report()
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary
        """
        broker_account = await self.broker.get_account()
        positions = await self.get_positions()
        cash_summary = self.settlement.get_cash_summary()
        
        return {
            "broker": broker_account,
            "positions": positions,
            "cash": cash_summary,
            "daily_stats": self.daily_stats,
            "mode": self.mode,
            "market_open": self.is_market_open,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _check_stop_orders(self):
        """Check stop orders against current prices"""
        if not self.is_market_open:
            return
        
        # Update prices for positions
        await self.positions.update_prices()
        
        # Get current prices
        for symbol in self.positions.positions.keys():
            pos = self.positions.get_position(symbol)
            if pos and "current_price" in pos:
                self.current_prices[symbol] = pos["current_price"]
        
        # Check stop orders
        triggered = self.order_manager.check_stop_orders(self.current_prices)
        
        # Execute triggered stops
        for trigger in triggered:
            market_order = trigger["market_order"]
            await self._submit_to_broker(market_order)
    
    async def _process_settlements(self):
        """Process pending settlements"""
        settled = self.settlement.settle_pending()
        
        if settled:
            logging.info(f"💰 Processed {len(settled)} settlements")
    
    async def stop(self):
        """Graceful shutdown"""
        if self.stop_check_task:
            self.stop_check_task.cancel()
        if self.settlement_task:
            self.settlement_task.cancel()
        
        # Cancel all active orders?
        await super().stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution engine status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "mode": self.mode,
            "market_open": self.is_market_open,
            "daily_stats": self.daily_stats,
            "order_summary": self.order_manager.get_order_summary(),
            "positions_count": len(self.positions.positions),
            "cash_balance": self.settlement.cash_balance
        }