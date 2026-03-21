"""
Alert Manager - Main orchestrator for HITL communications
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import all HITL components
from agentic_trading_system.hitl.channels.whatsapp_client import WhatsAppClient

from agentic_trading_system.hitl.message_builder import MessageBuilder
from agentic_trading_system.hitl.response_parser import ResponseParser
from agentic_trading_system.hitl.pending_queue import PendingQueue
from agentic_trading_system.hitl.timeout_manager import TimeoutManager
from agentic_trading_system.hitl.decision_tracker import DecisionTracker
from agentic_trading_system.hitl.feedback_logger import FeedbackLogger


class AlertManager(BaseAgent):
    """
    WhatsApp-only Alert Manager for human approvals
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="WhatsApp-based human approval system",
            config=config
        )
        
        # Initialize WhatsApp only
        self.whatsapp = WhatsAppClient(config.get("whatsapp_config", {}))
        
        # Initialize other components
        self.message_builder = MessageBuilder(config.get("message_config", {}))
        self.response_parser = ResponseParser(config.get("parser_config", {}))
        self.pending_queue = PendingQueue(config.get("queue_config", {}))
        self.timeout_manager = TimeoutManager(config.get("timeout_config", {}))
        self.decision_tracker = DecisionTracker(config.get("decision_config", {}))
        self.feedback_logger = FeedbackLogger(config.get("feedback_config", {}))
        
        # Set timeout callbacks
        self.timeout_manager.set_timeout_callback(self._handle_timeout)
        
        logging.info(f"✅ WhatsApp AlertManager initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process WhatsApp-related messages
        """
        if message.message_type == "send_trade_alert":
            trade = message.content
            return await self.send_trade_alert(trade, message.sender)
        
        elif message.message_type == "human_response":
            response = message.content
            return await self.handle_response(response, message.sender)
        
        elif message.message_type == "check_pending":
            pending = await self.pending_queue.get_pending_count()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="pending_status",
                content={"pending": pending}
            )
        
        return None
    
    async def send_trade_alert(self, trade: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Send a trade alert for WhatsApp approval
        """
        symbol = trade.get("symbol")
        action = trade.get("action")
        
        logging.info(f"📢 Sending WhatsApp alert for {symbol} ({action})")
        
        # Build WhatsApp message
        messages = self.message_builder.build_approval_request(trade)
        
        # Add to pending queue
        item_id = await self.pending_queue.add(trade, priority="high")
        
        # Register timeout
        self.timeout_manager.register_timeout(item_id, data={
            "trade": trade,
            "requester": requester
        })
        
        # Send via WhatsApp
        result = await self.whatsapp.send_approval_request(trade)
        
        if result.get("success"):
            logging.info(f"✅ WhatsApp alert sent for {symbol}")
        else:
            logging.warning(f"⚠️ Failed to send WhatsApp alert: {result.get('error')}")
        
        return AgentMessage(
            sender=self.name,
            receiver=requester,
            message_type="alert_sent",
            content={
                "item_id": item_id,
                "symbol": symbol,
                "sent": result.get("success", False),
                "simulated": result.get("simulated", False),
                "status": "pending_approval"
            }
        )
    
    async def handle_response(self, response: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Handle a WhatsApp response
        """
        message = response.get("message", "")
        
        logging.info(f"📨 Received WhatsApp response: {message}")
        
        # Parse response
        parsed = self.response_parser.parse(message, "whatsapp")
        
        if not parsed["parsed"]:
            # Send help message
            help_text = self.response_parser.get_help_text()
            await self.whatsapp.send_message(self.whatsapp.to_number, help_text)
            
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="response_error",
                content={"error": "Could not parse response", "help": help_text}
            )
        
        # Process based on type
        if parsed["type"] in ["approve", "reject"]:
            results = []
            for symbol in parsed["symbols"]:
                items = await self.pending_queue.get_by_symbol(symbol)
                
                if not items:
                    results.append({
                        "symbol": symbol,
                        "status": "not_found",
                        "message": f"No pending approval found for {symbol}"
                    })
                    continue
                
                for item in items:
                    result = await self._process_decision(
                        item, parsed["type"], requester
                    )
                    results.append(result)
            
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="response_processed",
                content={"results": results}
            )
        
        elif parsed["type"] == "status":
            pending = await self.pending_queue.get_pending_count()
            decisions = self.decision_tracker.get_decision_summary(1)
            
            status_msg = (
                f"📊 *System Status*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏳ Pending approvals: {pending['total']}\n"
                f"📈 Decisions today: {decisions.get('total_decisions', 0)}\n"
                f"✅ Approval rate: {decisions.get('approval_rate', 0)*100:.0f}%"
            )
            
            await self.whatsapp.send_message(self.whatsapp.to_number, status_msg)
            
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="status_response",
                content={"status": pending, "decisions": decisions}
            )
        
        elif parsed["type"] == "help":
            help_text = self.response_parser.get_help_text()
            await self.whatsapp.send_message(self.whatsapp.to_number, help_text)
            
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="help_response",
                content={"help": help_text}
            )
        
        return AgentMessage(
            sender=self.name,
            receiver=requester,
            message_type="response_acknowledged",
            content={"parsed": parsed}
        )
    
    async def _process_decision(self, item: Dict[str, Any], decision: str, 
                                requester: str) -> Dict[str, Any]:
        """
        Process a single decision
        """
        item_id = item["id"]
        trade = item["data"]
        symbol = trade.get("symbol")
        
        # Calculate response time
        created_at = datetime.fromisoformat(item["created_at"])
        response_time = (datetime.now() - created_at).total_seconds()
        
        # Cancel timeout
        await self.timeout_manager.cancel_timeout(item_id)
        
        # Mark as processed
        await self.pending_queue.mark_processed(item_id, decision)
        
        # Track decision
        await self.decision_tracker.track_decision({
            "symbol": symbol,
            "decision": decision,
            "response_time": response_time,
            "confidence": trade.get("confidence"),
            "risk_score": trade.get("risk_score"),
            "action": trade.get("action"),
            "source": "whatsapp"
        })
        
        if decision == "approve":
            # Send confirmation to human
            await self.whatsapp.send_confirmation({
                "symbol": symbol,
                "shares": trade.get("shares", 0),
                "price": trade.get("price", 0)
            })
            
            # Notify execution engine
            await self.send_message(AgentMessage(
                sender=self.name,
                receiver="ExecutionEngine",
                message_type="execute_trade",
                content={**trade, "approved": True, "item_id": item_id}
            ))
            
        elif decision == "reject":
            # Send rejection notification
            await self.whatsapp.send_message(
                self.whatsapp.to_number,
                f"❌ *REJECTED*\n━━━━━━━━━━━━━━━━━━━━━━━━\n📉 {symbol}\nTrade rejected."
            )
            
            # Log feedback
            await self.feedback_logger.log_trigger_feedback(
                trigger_name="human_approval",
                triggered=True,
                outcome="rejected",
                metadata={"symbol": symbol, "trade": trade}
            )
        
        return {
            "symbol": symbol,
            "decision": decision,
            "status": "processed",
            "item_id": item_id
        }
    
    async def _handle_timeout(self, item_id: str, data: Dict[str, Any]):
        """
        Handle a timeout event
        """
        trade = data.get("trade", {})
        symbol = trade.get("symbol")
        
        logging.warning(f"⏰ Timeout for {symbol}")
        
        # Auto-reject the trade
        await self.pending_queue.cancel(item_id, reason="timeout")
        
        # Send timeout notification
        await self.whatsapp.send_timeout_alert(symbol)
        
        # Track decision
        await self.decision_tracker.track_decision({
            "symbol": symbol,
            "decision": "reject",
            "response_time": self.timeout_manager.default_timeout,
            "source": "timeout",
            "notes": "Auto-rejected due to timeout"
        })
        
        # Log feedback
        await self.feedback_logger.log_trigger_feedback(
            trigger_name="timeout",
            triggered=True,
            outcome="rejected",
            metadata={"symbol": symbol, "trade": trade}
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "pending_queue": self.pending_queue.get_stats(),
            "whatsapp_configured": bool(self.whatsapp.client),
            "decision_stats": self.decision_tracker.get_stats()
        }