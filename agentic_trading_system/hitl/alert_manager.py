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
from agentic_trading_system.hitl.channels.email_client import EmailClient
from agentic_trading_system.hitl.channels.sms_client import SMSClient
from agentic_trading_system.hitl.channels.dashboard_notifier import DashboardNotifier
from agentic_trading_system.hitl.message_builder import MessageBuilder
from agentic_trading_system.hitl.response_parser import ResponseParser
from agentic_trading_system.hitl.pending_queue import PendingQueue
from agentic_trading_system.hitl.timeout_manager import TimeoutManager
from agentic_trading_system.hitl.decision_tracker import DecisionTracker
from agentic_trading_system.hitl.feedback_logger import FeedbackLogger

class AlertManager(BaseAgent):
    """
    Alert Manager - Main orchestrator for HITL communications
    
    Responsibilities:
    - Send alerts via multiple channels
    - Manage pending approvals
    - Handle timeouts
    - Parse responses
    - Track decisions
    - Log feedback
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Human-in-the-Loop alert and approval management",
            config=config
        )
        
        # Initialize channels
        self.whatsapp = WhatsAppClient(config.get("whatsapp_config", {}))
        self.email = EmailClient(config.get("email_config", {}))
        self.sms = SMSClient(config.get("sms_config", {}))
        self.dashboard = DashboardNotifier(config.get("dashboard_config", {}))
        
        # Initialize components
        self.message_builder = MessageBuilder(config.get("message_config", {}))
        self.response_parser = ResponseParser(config.get("parser_config", {}))
        self.pending_queue = PendingQueue(config.get("queue_config", {}))
        self.timeout_manager = TimeoutManager(config.get("timeout_config", {}))
        self.decision_tracker = DecisionTracker(config.get("decision_config", {}))
        self.feedback_logger = FeedbackLogger(config.get("feedback_config", {}))
        
        # Set timeout callbacks
        self.timeout_manager.set_timeout_callback(self._handle_timeout)
        self.timeout_manager.set_reminder_callback(self._send_reminder)
        
        # Default channel preferences
        self.channels = config.get("channels", {
            "trade_alert": ["whatsapp", "dashboard"],
            "urgent": ["sms", "whatsapp"],
            "daily_summary": ["email", "dashboard"],
            "system_alert": ["whatsapp", "email", "dashboard"]
        })
        
        logging.info(f"✅ AlertManager initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process alert-related messages
        """
        if message.message_type == "send_trade_alert":
            # Send trade alert for approval
            trade = message.content
            return await self.send_trade_alert(trade, message.sender)
        
        elif message.message_type == "human_response":
            # Handle human response
            response = message.content
            return await self.handle_response(response, message.sender)
        
        elif message.message_type == "send_notification":
            # Send general notification
            notification = message.content
            return await self.send_notification(notification)
        
        elif message.message_type == "check_pending":
            # Check pending approvals
            pending = await self.pending_queue.get_pending_count()
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="pending_status",
                content={"pending": pending}
            )
        
        elif message.message_type == "get_decisions":
            # Get decision history
            days = message.content.get("days", 7)
            summary = self.decision_tracker.get_decision_summary(days)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="decision_summary",
                content=summary
            )
        
        return None
    
    async def send_trade_alert(self, trade: Dict[str, Any], requester: str) -> AgentMessage:
        """
        Send a trade alert for human approval
        """
        symbol = trade.get("symbol")
        action = trade.get("action")
        
        logging.info(f"📢 Sending trade alert for {symbol} ({action})")
        
        # Build message
        messages = self.message_builder.build_approval_request(trade)
        
        # Add to pending queue
        item_id = await self.pending_queue.add(trade, priority="high")
        
        # Register timeout
        self.timeout_manager.register_timeout(item_id, data={
            "trade": trade,
            "requester": requester,
            "channels": self.channels["trade_alert"]
        })
        
        # Send via configured channels
        sent_to = []
        
        for channel in self.channels["trade_alert"]:
            if channel == "whatsapp" and self.whatsapp.client:
                result = await self.whatsapp.send_message(
                    self.whatsapp.to_number,
                    messages["whatsapp"]
                )
                if result.get("success"):
                    sent_to.append("whatsapp")
            
            elif channel == "dashboard":
                await self.dashboard.push_trade_signal(trade)
                sent_to.append("dashboard")
            
            elif channel == "email":
                result = await self.email.send_email(
                    self.email.to_addresses[0] if self.email.to_addresses else None,
                    f"Trade Alert: {action} {symbol}",
                    messages["email"],
                    html=messages.get("email_html", False)
                )
                if result.get("success"):
                    sent_to.append("email")
        
        return AgentMessage(
            sender=self.name,
            receiver=requester,
            message_type="alert_sent",
            content={
                "item_id": item_id,
                "symbol": symbol,
                "sent_to": sent_to,
                "status": "pending_approval"
            }
        )
    
    async def handle_response(self, response: Dict[str, Any], 
                             requester: str) -> AgentMessage:
        """
        Handle a human response
        """
        message = response.get("message", "")
        channel = response.get("channel", "whatsapp")
        
        logging.info(f"📨 Received response via {channel}: {message}")
        
        # Parse response
        parsed = self.response_parser.parse(message, channel)
        
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
        
        # Handle based on type
        if parsed["type"] in ["approve", "reject", "modify"]:
            # Get pending items for symbols
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
                
                # Process each pending item
                for item in items:
                    result = await self._process_decision(
                        item, parsed["type"], parsed.get("quantity"), requester
                    )
                    results.append(result)
            
            return AgentMessage(
                sender=self.name,
                receiver=requester,
                message_type="response_processed",
                content={"results": results}
            )
        
        elif parsed["type"] == "status":
            # Get queue status
            pending = await self.pending_queue.get_pending_count()
            decisions = self.decision_tracker.get_decision_summary(1)
            
            status_msg = (
                f"📊 System Status:\n"
                f"• Pending approvals: {pending['total']}\n"
                f"• Decisions today: {decisions.get('total_decisions', 0)}\n"
                f"• Approval rate: {decisions.get('approval_rate', 0)*100:.1f}%"
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
    
    async def send_notification(self, notification: Dict[str, Any]) -> AgentMessage:
        """
        Send a general notification
        """
        ntype = notification.get("type", "general")
        message = notification.get("message", "")
        data = notification.get("data", {})
        
        # Build message for each channel
        if ntype == "execution":
            messages = self.message_builder.build_execution_confirmation(data)
        elif ntype == "rejection":
            messages = self.message_builder.build_rejection_notification(data)
        elif ntype == "timeout":
            messages = self.message_builder.build_timeout_alert(data.get("symbol"))
        elif ntype == "daily_summary":
            messages = self.message_builder.build_daily_summary(data)
        else:
            messages = {"whatsapp": message, "dashboard": data}
        
        # Send via configured channels
        channels = self.channels.get(ntype, ["dashboard"])
        sent_to = []
        
        for channel in channels:
            if channel == "whatsapp" and self.whatsapp.client and "whatsapp" in messages:
                await self.whatsapp.send_message(
                    self.whatsapp.to_number,
                    messages["whatsapp"]
                )
                sent_to.append("whatsapp")
            
            elif channel == "dashboard":
                await self.dashboard.push_notification(ntype, data)
                sent_to.append("dashboard")
            
            elif channel == "email" and "email" in messages:
                await self.email.send_email(
                    self.email.to_addresses[0] if self.email.to_addresses else None,
                    f"Notification: {ntype}",
                    messages.get("email", message)
                )
                sent_to.append("email")
            
            elif channel == "sms" and self.sms.client and "sms" in messages:
                await self.sms.send_sms(self.sms.to_number, messages["sms"])
                sent_to.append("sms")
        
        return AgentMessage(
            sender=self.name,
            receiver=notification.get("requester", "system"),
            message_type="notification_sent",
            content={"type": ntype, "sent_to": sent_to}
        )
    
    async def _process_decision(self, item: Dict[str, Any], decision: str,
                                quantity: Optional[int], requester: str) -> Dict[str, Any]:
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
            "source": "whatsapp",
            "notes": f"Quantity: {quantity}" if quantity else None
        })
        
        # Send confirmation
        if decision == "approve":
            # Notify execution engine
            await self.send_message(AgentMessage(
                sender=self.name,
                receiver="ExecutionEngine",
                message_type="execute_trade",
                content={**trade, "approved": True, "item_id": item_id}
            ))
            
            # Send confirmation to human
            await self.send_notification({
                "type": "execution",
                "data": {
                    "symbol": symbol,
                    "shares": quantity or trade.get("shares", 0),
                    "price": trade.get("price", 0),
                    "total": (quantity or trade.get("shares", 0)) * trade.get("price", 0)
                }
            })
            
        elif decision == "reject":
            # Send rejection notification
            await self.send_notification({
                "type": "rejection",
                "data": {
                    "symbol": symbol,
                    "reason": "Rejected by human"
                }
            })
            
            # Log feedback
            await self.feedback_logger.log_trigger_feedback(
                trigger_name="human_approval",
                triggered=True,
                outcome="rejected",
                metadata={"symbol": symbol, "trade": trade}
            )
        
        elif decision == "modify" and quantity:
            # Modify quantity and approve
            trade["shares"] = quantity
            trade["position_value"] = quantity * trade.get("price", 0)
            
            await self.send_message(AgentMessage(
                sender=self.name,
                receiver="ExecutionEngine",
                message_type="execute_trade",
                content={**trade, "approved": True, "modified": True, "item_id": item_id}
            ))
        
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
        
        logging.warning(f"⏰ Timeout for {symbol} (ID: {item_id})")
        
        # Auto-reject the trade
        await self.pending_queue.cancel(item_id, reason="timeout")
        
        # Send timeout notification
        await self.send_notification({
            "type": "timeout",
            "data": {"symbol": symbol}
        })
        
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
    
    async def _send_reminder(self, data: Dict[str, Any]):
        """
        Send a reminder for pending approval
        """
        item_id = data["item_id"]
        reminder_count = data["reminder_count"]
        trade_data = data["data"].get("trade", {})
        symbol = trade_data.get("symbol")
        
        reminder_msg = (
            f"🔔 *Reminder ({reminder_count}/{data['max_reminders']})*\n"
            f"Still waiting for your decision on {symbol}.\n"
            f"Reply YES {symbol} or NO {symbol}"
        )
        
        await self.whatsapp.send_message(self.whatsapp.to_number, reminder_msg)
        logging.info(f"🔔 Sent reminder {reminder_count} for {item_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert manager status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "queue_size": self.message_queue.qsize(),
            "pending_queue": self.pending_queue.get_stats(),
            "channels": {
                "whatsapp": bool(self.whatsapp.client),
                "email": bool(self.email.username),
                "sms": bool(self.sms.client),
                "dashboard": True
            }
        }