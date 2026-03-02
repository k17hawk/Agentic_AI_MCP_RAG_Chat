"""
WhatsApp Client - Send WhatsApp messages via Twilio
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from twilio.rest import Client
from utils.logger import logger as logging

class WhatsAppClient:
    """
    WhatsApp Client using Twilio API
    
    Uses credentials from environment:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_WHATSAPP_NUM
    - YOUR_WHATSAPP_NUM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get credentials from environment or config
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID") or config.get("account_sid")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN") or config.get("auth_token")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUM") or config.get("from_number")
        self.to_number = os.getenv("YOUR_WHATSAPP_NUM") or config.get("to_number")
        
        # Initialize Twilio client
        self.client = None
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logging.info("✅ WhatsAppClient initialized with Twilio")
        else:
            logging.warning("⚠️ Twilio credentials not found - WhatsApp client disabled")
        
        # Message queue for rate limiting
        self.message_history = []
        self.rate_limit = config.get("rate_limit", 10)  # messages per minute
        
    async def send_message(self, to: str, message: str) -> Dict[str, Any]:
        """
        Send a WhatsApp message
        """
        if not self.client:
            logging.warning(f"WhatsApp client not configured - would send: {message[:50]}...")
            return {
                "success": False,
                "simulated": True,
                "message": message[:50],
                "to": to
            }
        
        try:
            # Check rate limit
            if not self._check_rate_limit():
                logging.warning("Rate limit exceeded - message queued")
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "queued": True
                }
            
            # Send message
            msg = self.client.messages.create(
                from_=f'whatsapp:{self.from_number}',
                body=message,
                to=f'whatsapp:{to}'
            )
            
            # Log success
            self.message_history.append({
                "timestamp": datetime.now(),
                "to": to,
                "sid": msg.sid,
                "status": msg.status
            })
            
            logging.info(f"✅ WhatsApp sent to {to}: {msg.sid}")
            
            return {
                "success": True,
                "sid": msg.sid,
                "status": msg.status,
                "to": to,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"❌ WhatsApp send error: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to
            }
    
    async def send_template(self, to: str, template_name: str, 
                           variables: Dict[str, str]) -> Dict[str, Any]:
        """
        Send a template message with variables
        """
        # Build message from template
        message = self._build_template(template_name, variables)
        return await self.send_message(to, message)
    
    async def send_bulk(self, recipients: List[str], message: str) -> List[Dict[str, Any]]:
        """
        Send same message to multiple recipients
        """
        results = []
        for recipient in recipients:
            result = await self.send_message(recipient, message)
            results.append(result)
        return results
    
    async def check_status(self, message_sid: str) -> Dict[str, Any]:
        """
        Check status of a sent message
        """
        if not self.client:
            return {"status": "unknown", "simulated": True}
        
        try:
            message = self.client.messages(message_sid).fetch()
            return {
                "sid": message.sid,
                "status": message.status,
                "to": message.to,
                "date_sent": message.date_sent.isoformat() if message.date_sent else None,
                "error_code": message.error_code,
                "error_message": message.error_message
            }
        except Exception as e:
            logging.error(f"Error checking message status: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Remove messages older than 1 minute
        self.message_history = [
            m for m in self.message_history 
            if (now - m["timestamp"]).total_seconds() < 60
        ]
        
        return len(self.message_history) < self.rate_limit
    
    def _build_template(self, template_name: str, variables: Dict[str, str]) -> str:
        """Build message from template"""
        templates = {
            "trade_approval": (
                "🤖 *TRADE SIGNAL*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📈 *{symbol}* - {action}\n"
                "💰 Price: ${price}\n"
                "📊 Confidence: {confidence}%\n"
                "🎯 Target: ${target}\n"
                "🛑 Stop: ${stop}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Reply *YES {symbol}* to approve\n"
                "Reply *NO {symbol}* to reject"
            ),
            "trade_confirmation": (
                "✅ *TRADE EXECUTED*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📈 {symbol}\n"
                "💰 {shares} shares @ ${price}\n"
                "💵 Total: ${total}\n"
                "🕐 {time}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            "alert": (
                "🔔 *SYSTEM ALERT*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "{message}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━"
            ),
            "daily_summary": (
                "📊 *DAILY SUMMARY*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📈 Trades: {trades}\n"
                "💰 P&L: {pnl}\n"
                "✅ Approved: {approved}\n"
                "❌ Rejected: {rejected}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        }
        
        template = templates.get(template_name, "{message}")
        return template.format(**variables)