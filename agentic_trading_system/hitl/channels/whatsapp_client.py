"""
WhatsApp Client - Send WhatsApp messages via Twilio
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from twilio.rest import Client
from agentic_trading_system.utils.logger import logging
from dotenv import load_dotenv
load_dotenv()

class WhatsAppClient:
    """
    WhatsApp Client using Twilio API
    
    Uses credentials from environment or config:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_WHATSAPP_NUM
    - YOUR_WHATSAPP_NUM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get credentials from environment or config
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID") 
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN") 
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUM")
        self.to_number = os.getenv("YOUR_WHATSAPP_NUM")

        print(f"[WhatsAppClient] from_number: {self.from_number}")
        print(f"[WhatsAppClient] to_number: {self.to_number}")
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logging.info("✅ WhatsAppClient initialized with Twilio")
        else:
            logging.warning("⚠️ Twilio credentials not found - WhatsApp client disabled")

        
        # Initialize Twilio client
        self.client = None
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logging.info("✅ WhatsAppClient initialized with Twilio")
        else:
            logging.warning("⚠️ Twilio credentials not found - WhatsApp client disabled")
        
        # Message history for rate limiting
        self.message_history = []
        self.rate_limit = config.get("rate_limit", 10)  # messages per minute
        
        logging.info(f"📱 WhatsAppClient initialized (to: {self.to_number})")
    
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
    
    async def send_approval_request(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a trade approval request
        """
        message = self._build_approval_message(trade)
        return await self.send_message(self.to_number, message)
    
    async def send_confirmation(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send trade execution confirmation
        """
        message = self._build_confirmation_message(execution)
        return await self.send_message(self.to_number, message)
    
    async def send_timeout_alert(self, symbol: str) -> Dict[str, Any]:
        """
        Send timeout alert
        """
        message = f"⏰ TIMEOUT: No response for {symbol}. Auto-rejected."
        return await self.send_message(self.to_number, message)
    
    def _build_approval_message(self, trade: Dict[str, Any]) -> str:
        """
        Build WhatsApp approval request message
        """
        symbol = trade.get("symbol", "UNKNOWN")
        action = trade.get("action", "BUY")
        price = trade.get("price", 0)
        confidence = trade.get("confidence", 0.5) * 100
        shares = trade.get("shares", 0)
        total = trade.get("position_value", shares * price)
        stop = trade.get("stop_loss", price * 0.95)
        target = trade.get("take_profit", price * 1.1)
        rr_ratio = trade.get("rr_ratio", 2.0)
        
        # Get reasons (top 3)
        reasons = trade.get("reasons", [])
        reasons_text = "\n".join([f"   ✅ {r}" for r in reasons[:3]]) if reasons else "   ✅ Strong technical signal"
        
        # Get concerns (top 3)
        concerns = trade.get("concerns", [])
        concerns_text = "\n".join([f"   ⚠️ {c}" for c in concerns[:3]]) if concerns else "   ⚠️ None"
        
        message = (
            f"🤖 *TRADE SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *{action}: {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 *Price*        : ${price:.2f}\n"
            f"🎯 *Confidence*   : {confidence:.0f}%\n"
            f"📊 *Risk Level*   : {trade.get('risk_level', 'MEDIUM')}\n\n"
            f"🛑 *Stop Loss*    : ${stop:.2f}\n"
            f"🎯 *Take Profit*  : ${target:.2f}\n"
            f"📐 *R/R Ratio*    : 1:{rr_ratio:.1f}\n\n"
            f"📦 *Position*     : {shares} shares\n"
            f"💵 *Total Cost*   : ${total:,.2f}\n\n"
            f"*Why {action}:*\n{reasons_text}\n\n"
            f"*Concerns:*\n{concerns_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Reply *YES {symbol}* to approve\n"
            f"Reply *NO {symbol}* to reject"
        )
        
        return message
    
    def _build_confirmation_message(self, execution: Dict[str, Any]) -> str:
        """
        Build WhatsApp execution confirmation message
        """
        symbol = execution.get("symbol", "UNKNOWN")
        shares = execution.get("shares", 0)
        price = execution.get("price", 0)
        total = shares * price
        time = datetime.now().strftime("%H:%M:%S")
        
        message = (
            f"✅ *TRADE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 {symbol}\n"
            f"💰 {shares} shares @ ${price:.2f}\n"
            f"💵 Total: ${total:,.2f}\n"
            f"🕐 {time}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        return message
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Remove messages older than 1 minute
        self.message_history = [
            m for m in self.message_history 
            if (now - m["timestamp"]).total_seconds() < 60
        ]
        
        return len(self.message_history) < self.rate_limit