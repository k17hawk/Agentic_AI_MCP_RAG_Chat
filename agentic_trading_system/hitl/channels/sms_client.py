"""
SMS Client - Send SMS messages via Twilio
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from twilio.rest import Client
from utils.logger import logger as  logging

class SMSClient:
    """
    SMS Client using Twilio API for urgent notifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get credentials from environment or config
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID") or config.get("account_sid")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN") or config.get("auth_token")
        self.from_number = os.getenv("TWILIO_SMS_NUM") or config.get("from_number", "+14155238886")
        self.to_number = os.getenv("YOUR_PHONE_NUM") or config.get("to_number")
        
        # Initialize Twilio client
        self.client = None
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logging.info("✅ SMSClient initialized with Twilio")
        else:
            logging.warning("⚠️ Twilio credentials not found - SMS client disabled")
        
        # Rate limiting
        self.sent_count = 0
        self.last_reset = datetime.now()
        self.hourly_limit = config.get("hourly_limit", 10)  # Max 10 SMS per hour
        
    async def send_sms(self, to: str, message: str) -> Dict[str, Any]:
        """
        Send an SMS message (only for urgent alerts)
        """
        # Truncate message for SMS (160 chars)
        if len(message) > 160:
            message = message[:157] + "..."
        
        if not self.client:
            logging.warning(f"SMS client not configured - would send: {message}")
            return {
                "success": False,
                "simulated": True,
                "message": message,
                "to": to
            }
        
        try:
            # Check rate limit
            if not self._check_rate_limit():
                logging.warning("SMS rate limit exceeded")
                return {
                    "success": False,
                    "error": "Rate limit exceeded"
                }
            
            # Send SMS
            msg = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to
            )
            
            self.sent_count += 1
            logging.info(f"✅ SMS sent to {to}")
            
            return {
                "success": True,
                "sid": msg.sid,
                "status": msg.status,
                "to": to,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"❌ SMS send error: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to
            }
    
    async def send_urgent_alert(self, alert_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an urgent alert via SMS
        """
        message = self._build_alert_message(alert_type, data)
        return await self.send_sms(self.to_number, message)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within hourly limits"""
        now = datetime.now()
        
        # Reset counter if new hour
        if now.hour != self.last_reset.hour or now.date() > self.last_reset.date():
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.hourly_limit
    
    def _build_alert_message(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Build alert message"""
        if alert_type == "trade_signal":
            return f"🚨 TRADE: {data.get('symbol')} {data.get('action')} @ ${data.get('price')} Conf:{data.get('confidence')}%"
        elif alert_type == "system_error":
            return f"⚠️ ERROR: {data.get('component')} - {data.get('message')}"
        elif alert_type == "price_alert":
            return f"💰 PRICE: {data.get('symbol')} {data.get('change')}% - ${data.get('price')}"
        elif alert_type == "timeout":
            return f"⏰ TIMEOUT: {data.get('symbol')} - No response, auto-rejected"
        else:
            return f"Alert: {str(data)[:100]}"