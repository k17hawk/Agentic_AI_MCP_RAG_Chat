"""
Message Builder - Builds formatted messages for different channels
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from agentic_trading_system.utils.logger import logger as logging

class MessageBuilder:
    """
    Message Builder - Creates formatted WhatsApp messages
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whatsapp_max_length = config.get("whatsapp_max_length", 4096)
        logging.info(f"✅ MessageBuilder initialized (WhatsApp only)")
    
    def build_approval_request(self, trade: Dict[str, Any]) -> str:
        """
        Build WhatsApp approval request message
        """
        symbol = trade.get("symbol", "UNKNOWN")
        action = trade.get("action", "WATCH")
        price = trade.get("price", 0)
        confidence = trade.get("confidence", 0) * 100
        target = trade.get("target", 0)
        stop = trade.get("stop", 0)
        reasons = trade.get("reasons", [])
        concerns = trade.get("concerns", [])
        
        reasons_text = "\n".join([f"   ✅ {r}" for r in reasons[:3]])
        concerns_text = "\n".join([f"   ⚠️ {c}" for c in concerns[:3]]) if concerns else "   None"
        
        message = (
            f"🤖 *TRADE SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 *{action}: {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 *Price*        : ${price}\n"
            f"🎯 *Confidence*   : {confidence:.1f}%\n"
            f"📊 *Risk Level*   : {trade.get('risk_level', 'MEDIUM')}\n\n"
            f"🛑 *Stop Loss*    : ${stop}\n"
            f"🎯 *Take Profit*  : ${target}\n"
            f"📐 *R/R Ratio*    : {trade.get('rr_ratio', 0)}:1\n\n"
            f"📦 *Position*     : {trade.get('shares', 0)} shares\n"
            f"💵 *Total Cost*   : ${trade.get('position_value', 0):,.2f}\n\n"
            f"*Why {action}:*\n{reasons_text}\n\n"
            f"*Concerns:*\n{concerns_text}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Reply *YES {symbol}* to approve\n"
            f"Reply *NO {symbol}* to reject"
        )
        return message[:self.whatsapp_max_length]
    
    def build_execution_confirmation(self, execution: Dict[str, Any]) -> str:
        """
        Build WhatsApp execution confirmation message
        """
        symbol = execution.get("symbol", "UNKNOWN")
        shares = execution.get("shares", 0)
        price = execution.get("price", 0)
        total = execution.get("total", shares * price)
        time = datetime.now().strftime("%H:%M:%S")
        
        message = (
            f"✅ *TRADE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 {symbol}\n"
            f"💰 {shares} shares @ ${price}\n"
            f"💵 Total: ${total:,.2f}\n"
            f"🕐 {time}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return message[:self.whatsapp_max_length]
    
    def build_rejection_notification(self, rejection: Dict[str, Any]) -> str:
        """
        Build WhatsApp rejection notification message
        """
        symbol = rejection.get("symbol", "UNKNOWN")
        reason = rejection.get("reason", "No reason provided")
        
        message = (
            f"❌ *TRADE REJECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📉 {symbol}\n"
            f"Reason: {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return message[:self.whatsapp_max_length]
    
    def build_timeout_alert(self, ticker: str) -> str:
        """
        Build WhatsApp timeout alert message
        """
        message = (
            f"⏰ *TIMEOUT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"No response for {ticker}\n"
            f"Auto-rejected.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return message[:self.whatsapp_max_length]
    
    def build_daily_summary(self, summary: Dict[str, Any]) -> str:
        """
        Build WhatsApp daily summary message
        """
        date = datetime.now().strftime("%Y-%m-%d")
        trades = summary.get("trades", 0)
        pnl = summary.get("pnl", 0)
        approved = summary.get("approved", 0)
        rejected = summary.get("rejected", 0)
        win_rate = summary.get("win_rate", 0)
        
        message = (
            f"📊 *DAILY SUMMARY - {date}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Trades: {trades}\n"
            f"💰 P&L: ${pnl:+,.2f}\n"
            f"✅ Approved: {approved}\n"
            f"❌ Rejected: {rejected}\n"
            f"🎯 Win Rate: {win_rate:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return message[:self.whatsapp_max_length]