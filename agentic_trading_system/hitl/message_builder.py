"""
Message Builder - Builds formatted messages for different channels
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logger import logger as logging

class MessageBuilder:
    """
    Message Builder - Creates formatted messages for different channels
    
    Supports:
    - WhatsApp (rich text with emojis)
    - Email (HTML and plain text)
    - SMS (short, concise)
    - Dashboard (JSON structure)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Channel-specific configurations
        self.sms_max_length = config.get("sms_max_length", 160)
        self.whatsapp_max_length = config.get("whatsapp_max_length", 4096)
        
        logging.info(f"✅ MessageBuilder initialized")
    
    def build_trade_alert(self, trade: Dict[str, Any], channel: str = "whatsapp") -> Dict[str, Any]:
        """
        Build trade alert message for specific channel
        """
        if channel == "whatsapp":
            return self._build_whatsapp_trade(trade)
        elif channel == "email":
            return self._build_email_trade(trade)
        elif channel == "sms":
            return self._build_sms_trade(trade)
        elif channel == "dashboard":
            return self._build_dashboard_trade(trade)
        else:
            return {"text": str(trade)}
    
    def build_approval_request(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build approval request message
        """
        symbol = trade.get("symbol", "UNKNOWN")
        action = trade.get("action", "WATCH")
        price = trade.get("price", 0)
        confidence = trade.get("confidence", 0) * 100
        target = trade.get("target", 0)
        stop = trade.get("stop", 0)
        reasons = trade.get("reasons", [])
        concerns = trade.get("concerns", [])
        
        # Format reasons
        reasons_text = "\n".join([f"   ✅ {r}" for r in reasons[:3]])
        concerns_text = "\n".join([f"   ⚠️ {c}" for c in concerns[:3]]) if concerns else "   None"
        
        whatsapp = (
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
        
        email = (
            f"<h2>Trade Signal: {action} {symbol}</h2>"
            f"<table border='1' cellpadding='5'>"
            f"<tr><td><b>Price</b></td><td>${price}</td></tr>"
            f"<tr><td><b>Confidence</b></td><td>{confidence:.1f}%</td></tr>"
            f"<tr><td><b>Risk Level</b></td><td>{trade.get('risk_level', 'MEDIUM')}</td></tr>"
            f"<tr><td><b>Stop Loss</b></td><td>${stop}</td></tr>"
            f"<tr><td><b>Take Profit</b></td><td>${target}</td></tr>"
            f"<tr><td><b>R/R Ratio</b></td><td>{trade.get('rr_ratio', 0)}:1</td></tr>"
            f"<tr><td><b>Position</b></td><td>{trade.get('shares', 0)} shares</td></tr>"
            f"<tr><td><b>Total Cost</b></td><td>${trade.get('position_value', 0):,.2f}</td></tr>"
            f"</table>"
            f"<h3>Why {action}:</h3><ul>"
            + "".join([f"<li>✅ {r}</li>" for r in reasons[:3]]) +
            f"</ul><h3>Concerns:</h3><ul>"
            + "".join([f"<li>⚠️ {c}</li>" for c in concerns[:3]]) +
            f"</ul><p>Reply with YES {symbol} to approve or NO {symbol} to reject.</p>"
        )
        
        sms = f"{action} {symbol} @ ${price} Conf:{confidence:.0f}% R/R:{trade.get('rr_ratio', 0)} Reply YES/NO {symbol}"
        
        return {
            "whatsapp": whatsapp,
            "email": email,
            "email_html": True,
            "sms": sms[:self.sms_max_length],
            "dashboard": trade
        }
    
    def build_execution_confirmation(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build execution confirmation message
        """
        symbol = execution.get("symbol", "UNKNOWN")
        shares = execution.get("shares", 0)
        price = execution.get("price", 0)
        total = execution.get("total", 0)
        time = datetime.now().strftime("%H:%M:%S")
        
        whatsapp = (
            f"✅ *TRADE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 {symbol}\n"
            f"💰 {shares} shares @ ${price}\n"
            f"💵 Total: ${total:,.2f}\n"
            f"🕐 {time}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        return {
            "whatsapp": whatsapp,
            "email": f"Trade Executed: {shares} {symbol} @ ${price}",
            "sms": f"✅ {symbol}: {shares} @ ${price}",
            "dashboard": execution
        }
    
    def build_rejection_notification(self, rejection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build rejection notification message
        """
        symbol = rejection.get("symbol", "UNKNOWN")
        reason = rejection.get("reason", "No reason provided")
        
        whatsapp = (
            f"❌ *TRADE REJECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📉 {symbol}\n"
            f"Reason: {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        return {
            "whatsapp": whatsapp,
            "email": f"Trade Rejected: {symbol} - {reason}",
            "sms": f"❌ {symbol} rejected",
            "dashboard": rejection
        }
    
    def build_timeout_alert(self, ticker: str) -> Dict[str, Any]:
        """
        Build timeout alert message
        """
        whatsapp = (
            f"⏰ *TIMEOUT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"No response for {ticker}\n"
            f"Auto-rejected.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        return {
            "whatsapp": whatsapp,
            "email": f"Timeout: {ticker} - No response, auto-rejected",
            "sms": f"⏰ {ticker} timeout - auto-rejected",
            "dashboard": {"ticker": ticker, "status": "timeout"}
        }
    
    def build_daily_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build daily summary message
        """
        date = datetime.now().strftime("%Y-%m-%d")
        trades = summary.get("trades", 0)
        pnl = summary.get("pnl", 0)
        approved = summary.get("approved", 0)
        rejected = summary.get("rejected", 0)
        win_rate = summary.get("win_rate", 0)
        
        whatsapp = (
            f"📊 *DAILY SUMMARY - {date}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Trades: {trades}\n"
            f"💰 P&L: ${pnl:+,.2f}\n"
            f"✅ Approved: {approved}\n"
            f"❌ Rejected: {rejected}\n"
            f"🎯 Win Rate: {win_rate:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        return {
            "whatsapp": whatsapp,
            "email": f"Daily Summary - {date}",
            "sms": f"Daily: {trades} trades, P&L ${pnl:+,.0f}",
            "dashboard": summary
        }
    
    def _build_whatsapp_trade(self, trade: Dict[str, Any]) -> str:
        """Build WhatsApp trade message"""
        symbol = trade.get("symbol", "UNKNOWN")
        action = trade.get("action", "WATCH")
        price = trade.get("price", 0)
        
        return f"*{action} {symbol}* @ ${price}"
    
    def _build_email_trade(self, trade: Dict[str, Any]) -> Dict[str, str]:
        """Build email trade message"""
        return {
            "subject": f"Trade Signal: {trade.get('action')} {trade.get('symbol')}",
            "body": str(trade)
        }
    
    def _build_sms_trade(self, trade: Dict[str, Any]) -> str:
        """Build SMS trade message"""
        text = f"{trade.get('action')} {trade.get('symbol')} @ ${trade.get('price')}"
        return text[:self.sms_max_length]
    
    def _build_dashboard_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Build dashboard trade payload"""
        return trade