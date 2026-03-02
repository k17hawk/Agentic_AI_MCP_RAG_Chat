"""
Email Client - Send emails via SMTP
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from utils.logger import logging

class EmailClient:
    """
    Email Client using SMTP
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Email configuration
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = os.getenv("EMAIL_USERNAME") or config.get("username")
        self.password = os.getenv("EMAIL_PASSWORD") or config.get("password")
        self.from_address = config.get("from_address", self.username)
        self.to_addresses = config.get("to_addresses", [])
        
        # Rate limiting
        self.sent_count = 0
        self.last_reset = datetime.now()
        self.daily_limit = config.get("daily_limit", 100)
        
        logging.info(f"✅ EmailClient initialized for {self.from_address}")
    
    async def send_email(self, to: str, subject: str, body: str, 
                        html: bool = False, attachments: List[Dict] = None) -> Dict[str, Any]:
        """
        Send an email
        """
        if not self._check_daily_limit():
            logging.warning("Daily email limit exceeded")
            return {"success": False, "error": "Daily limit exceeded"}
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = to
            msg['Subject'] = subject
            
            # Add body
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    part = MIMEApplication(attachment['data'], Name=attachment['filename'])
                    part['Content-Disposition'] = f'attachment; filename="{attachment["filename"]}"'
                    msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            self.sent_count += 1
            logging.info(f"✅ Email sent to {to}: {subject}")
            
            return {
                "success": True,
                "to": to,
                "subject": subject,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"❌ Email send error: {e}")
            return {
                "success": False,
                "error": str(e),
                "to": to
            }
    
    async def send_template(self, to: str, template_name: str, 
                           variables: Dict[str, str]) -> Dict[str, Any]:
        """
        Send a template email
        """
        template = self._get_template(template_name)
        subject = template["subject"].format(**variables)
        body = template["body"].format(**variables)
        html = template.get("html", False)
        
        return await self.send_email(to, subject, body, html)
    
    async def send_bulk(self, recipients: List[str], subject: str, 
                       body: str) -> List[Dict[str, Any]]:
        """
        Send same email to multiple recipients
        """
        results = []
        for recipient in recipients:
            result = await self.send_email(recipient, subject, body)
            results.append(result)
        return results
    
    def _check_daily_limit(self) -> bool:
        """Check if we're within daily limits"""
        now = datetime.now()
        
        # Reset counter if new day
        if now.date() > self.last_reset.date():
            self.sent_count = 0
            self.last_reset = now
        
        return self.sent_count < self.daily_limit
    
    def _get_template(self, name: str) -> Dict[str, str]:
        """Get email template"""
        templates = {
            "trade_alert": {
                "subject": "🔔 Trade Signal: {symbol} - {action}",
                "body": (
                    "Trade Signal\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Symbol: {symbol}\n"
                    "Action: {action}\n"
                    "Price: ${price}\n"
                    "Confidence: {confidence}%\n"
                    "Target: ${target}\n"
                    "Stop: ${stop}\n\n"
                    "Please review and respond."
                )
            },
            "daily_report": {
                "subject": "📊 Daily Trading Report - {date}",
                "body": (
                    "Daily Trading Report\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Date: {date}\n"
                    "Total Trades: {trades}\n"
                    "Approved: {approved}\n"
                    "Rejected: {rejected}\n"
                    "P&L: {pnl}\n"
                    "Win Rate: {win_rate}%\n\n"
                    "See attachment for details."
                ),
                "html": False
            },
            "system_alert": {
                "subject": "⚠️ System Alert - {severity}",
                "body": (
                    "System Alert\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    "Component: {component}\n"
                    "Severity: {severity}\n"
                    "Message: {message}\n"
                    "Time: {time}\n\n"
                    "Please investigate."
                )
            }
        }
        
        return templates.get(name, {
            "subject": "Notification",
            "body": "{message}"
        })