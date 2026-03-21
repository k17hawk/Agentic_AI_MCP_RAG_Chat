"""
Dashboard Notifier - Push notifications to web dashboard
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from agentic_trading_system.utils.logger import logger as logging

class DashboardNotifier:
    """
    Dashboard Notifier - Push real-time updates to web dashboard
    
    In production, this would connect to WebSocket/Socket.IO
    For now, it logs to a file that the dashboard can read
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configuration
        self.notification_file = config.get("notification_file", "data/dashboard_notifications.json")
        self.max_notifications = config.get("max_notifications", 100)
        self.enable_websocket = config.get("enable_websocket", False)
        
        # In-memory queue
        self.notifications = []
        self.subscribers = []
        
        # WebSocket server (would be initialized in production)
        self.ws_server = None
        
        logging.info(f"✅ DashboardNotifier initialized")
    
    async def push_notification(self, notification_type: str, 
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push a notification to the dashboard
        """
        notification = {
            "id": f"notif_{datetime.now().timestamp()}",
            "type": notification_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }
        
        # Add to queue
        self.notifications.append(notification)
        
        # Trim queue
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Save to file
        self._save_notifications()
        
        # Push to WebSocket if enabled
        if self.enable_websocket:
            await self._push_websocket(notification)
        
        logging.info(f"📊 Dashboard notification: {notification_type}")
        
        return notification
    
    async def push_trade_signal(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Push trade signal to dashboard"""
        return await self.push_notification("trade_signal", {
            "symbol": trade.get("symbol"),
            "action": trade.get("action"),
            "price": trade.get("price"),
            "confidence": trade.get("confidence"),
            "target": trade.get("target"),
            "stop": trade.get("stop"),
            "analysis": trade.get("analysis", {})
        })
    
    async def push_trade_execution(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Push trade execution to dashboard"""
        return await self.push_notification("trade_execution", {
            "symbol": execution.get("symbol"),
            "shares": execution.get("shares"),
            "price": execution.get("price"),
            "total": execution.get("total"),
            "status": execution.get("status")
        })
    
    async def push_portfolio_update(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Push portfolio update to dashboard"""
        return await self.push_notification("portfolio_update", {
            "total_value": portfolio.get("total_value"),
            "cash": portfolio.get("cash"),
            "positions": len(portfolio.get("positions", [])),
            "daily_pnl": portfolio.get("daily_pnl"),
            "total_pnl": portfolio.get("total_pnl")
        })
    
    async def push_system_alert(self, severity: str, message: str, 
                               component: str = None) -> Dict[str, Any]:
        """Push system alert to dashboard"""
        return await self.push_notification("system_alert", {
            "severity": severity,
            "message": message,
            "component": component
        })
    
    async def get_notifications(self, limit: int = 50, unread_only: bool = False) -> List[Dict]:
        """Get recent notifications"""
        notifications = self.notifications[-limit:]
        
        if unread_only:
            notifications = [n for n in notifications if not n["read"]]
        
        return notifications
    
    async def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        for notification in self.notifications:
            if notification["id"] == notification_id:
                notification["read"] = True
                self._save_notifications()
                return True
        return False
    
    async def mark_all_read(self) -> int:
        """Mark all notifications as read"""
        count = 0
        for notification in self.notifications:
            if not notification["read"]:
                notification["read"] = True
                count += 1
        
        if count > 0:
            self._save_notifications()
        
        return count
    
    def subscribe(self, callback):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
        logging.info(f"📡 New dashboard subscriber: {len(self.subscribers)} total")
    
    def _save_notifications(self):
        """Save notifications to file"""
        try:
            with open(self.notification_file, 'w') as f:
                json.dump({
                    "notifications": self.notifications,
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving notifications: {e}")
    
    async def _push_websocket(self, notification: Dict):
        """Push to WebSocket clients"""
        # In production, implement WebSocket broadcast
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logging.error(f"Error in subscriber callback: {e}")