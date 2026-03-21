#!/usr/bin/env python3
"""
Sequential Test: HITL (Human-in-the-Loop) Module
Tests WhatsApp alerts, response parsing, timeout handling, and decision tracking
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from agentic_trading_system.hitl.alert_manager import AlertManager
from agentic_trading_system.hitl.message_builder import MessageBuilder
from agentic_trading_system.hitl.response_parser import ResponseParser
from agentic_trading_system.hitl.pending_queue import PendingQueue
from agentic_trading_system.hitl.timeout_manager import TimeoutManager
from agentic_trading_system.hitl.decision_tracker import DecisionTracker
from agentic_trading_system.hitl.feedback_logger import FeedbackLogger
from agentic_trading_system.agents.base_agent import AgentMessage
from agentic_trading_system.utils.logger import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class HITLSequentialTest:
    """
    Sequential test for HITL module
    """
    
    def __init__(self):
        self.portfolio_results = self.load_portfolio_results()
        self.results = []
        self.errors = []
        
        # Initialize HITL components
        self.alert_manager = AlertManager(
            name="TestAlertManager",
            config={
                "whatsapp_config": {"enabled": False, "mock_mode": True},
                "email_config": {"enabled": False},
                "sms_config": {"enabled": False},
                "dashboard_config": {"enabled": True},
                "message_config": {},
                "parser_config": {},
                "queue_config": {"max_size": 100, "default_ttl_seconds": 60},
                "timeout_config": {"default_timeout_seconds": 30, "reminder_interval_seconds": 10},
                "decision_config": {},
                "feedback_config": {}
            }
        )
        
        # WhatsApp-only message builder
        self.message_builder = MessageBuilder({})
        self.response_parser = ResponseParser({})
        self.pending_queue = PendingQueue({})
        self.timeout_manager = TimeoutManager({})
        self.decision_tracker = DecisionTracker({})
        self.feedback_logger = FeedbackLogger({})
        
        logging.info("✅ HITL components initialized")
    
    def load_portfolio_results(self):
        """Load previous portfolio results"""
        try:
            with open("data/portfolio_results.json", "r") as f:
                data = json.load(f)
                return data.get("risk_results", [])
        except FileNotFoundError:
            print("⚠️ No portfolio results found. Using mock data.")
            return [
                {"symbol": "AAPL", "recommended_position": 1664, "recommended_shares": 6, "price": 247.99, "action": "BUY"},
                {"symbol": "MSFT", "recommended_position": 1341, "recommended_shares": 3, "price": 381.87, "action": "BUY"},
                {"symbol": "GOOGL", "recommended_position": 1327, "recommended_shares": 4, "price": 301.00, "action": "BUY"},
                {"symbol": "NVDA", "recommended_position": 983, "recommended_shares": 5, "price": 172.70, "action": "BUY"},
                {"symbol": "META", "recommended_position": 1026, "recommended_shares": 1, "price": 593.66, "action": "BUY"}
            ]
    
    async def test_message_builder(self):
        """
        Test 1: Message Builder (WhatsApp-only)
        """
        print(f"\n   📝 Testing Message Builder:")
        print("   " + "-" * 40)
        
        trade = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 247.99,
            "confidence": 0.75,
            "shares": 6,
            "position_value": 1664,
            "stop_loss": 235.59,
            "take_profit": 260.00,
            "rr_ratio": 2.5,
            "reasons": ["Strong technical signal", "Good fundamentals", "Bullish market trend"],
            "concerns": ["Market volatility", "Sector rotation"]
        }
        
        # WhatsApp message is a string now
        whatsapp_msg = self.message_builder.build_approval_request(trade)
        
        print(f"      • WhatsApp Message Preview:")
        print(f"         {whatsapp_msg[:200]}...")
        
        
        return {"whatsapp": whatsapp_msg}
    
    async def test_response_parser(self):
        """
        Test 2: Response Parser
        """
        print(f"\n   🔍 Testing Response Parser:")
        print("   " + "-" * 40)
        
        test_responses = [
            "YES AAPL",
            "NO TSLA",
            "yes msft",
            "NO",
            "approve GOOGL",
            "reject NVDA",
            "BUY 100 META",
            "what is the status",
            "help"
        ]
        
        for response in test_responses:
            parsed = self.response_parser.parse(response)
            print(f"      • '{response}' → Type: {parsed['type']}, Action: {parsed['action']}, Symbols: {parsed['symbols']}")
        
        return {"parsed_responses": [self.response_parser.parse(r) for r in test_responses]}
    
    async def test_pending_queue(self):
        """
        Test 3: Pending Queue
        """
        print(f"\n   📋 Testing Pending Queue:")
        print("   " + "-" * 40)
        
        # Add items to queue
        for r in self.portfolio_results[:3]:
            item_id = await self.pending_queue.add({
                "symbol": r['symbol'],
                "shares": r['recommended_shares'],
                "price": r['price'],
                "value": r['recommended_position']
            }, priority="high")
            print(f"      • Added: {r['symbol']} (ID: {item_id})")
        
        # Get queue stats
        stats = self.pending_queue.get_stats()
        print(f"      • Queue Size: {stats['pending_total']}")   # fixed key
        print(f"      • Total Added: {stats['total_added']}")
        
        # Get next item
        next_item = await self.pending_queue.get_next()
        if next_item:
            print(f"      • Next Item: {next_item['data']['symbol']}")
        
        return stats
    
    async def test_timeout_manager(self):
        """
        Test 4: Timeout Manager
        """
        print(f"\n   ⏰ Testing Timeout Manager:")
        print("   " + "-" * 40)
        
        # Register timeout
        timeout_info = self.timeout_manager.register_timeout(
            "test_item_1",
            timeout_seconds=5,
            data={"symbol": "AAPL", "value": 1664}
        )
        print(f"      • Timeout registered: {timeout_info}")
        
        # Get remaining time
        remaining = self.timeout_manager.get_remaining_time("test_item_1")
        print(f"      • Remaining time: {remaining}s")
        
        # Cancel timeout
        cancelled = await self.timeout_manager.cancel_timeout("test_item_1")
        print(f"      • Cancelled: {cancelled}")
        
        return {"registered": timeout_info, "cancelled": cancelled}
    
    async def test_decision_tracker(self):
        """
        Test 5: Decision Tracker
        """
        print(f"\n   📊 Testing Decision Tracker:")
        print("   " + "-" * 40)
        
        # Track a decision
        decision_id = await self.decision_tracker.track_decision({
            "symbol": "AAPL",
            "decision": "approve",
            "response_time": 2.5,
            "confidence": 0.75,
            "risk_score": 0.38,
            "action": "BUY",
            "source": "whatsapp"
        })
        print(f"      • Decision tracked: {decision_id}")
        
        # Update outcome
        updated = await self.decision_tracker.update_outcome(decision_id, "win", pnl=150.00)
        print(f"      • Outcome updated: {updated}")
        
        # Get approval rate
        approval_rate = self.decision_tracker.get_approval_rate(days=30)
        print(f"      • Approval Rate (30d): {approval_rate}")
        
        # Get summary
        summary = self.decision_tracker.get_decision_summary(days=7)
        print(f"      • Recent Decisions: {summary}")
        
        return {"decision_id": decision_id, "approval_rate": approval_rate}
    
    async def test_feedback_logger(self):
        """
        Test 6: Feedback Logger
        """
        print(f"\n   💬 Testing Feedback Logger:")
        print("   " + "-" * 40)
        
        # Log trigger feedback
        fb_id = await self.feedback_logger.log_trigger_feedback(
            trigger_name="price_alert",
            triggered=True,
            outcome="win",
            metadata={"symbol": "AAPL", "z_score": 2.5}
        )
        print(f"      • Trigger feedback logged: {fb_id}")
        
        # Log analysis feedback
        fb_id2 = await self.feedback_logger.log_analysis_feedback(
            symbol="MSFT",
            analysis_type="technical",
            predicted_score=0.72,
            actual_outcome="win",
            metadata={"rsi": 65, "macd": "bullish"}
        )
        print(f"      • Analysis feedback logged: {fb_id2}")
        
        # Get suggestions
        suggestions = self.feedback_logger.get_improvement_suggestions()
        print(f"      • Improvement suggestions: {len(suggestions)}")
        
        return {"suggestions": suggestions}
    
    async def test_alert_manager_send_alert(self):
        """
        Test 7: Alert Manager - Send Trade Alert
        """
        print(f"\n   📢 Testing Alert Manager - Send Trade Alert:")
        print("   " + "-" * 40)
        
        trade = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 247.99,
            "confidence": 0.75,
            "shares": 6,
            "position_value": 1664,
            "stop_loss": 235.59,
            "take_profit": 260.00,
            "rr_ratio": 2.5,
            "reasons": ["Strong technical signal", "Good fundamentals"],
            "concerns": ["Market volatility"]
        }
        
        result = await self.alert_manager.send_trade_alert(trade, requester="Test")
        
        print(f"      • Alert Sent: {result.message_type}")
        print(f"      • Item ID: {result.content.get('item_id')}")
        print(f"      • Symbol: {result.content.get('symbol')}")      # fixed key
        print(f"      • Sent Status: {result.content.get('sent')}")   # fixed key
        
        return result.content
    
    async def test_alert_manager_handle_response(self):
        """
        Test 8: Alert Manager - Handle Human Response
        """
        print(f"\n   📨 Testing Alert Manager - Handle Response:")
        print("   " + "-" * 40)
        
        # First, send an alert to create a pending item
        trade = {
            "symbol": "AAPL",
            "action": "BUY",
            "price": 247.99,
            "confidence": 0.75,
            "shares": 6,
            "position_value": 1664,
            "stop_loss": 235.59,
            "take_profit": 260.00
        }
        
        alert_result = await self.alert_manager.send_trade_alert(trade, requester="Test")
        item_id = alert_result.content.get("item_id")
        
        # Simulate a human response
        response = {
            "message": "YES AAPL",
            "channel": "whatsapp"
        }
        
        # Handle the response
        result = await self.alert_manager.handle_response(response, requester="Test")
        
        print(f"      • Response Type: {result.message_type}")
        print(f"      • Processed Items: {len(result.content.get('results', []))}")
        
        for r in result.content.get('results', []):
            print(f"         {r['symbol']}: {r['decision']} - {r['status']}")
        
        return result.content
    
    async def test_alert_manager_timeout(self):
        """
        Test 9: Alert Manager - Timeout Handling
        """
        print(f"\n   ⏰ Testing Alert Manager - Timeout:")
        print("   " + "-" * 40)
        
        # Send an alert with short timeout
        trade = {
            "symbol": "TSLA",
            "action": "BUY",
            "price": 367.96,
            "confidence": 0.65,
            "shares": 2,
            "position_value": 736,
            "stop_loss": 337.06,
            "take_profit": 400.00
        }
        
        # Override timeout for this test
        self.alert_manager.timeout_manager.default_timeout = 5
        
        alert_result = await self.alert_manager.send_trade_alert(trade, requester="Test")
        item_id = alert_result.content.get("item_id")
        
        print(f"      • Alert sent, waiting 6 seconds for timeout...")
        
        # Wait for timeout
        await asyncio.sleep(6)
        
        # Check if item was auto-rejected
        pending = await self.alert_manager.pending_queue.get_pending_count()
        print(f"      • Pending after timeout: {pending}")
        
        # Check decision tracker for timeout decision
        summary = self.decision_tracker.get_decision_summary(days=1)
        print(f"      • Decisions today: {summary.get('total_decisions', 0)}")
        
        return {"timeout_handled": True}
    
    async def test_alert_manager_status(self):
        """
        Test 10: Alert Manager - Get Status
        """
        print(f"\n   📊 Testing Alert Manager - Get Status:")
        print("   " + "-" * 40)
        
        status = self.alert_manager.get_status()
        
        print(f"      • Status: {status['status']}")
        print(f"      • Pending Queue Size: {status['pending_queue']['pending_total']}")   # fixed key
        print(f"      • WhatsApp Configured: {status['whatsapp_configured']}")              # fixed
        
        return status
    
    async def run(self):
        """
        Run all HITL tests
        """
        print("\n" + "="*70)
        print("🚀 HITL MODULE SEQUENTIAL TEST")
        print("="*70)
        
        # First, show loaded portfolio results
        print("\n📋 Loaded Portfolio Results:")
        for r in self.portfolio_results:
            # Ensure action is present (default to WATCH if missing)
            action = r.get('action', 'WATCH')
            print(f"   • {r['symbol']}: ${r['recommended_position']:,.0f} ({r['recommended_shares']} shares) - {action}")
        
        # Test 1: Message Builder
        await self.test_message_builder()
        
        # Test 2: Response Parser
        await self.test_response_parser()
        
        # Test 3: Pending Queue
        await self.test_pending_queue()
        
        # Test 4: Timeout Manager
        await self.test_timeout_manager()
        
        # Test 5: Decision Tracker
        await self.test_decision_tracker()
        
        # Test 6: Feedback Logger
        await self.test_feedback_logger()
        
        # Test 7: Alert Manager - Send Alert
        alert_result = await self.test_alert_manager_send_alert()
        
        # Test 8: Alert Manager - Handle Response
        await self.test_alert_manager_handle_response()
        
        # Test 9: Alert Manager - Timeout
        await self.test_alert_manager_timeout()
        
        # Test 10: Alert Manager - Get Status
        await self.test_alert_manager_status()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print summary of HITL analysis"""
        print("\n" + "="*70)
        print("📊 HITL MODULE SUMMARY")
        print("="*70)
        
        # Get decision stats
        decision_stats = self.decision_tracker.get_stats()
        
        print(f"\n📈 Decision Statistics:")
        print(f"   • Total Decisions: {decision_stats.get('total_decisions', 0)}")
        print(f"   • Approval Rate: {decision_stats.get('approval_rate', 0)*100:.1f}%")
        print(f"   • Avg Response Time: {decision_stats.get('avg_response_time', 0):.1f}s")
        
        print(f"\n📋 Pending Queue Stats:")
        queue_stats = self.pending_queue.get_stats()
        print(f"   • Current Size: {queue_stats['pending_total']}")          # fixed key
        print(f"   • Total Processed: {queue_stats['total_processed']}")
        
        print(f"\n💬 Feedback Stats:")
        suggestions = self.feedback_logger.get_improvement_suggestions()
        print(f"   • Improvement Suggestions: {len(suggestions)}")
        if suggestions:
            for s in suggestions[:3]:
                print(f"      - {s.get('suggestion', 'N/A')}")
        
        print(f"\n📢 Alert Manager Status:")
        status = self.alert_manager.get_status()
        print(f"   • Status: {status['status']}")
        print(f"   • Queue Size: {status['pending_queue']['pending_total']}")   # fixed key
    
    def save_results(self):
        """Save results to file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "decision_stats": self.decision_tracker.get_stats(),
            "queue_stats": self.pending_queue.get_stats(),
            "feedback_suggestions": self.feedback_logger.get_improvement_suggestions(),
            "alert_manager_status": self.alert_manager.get_status()
        }
        
        # Create directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        with open("data/hitl_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: data/hitl_results.json")

async def main():
    """Main entry point"""
    tester = HITLSequentialTest()
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())