"""
Report Generator - Main orchestrator for all report generation
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import json
from jinja2 import Environment, FileSystemLoader
import markdown

from agentic_trading_system.reporting.pdf_builder import PDFBuilder
from utils.logger import logging
from agents.base_agent import BaseAgent, AgentMessage

class ReportGenerator(BaseAgent):
    """
    Report Generator - Main orchestrator for all report generatio
    
    Responsibilities:
    - Generate daily/weekly/monthly reports
    - Coordinate with template engine
    - Integrate with PDF/Email builders
    - Schedule report generation
    - Maintain report history
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Report generation and distribution",
            config=config
        )
        
        # Template configuration
        self.template_dir = config.get("template_dir", "reporting/templates")
        self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        
        # Output directories
        self.report_dir = config.get("report_dir", "data/reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Report builders (lazy loaded)
        self.pdf_builder = None
        self.email_builder = None
        self.whatsapp_builder = None
        self.export_engine = None
        
        # Report schedule
        self.schedule = config.get("schedule", {
            "daily": {"hour": 18, "minute": 0},  # 6 PM daily
            "weekly": {"day": 5, "hour": 17, "minute": 0},  # Friday 5 PM
            "monthly": {"day": 1, "hour": 9, "minute": 0}  # 1st of month 9 AM
        })
        
        # Report history
        self.report_history = []
        self.max_history = config.get("max_history", 100)
        
        logging.info(f"✅ ReportGenerator initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process report generation requests
        """
        msg_type = message.message_type
        
        if msg_type == "generate_daily_report":
            # Generate daily report
            date = message.content.get("date", datetime.now().strftime("%Y-%m-%d"))
            report = await self.generate_daily_report(date)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="report_generated",
                content=report
            )
        
        elif msg_type == "generate_weekly_report":
            # Generate weekly report
            end_date = message.content.get("end_date", datetime.now().strftime("%Y-%m-%d"))
            report = await self.generate_weekly_report(end_date)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="report_generated",
                content=report
            )
        
        elif msg_type == "generate_monthly_report":
            # Generate monthly report
            month = message.content.get("month", datetime.now().strftime("%Y-%m"))
            report = await self.generate_monthly_report(month)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="report_generated",
                content=report
            )
        
        elif msg_type == "generate_trade_confirmation":
            # Generate trade confirmation
            trade_data = message.content
            report = await self.generate_trade_confirmation(trade_data)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="confirmation_generated",
                content=report
            )
        
        elif msg_type == "send_report":
            # Send report via specified channel
            report_path = message.content.get("report_path")
            channel = message.content.get("channel", "email")
            recipients = message.content.get("recipients", [])
            
            result = await self.send_report(report_path, channel, recipients)
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="report_sent",
                content=result
            )
        
        return None
    
    async def generate_daily_report(self, date: str) -> Dict[str, Any]:
        """
        Generate daily performance report
        """
        logging.info(f"📊 Generating daily report for {date}")
        
        # Fetch data from memory/analytics
        # This would integrate with other modules
        trades = await self._get_trades_for_date(date)
        metrics = await self._get_metrics_for_date(date)
        signals = await self._get_signals_for_date(date)
        
        # Prepare template data
        template_data = {
            "date": date,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_pnl": metrics.get('total_pnl', 0),
                "win_rate": metrics.get('win_rate', 0),
                "total_trades": len(trades),
                "wins": len([t for t in trades if t.get('pnl', 0) > 0]),
                "losses": len([t for t in trades if t.get('pnl', 0) < 0]),
                "open_positions": metrics.get('open_positions', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0)
            },
            "trades": trades,
            "metrics": {
                "gross_profit": metrics.get('gross_profit', 0),
                "gross_loss": metrics.get('gross_loss', 0),
                "net_profit": metrics.get('net_profit', 0),
                "profit_factor": metrics.get('profit_factor', 0),
                "avg_win": metrics.get('avg_win', 0),
                "avg_loss": metrics.get('avg_loss', 0),
                "max_drawdown": metrics.get('max_drawdown', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "sortino_ratio": metrics.get('sortino_ratio', 0)
            },
            "signals": signals
        }
        
        # Render HTML
        html_content = await self._render_template("daily_digest.html", template_data)
        
        # Generate PDF
        pdf_path = await self._generate_pdf(html_content, f"daily_report_{date}")
        
        # Save metadata
        report = {
            "type": "daily",
            "date": date,
            "html_path": None,  # Not saving HTML by default
            "pdf_path": pdf_path,
            "generated_at": datetime.now().isoformat(),
            "summary": template_data["summary"]
        }
        
        self._add_to_history(report)
        
        return report
    
    async def generate_weekly_report(self, end_date: str) -> Dict[str, Any]:
        """
        Generate weekly performance report
        """
        logging.info(f"📊 Generating weekly report ending {end_date}")
        
        # Calculate date range
        end = datetime.fromisoformat(end_date)
        start = end - timedelta(days=7)
        start_date = start.strftime("%Y-%m-%d")
        
        # Fetch data
        trades = await self._get_trades_for_range(start_date, end_date)
        metrics = await self._get_metrics_for_range(start_date, end_date)
        
        # Prepare daily breakdown
        daily_breakdown = []
        current = start
        while current <= end:
            day_trades = [t for t in trades if t.get('date', '')[:10] == current.strftime("%Y-%m-%d")]
            day_pnl = sum(t.get('pnl', 0) for t in day_trades)
            day_wins = len([t for t in day_trades if t.get('pnl', 0) > 0])
            day_total = len(day_trades)
            
            daily_breakdown.append({
                "date": current.strftime("%Y-%m-%d"),
                "trades": day_total,
                "pnl": day_pnl,
                "win_rate": day_wins / day_total if day_total > 0 else 0
            })
            current += timedelta(days=1)
        
        # Strategy and symbol performance
        strategy_performance = await self._get_strategy_performance(start_date, end_date)
        symbol_performance = await self._get_symbol_performance(start_date, end_date)
        
        # Risk metrics
        risk_metrics = await self._get_risk_metrics(start_date, end_date)
        
        template_data = {
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_pnl": metrics.get('total_pnl', 0),
                "pnl_change": metrics.get('pnl_change', 0),
                "win_rate": metrics.get('win_rate', 0),
                "total_trades": len(trades),
                "wins": len([t for t in trades if t.get('pnl', 0) > 0]),
                "losses": len([t for t in trades if t.get('pnl', 0) < 0]),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "max_drawdown": metrics.get('max_drawdown', 0)
            },
            "metrics": {
                "gross_profit": metrics.get('gross_profit', 0),
                "gross_loss": metrics.get('gross_loss', 0),
                "net_profit": metrics.get('net_profit', 0)
            },
            "daily_breakdown": daily_breakdown,
            "strategy_performance": strategy_performance,
            "symbol_performance": symbol_performance,
            "risk_metrics": risk_metrics
        }
        
        # Render HTML
        html_content = await self._render_template("weekly_report.html", template_data)
        
        # Generate PDF
        pdf_path = await self._generate_pdf(html_content, f"weekly_report_{end_date}")
        
        report = {
            "type": "weekly",
            "start_date": start_date,
            "end_date": end_date,
            "pdf_path": pdf_path,
            "generated_at": datetime.now().isoformat(),
            "summary": template_data["summary"]
        }
        
        self._add_to_history(report)
        
        return report
    
    async def generate_monthly_report(self, month: str) -> Dict[str, Any]:
        """
        Generate monthly performance report
        """
        logging.info(f"📊 Generating monthly report for {month}")
        
        # Parse month
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Fetch data
        trades = await self._get_trades_for_range(start_str, end_str)
        metrics = await self._get_metrics_for_range(start_str, end_str)
        previous_metrics = await self._get_metrics_for_range(
            (start_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            start_str
        )
        
        # Strategy comparison
        strategy_comparison = await self._get_strategy_comparison(start_str, end_str)
        
        # Generate insights
        key_insight = self._generate_key_insight(metrics, previous_metrics)
        recommendations = self._generate_recommendations(metrics, strategy_comparison)
        
        # Executive summary
        executive_summary = self._generate_executive_summary(metrics, key_insight)
        
        # Detailed metrics
        details = {
            "total_trades": len(trades),
            "winning_trades": len([t for t in trades if t.get('pnl', 0) > 0]),
            "losing_trades": len([t for t in trades if t.get('pnl', 0) < 0]),
            "breakeven_trades": len([t for t in trades if t.get('pnl', 0) == 0]),
            "avg_win": metrics.get('avg_win', 0),
            "avg_loss": metrics.get('avg_loss', 0),
            "largest_win": metrics.get('largest_win', 0),
            "largest_loss": metrics.get('largest_loss', 0),
            "sharpe_ratio": metrics.get('sharpe_ratio', 0),
            "sortino_ratio": metrics.get('sortino_ratio', 0),
            "calmar_ratio": metrics.get('calmar_ratio', 0),
            "profit_factor": metrics.get('profit_factor', 0),
            "max_drawdown": metrics.get('max_drawdown', 0),
            "recovery_factor": metrics.get('recovery_factor', 0),
            "var_95": metrics.get('var_95', 0),
            "cvar_95": metrics.get('cvar_95', 0)
        }
        
        template_data = {
            "month": start_date.strftime("%B"),
            "year": year,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "executive_summary": executive_summary,
            "key_insight": key_insight,
            "recommendations": recommendations,
            "summary": {
                "total_pnl": metrics.get('total_pnl', 0),
                "pnl_change": ((metrics.get('total_pnl', 0) - previous_metrics.get('total_pnl', 0)) / 
                               abs(previous_metrics.get('total_pnl', 1)) * 100),
                "win_rate": metrics.get('win_rate', 0),
                "wins": len([t for t in trades if t.get('pnl', 0) > 0]),
                "losses": len([t for t in trades if t.get('pnl', 0) < 0]),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "max_drawdown": metrics.get('max_drawdown', 0),
                "recovery_days": metrics.get('recovery_days', 0)
            },
            "strategy_comparison": strategy_comparison,
            "details": details
        }
        
        # Render HTML
        html_content = await self._render_template("monthly_report.html", template_data)
        
        # Generate PDF
        pdf_path = await self._generate_pdf(html_content, f"monthly_report_{month}")
        
        report = {
            "type": "monthly",
            "month": month,
            "year": year,
            "pdf_path": pdf_path,
            "generated_at": datetime.now().isoformat(),
            "summary": template_data["summary"]
        }
        
        self._add_to_history(report)
        
        return report
    
    async def generate_trade_confirmation(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trade confirmation document
        """
        logging.info(f"📄 Generating trade confirmation for {trade_data.get('trade_id')}")
        
        template_data = {
            "trade_id": trade_data.get("trade_id"),
            "order_id": trade_data.get("order_id"),
            "symbol": trade_data.get("symbol"),
            "action": trade_data.get("action"),
            "quantity": trade_data.get("quantity"),
            "price": trade_data.get("price"),
            "total_value": trade_data.get("quantity", 0) * trade_data.get("price", 0),
            "status": trade_data.get("status", "EXECUTED"),
            "order_type": trade_data.get("order_type", "MARKET"),
            "time_in_force": trade_data.get("time_in_force", "DAY"),
            "order_time": trade_data.get("order_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "execution_time": trade_data.get("execution_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "broker": trade_data.get("broker", "Paper Trading"),
            "commission": trade_data.get("commission", 0),
            "slippage": trade_data.get("slippage", 0),
            "expected_price": trade_data.get("expected_price", trade_data.get("price")),
            "execution_price": trade_data.get("execution_price", trade_data.get("price")),
            "price_improvement": trade_data.get("price_improvement", 0),
            "fill_quality": trade_data.get("fill_quality", "Good"),
            "analysis": trade_data.get("analysis"),
            "confirmation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Render HTML
        html_content = await self._render_template("trade_confirmation.html", template_data)
        
        # Generate PDF
        pdf_path = await self._generate_pdf(
            html_content, 
            f"trade_confirmation_{trade_data.get('trade_id')}"
        )
        
        return {
            "trade_id": trade_data.get("trade_id"),
            "pdf_path": pdf_path,
            "generated_at": datetime.now().isoformat()
        }
    
    async def send_report(self, report_path: str, channel: str, recipients: List[str]) -> Dict[str, Any]:
        """
        Send report via specified channel
        """
        result = {"channel": channel, "success": False, "recipients": recipients}
        
        if channel == "email":
            if not self.email_builder:
                from reporting.email_builder import EmailBuilder
                self.email_builder = EmailBuilder(self.config.get("email_config", {}))
            
            subject = f"Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
            result = await self.email_builder.send_report(report_path, subject, recipients)
        
        elif channel == "whatsapp":
            if not self.whatsapp_builder:
                from reporting.whatsapp_builder import WhatsAppBuilder
                self.whatsapp_builder = WhatsAppBuilder(self.config.get("whatsapp_config", {}))
            
            # WhatsApp is better for summaries, not full reports
            result = {"channel": "whatsapp", "success": False, "message": "WhatsApp not suitable for full reports"}
        
        return result
    
    async def _render_template(self, template_name: str, data: Dict) -> str:
        """Render HTML template with data"""
        template = self.jinja_env.get_template(template_name)
        return template.render(**data)
    
    async def _generate_pdf(self, html_content: str, filename_prefix: str) -> str:
        """Generate PDF from HTML"""
        if not self.pdf_builder:
            
            self.pdf_builder = PDFBuilder(self.config.get("pdf_config", {}))
        
        return await self.pdf_builder.generate(html_content, filename_prefix)
    
    async def _get_trades_for_date(self, date: str) -> List[Dict]:
        """Get trades for a specific date"""
        # This would query the trade repository
        # Placeholder implementation
        return []
    
    async def _get_trades_for_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get trades for a date range"""
        # This would query the trade repository
        # Placeholder implementation
        return []
    
    async def _get_metrics_for_date(self, date: str) -> Dict:
        """Get metrics for a specific date"""
        # This would query the metrics engine
        # Placeholder implementation
        return {}
    
    async def _get_metrics_for_range(self, start_date: str, end_date: str) -> Dict:
        """Get metrics for a date range"""
        # This would query the metrics engine
        # Placeholder implementation
        return {}
    
    async def _get_signals_for_date(self, date: str) -> List[Dict]:
        """Get signals for a specific date"""
        # This would query the signal repository
        # Placeholder implementation
        return []
    
    async def _get_strategy_performance(self, start_date: str, end_date: str) -> List[Dict]:
        """Get performance by strategy"""
        # This would query the attribution engine
        # Placeholder implementation
        return []
    
    async def _get_symbol_performance(self, start_date: str, end_date: str) -> List[Dict]:
        """Get performance by symbol"""
        # This would query the trade repository
        # Placeholder implementation
        return []
    
    async def _get_risk_metrics(self, start_date: str, end_date: str) -> Dict:
        """Get risk metrics"""
        # This would query the risk manager
        # Placeholder implementation
        return {}
    
    async def _get_strategy_comparison(self, start_date: str, end_date: str) -> List[Dict]:
        """Get strategy comparison data"""
        # This would query the attribution engine
        # Placeholder implementation
        return []
    
    def _generate_key_insight(self, current: Dict, previous: Dict) -> str:
        """Generate key insight from metrics (safe version)."""
        current_pnl = current.get('total_pnl')
        previous_pnl = previous.get('total_pnl')
        
        if current_pnl is None or previous_pnl is None:
            return "Insufficient data to compare performance."
        
        if previous_pnl == 0:
            if current_pnl > 0:
                return "Performance turned positive from zero baseline."
            elif current_pnl < 0:
                return "Performance turned negative from zero baseline."
            else:
                return "No significant change from previous period."
        
        if current_pnl > previous_pnl:
            change = ((current_pnl - previous_pnl) / abs(previous_pnl)) * 100
            return f"Performance improved by {change:.1f}% compared to previous period."
        else:
            change = ((previous_pnl - current_pnl) / abs(previous_pnl)) * 100
            return f"Performance declined by {change:.1f}% compared to previous period."
    
    def _generate_recommendations(self, metrics: Dict, strategy_comparison: List) -> List[str]:
        """Generate recommendations based on metrics (safe version)."""
        recommendations = []
        
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.5:
            recommendations.append("Consider reviewing entry criteria - win rate below 50%")
        
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < 1.5 and profit_factor != 0:
            recommendations.append("Profit factor is low - consider adjusting risk management")
        
        max_drawdown = metrics.get('max_drawdown', 0)
        if max_drawdown > 15:
            recommendations.append("Max drawdown is high - consider reducing position sizes")
        
        if strategy_comparison:
            best_strategy = max(strategy_comparison, key=lambda x: x.get('pnl', 0))
            recommendations.append(f"Best performing strategy: {best_strategy.get('name')} - consider increasing allocation")
        
        return recommendations or ["Continue with current strategy - metrics look healthy"]
    
    def _generate_executive_summary(self, metrics: Dict, key_insight: str) -> str:
        """Generate executive summary (safe version)."""
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0) * 100
        total_pnl = metrics.get('total_pnl', 0)
        profit_factor = metrics.get('profit_factor', 0)
        
        return f"""
        During this period, the trading system executed {total_trades} trades 
        with a win rate of {win_rate:.1f}%. 
        Total P&L was ${total_pnl:,.2f} with a profit factor of {profit_factor:.2f}. 
        {key_insight}
        """
    
    def _add_to_history(self, report: Dict):
        """Add report to history"""
        self.report_history.append(report)
        if len(self.report_history) > self.max_history:
            self.report_history = self.report_history[-self.max_history:]
    
    def get_recent_reports(self, limit: int = 10) -> List[Dict]:
        """Get recent reports"""
        return self.report_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get report generator status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "reports_generated": len(self.report_history),
            "last_report": self.report_history[-1] if self.report_history else None
        }