"""
HTML Reporter - Generates HTML performance reports
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import base64
from utils.logger import logger as logging

class HTMLReporter:
    """
    HTML Reporter - Generates interactive HTML performance reports
    
    Features:
    - Interactive charts
    - Performance tables
    - Metric dashboards
    - Export functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Output directory
        self.output_dir = config.get("output_dir", "reports")
        self.template_dir = config.get("template_dir", "templates")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)
        
        logging.info(f"✅ HTMLReporter initialized")
    
    def generate_report(self, metrics: Dict[str, Any], 
                       plots: List[str] = None,
                       title: str = "Trading Performance Report",
                       filename: str = None) -> str:
        """
        Generate HTML performance report
        """
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert plots to base64 for embedding
        plot_data = []
        if plots:
            for plot_path in plots:
                if os.path.exists(plot_path):
                    with open(plot_path, 'rb') as f:
                        plot_data.append({
                            'name': os.path.basename(plot_path),
                            'data': base64.b64encode(f.read()).decode('utf-8')
                        })
        
        # Generate HTML
        html = self._generate_html(metrics, plot_data, title)
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        logging.info(f"💾 Saved HTML report to {filepath}")
        
        return filepath
    
    def _generate_html(self, metrics: Dict[str, Any], 
                      plots: List[Dict], title: str) -> str:
        """
        Generate HTML content
        """
        # Extract key metrics
        summary = metrics.get('summary', {})
        pnl = metrics.get('pnl', {})
        risk = metrics.get('risk_adjusted', {})
        trading_stats = metrics.get('trading_stats', {})
        
        # Format numbers
        total_pnl = pnl.get('total_pnl', 0)
        sharpe = risk.get('sharpe_ratio', {}).get('annualized_sharpe', 0)
        win_rate = trading_stats.get('win_rate', {}).get('win_rate', 0) * 100
        profit_factor = trading_stats.get('profit_factor', {}).get('profit_factor', 0)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #666;
                    margin-top: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .metric-card.green {{
                    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                    color: #333;
                }}
                .metric-card.blue {{
                    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
                    color: #333;
                }}
                .metric-card.orange {{
                    background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
                    color: #333;
                }}
                .metric-value {{
                    font-size: 32px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .positive {{
                    color: #4CAF50;
                }}
                .negative {{
                    color: #f44336;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .plot-container {{
                    margin-top: 30px;
                    text-align: center;
                }}
                .plot-image {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 5px;
                }}
                .timestamp {{
                    text-align: right;
                    color: #999;
                    font-size: 12px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                
                <h2>Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card green">
                        <div class="metric-label">Total P&L</div>
                        <div class="metric-value {'positive' if total_pnl > 0 else 'negative'}">${total_pnl:,.2f}</div>
                    </div>
                    <div class="metric-card blue">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{sharpe:.2f}</div>
                    </div>
                    <div class="metric-card orange">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{win_rate:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{profit_factor:.2f}</div>
                    </div>
                </div>
                
                <h2>Detailed Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>{summary.get('total_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td>{pnl.get('realized', {}).get('winning_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td>{pnl.get('realized', {}).get('losing_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Gross Profit</td>
                        <td class="positive">${pnl.get('realized', {}).get('gross_profit', 0):,.2f}</td>
                    </tr>
                    <tr>
                        <td>Gross Loss</td>
                        <td class="negative">${pnl.get('realized', {}).get('gross_loss', 0):,.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td class="negative">{metrics.get('drawdown', {}).get('max_drawdown', {}).get('max_drawdown_pct', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td>{risk.get('sortino_ratio', {}).get('annualized_sortino', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{risk.get('calmar_ratio', {}).get('calmar_ratio', 0):.2f}</td>
                    </tr>
                </table>
        """
        
        # Add plots
        if plots:
            html += "<h2>Performance Charts</h2>"
            for plot in plots:
                html += f"""
                <div class="plot-container">
                    <h3>{plot['name']}</h3>
                    <img class="plot-image" src="data:image/png;base64,{plot['data']}" alt="{plot['name']}">
                </div>
                """
        
        # Add JSON data for debugging
        html += f"""
                <h2>Raw Data</h2>
                <pre style="background-color: #f5f5f5; padding: 10px; overflow: auto; max-height: 300px;">
{json.dumps(metrics, indent=2, default=str)}
                </pre>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_daily_digest(self, daily_metrics: Dict[str, Any],
                             filename: str = None) -> str:
        """
        Generate daily digest report
        """
        date = datetime.now().strftime('%Y-%m-%d')
        title = f"Daily Trading Digest - {date}"
        
        return self.generate_report(daily_metrics, title=title, filename=filename)
    
    def generate_monthly_report(self, monthly_metrics: Dict[str, Any],
                               filename: str = None) -> str:
        """
        Generate monthly report
        """
        month = datetime.now().strftime('%Y-%m')
        title = f"Monthly Performance Report - {month}"
        
        return self.generate_report(monthly_metrics, title=title, filename=filename)