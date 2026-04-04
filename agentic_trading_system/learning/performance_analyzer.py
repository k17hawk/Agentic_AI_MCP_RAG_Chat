
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from collections import defaultdict

class PerformanceAnalyzer:
    """Analyzes trading performance and provides insights for learning"""
    
    def __init__(self, data_path: str = "discovery_outputs"):
        self.data_path = Path(data_path)
        self.performance_data = {}
        
    def load_trades(self, days_back: int = 30) -> pd.DataFrame:
        """Load trade data from artifacts"""
        trades = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for artifact in self.data_path.rglob("*.json"):
            try:
                with open(artifact) as f:
                    data = json.load(f)
                
                # Extract timestamp from folder structure
                timestamp_str = artifact.parent.name if artifact.parent.name != "discovery_outputs" else artifact.stem
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if timestamp >= cutoff_date:
                        trade_data = self._extract_trade_data(data, timestamp)
                        if trade_data:
                            trades.append(trade_data)
                except:
                    pass
            except Exception as e:
                print(f"Error loading {artifact}: {e}")
        
        df = pd.DataFrame(trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _extract_trade_data(self, data: Dict, timestamp: datetime) -> Optional[Dict]:
        """Extract relevant trade data from artifact"""
        trade = {
            'timestamp': timestamp,
            'ticker': data.get('ticker', 'UNKNOWN'),
        }
        
        # Extract analysis scores
        if 'analysis' in data:
            analysis = data['analysis']
            trade['technical_score'] = analysis.get('technical_score', 0)
            trade['sentiment_score'] = analysis.get('sentiment_score', 0)
            trade['fundamental_score'] = analysis.get('fundamental_score', 0)
            trade['final_score'] = analysis.get('final_score', 0)
        
        # Extract recommendation
        if 'portfolio' in data:
            trade['recommendation'] = data['portfolio'].get('action', 'HOLD')
            trade['confidence'] = data['portfolio'].get('confidence', 0)
        
        # Extract execution outcome
        if 'executed' in data:
            trade['executed'] = data['executed']
            trade['quantity'] = data.get('quantity', 0)
            trade['price'] = data.get('price', 0)
        
        # Extract PnL if available
        if 'pnl' in data:
            trade['pnl'] = data['pnl']
            trade['pnl_percent'] = data.get('pnl_percent', 0)
        
        return trade if any([trade.get('final_score', 0) > 0, 'pnl' in trade]) else None
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if df.empty:
            return {'error': 'No data available'}
        
        metrics = {}
        
        # Basic metrics
        metrics['total_trades'] = len(df)
        metrics['start_date'] = df['timestamp'].min()
        metrics['end_date'] = df['timestamp'].max()
        
        # PnL metrics (if available)
        if 'pnl' in df.columns:
            profitable = df[df['pnl'] > 0]
            losing = df[df['pnl'] < 0]
            
            metrics['total_pnl'] = df['pnl'].sum()
            metrics['avg_pnl'] = df['pnl'].mean()
            metrics['win_rate'] = len(profitable) / len(df) if len(df) > 0 else 0
            metrics['avg_win'] = profitable['pnl'].mean() if len(profitable) > 0 else 0
            metrics['avg_loss'] = losing['pnl'].mean() if len(losing) > 0 else 0
            metrics['profit_factor'] = abs(profitable['pnl'].sum() / losing['pnl'].sum()) if losing['pnl'].sum() != 0 else float('inf')
            metrics['largest_win'] = df['pnl'].max()
            metrics['largest_loss'] = df['pnl'].min()
        
        # Signal effectiveness
        for signal in ['technical_score', 'sentiment_score', 'fundamental_score', 'final_score']:
            if signal in df.columns:
                metrics[f'{signal}_avg'] = df[signal].mean()
                if 'pnl' in df.columns:
                    # Correlation with PnL
                    metrics[f'{signal}_pnl_correlation'] = df[signal].corr(df['pnl']) if len(df) > 1 else 0
                    
                    # Success rate by signal level
                    high_signal = df[df[signal] > df[signal].median()]
                    low_signal = df[df[signal] <= df[signal].median()]
                    
                    metrics[f'{signal}_high_win_rate'] = (high_signal['pnl'] > 0).mean() if len(high_signal) > 0 else 0
                    metrics[f'{signal}_low_win_rate'] = (low_signal['pnl'] > 0).mean() if len(low_signal) > 0 else 0
        
        # Time-based analysis
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            if 'pnl' in df.columns:
                metrics['best_hour'] = df.groupby('hour')['pnl'].mean().idxmax()
                metrics['best_day'] = df.groupby('day_of_week')['pnl'].mean().idxmax()
        
        # Recommendation accuracy
        if 'recommendation' in df.columns and 'pnl' in df.columns:
            buy_trades = df[df['recommendation'] == 'BUY']
            sell_trades = df[df['recommendation'] == 'SELL']
            
            metrics['buy_win_rate'] = (buy_trades['pnl'] > 0).mean() if len(buy_trades) > 0 else 0
            metrics['sell_win_rate'] = (sell_trades['pnl'] < 0).mean() if len(sell_trades) > 0 else 0  # For shorts
            metrics['recommendation_accuracy'] = (
                metrics['buy_win_rate'] * len(buy_trades) + metrics['sell_win_rate'] * len(sell_trades)
            ) / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        return metrics
    
    def analyze_signal_effectiveness(self, df: pd.DataFrame) -> Dict:
        """Analyze which signals are most predictive"""
        if df.empty or 'pnl' not in df.columns:
            return {'error': 'No PnL data available'}
        
        signal_analysis = {}
        
        for signal in ['technical_score', 'sentiment_score', 'fundamental_score', 'final_score']:
            if signal not in df.columns:
                continue
            
            # Split into quartiles
            df[f'{signal}_quartile'] = pd.qcut(df[signal], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            quartile_performance = df.groupby(f'{signal}_quartile')['pnl'].agg(['mean', 'count', lambda x: (x > 0).mean()])
            quartile_performance.columns = ['avg_pnl', 'count', 'win_rate']
            
            signal_analysis[signal] = {
                'correlation_with_pnl': df[signal].corr(df['pnl']),
                'best_quartile': quartile_performance['avg_pnl'].idxmax(),
                'quartile_performance': quartile_performance.to_dict(),
                'predictive_power': abs(df[signal].corr(df['pnl']))
            }
        
        return signal_analysis
    
    def identify_improvement_opportunities(self, df: pd.DataFrame, metrics: Dict) -> List[Dict]:
        """Identify specific areas for improvement"""
        opportunities = []
        
        if df.empty:
            return opportunities
        
        # Check for low win rate
        if metrics.get('win_rate', 0) < 0.4:
            opportunities.append({
                'area': 'signal_quality',
                'priority': 'high',
                'description': 'Win rate is below 40%. Consider adjusting analysis weights.',
                'recommended_action': 'run_weight_optimization',
                'expected_impact': 'Increase win rate by 10-20%'
            })
        
        # Check for poor signal correlation
        for signal in ['technical_score', 'sentiment_score', 'fundamental_score']:
            correlation = metrics.get(f'{signal}_pnl_correlation', 0)
            if abs(correlation) < 0.1:
                opportunities.append({
                    'area': signal.replace('_score', ''),
                    'priority': 'medium',
                    'description': f'{signal.replace("_score", "").title()} signal shows weak correlation with PnL ({correlation:.2f})',
                    'recommended_action': 'recalibrate_signal_weights',
                    'expected_impact': 'Better alignment with actual outcomes'
                })
        
        # Check for recommendation accuracy
        if metrics.get('recommendation_accuracy', 0) < 0.5:
            opportunities.append({
                'area': 'recommendation_engine',
                'priority': 'high',
                'description': 'Recommendation accuracy is below 50%',
                'recommended_action': 'retrain_portfolio_optimizer',
                'expected_impact': 'More profitable trade recommendations'
            })
        
        # Check for consistent losses in specific conditions
        if 'hour' in df.columns and 'pnl' in df.columns:
            hourly_pnl = df.groupby('hour')['pnl'].mean()
            worst_hours = hourly_pnl[hourly_pnl < 0].index.tolist()
            if worst_hours:
                opportunities.append({
                    'area': 'timing',
                    'priority': 'medium',
                    'description': f'Consistent losses during hours: {worst_hours}',
                    'recommended_action': 'adjust_trading_hours_or_strategy',
                    'expected_impact': 'Avoid predictable loss periods'
                })
        
        return opportunities
    
    def generate_learning_recommendations(self, metrics: Dict, signal_analysis: Dict, opportunities: List) -> Dict:
        """Generate actionable learning recommendations"""
        recommendations = {
            'immediate_actions': [],
            'short_term_actions': [],  # 1-7 days
            'long_term_actions': [],    # 2-4 weeks
            'learning_priority': 'low'
        }
        
        # Determine learning priority
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.3:
            recommendations['learning_priority'] = 'critical'
        elif win_rate < 0.45:
            recommendations['learning_priority'] = 'high'
        elif win_rate < 0.55:
            recommendations['learning_priority'] = 'medium'
        else:
            recommendations['learning_priority'] = 'low'
        
        # Immediate actions (fix critical issues)
        for opp in opportunities:
            if opp['priority'] == 'high':
                recommendations['immediate_actions'].append(opp)
        
        # Short-term actions
        if metrics.get('total_trades', 0) < 20:
            recommendations['short_term_actions'].append({
                'action': 'collect_more_data',
                'description': f'Only {metrics.get("total_trades", 0)} trades analyzed. Need more data for reliable learning.',
                'target': 50
            })
        
        # Check if specific signals need tuning
        for signal, analysis in signal_analysis.items():
            if analysis.get('predictive_power', 0) < 0.1:
                recommendations['short_term_actions'].append({
                    'action': 'recalibrate_signal',
                    'signal': signal,
                    'description': f'{signal} signal has low predictive power',
                    'method': 'bayesian_update'
                })
        
        # Long-term actions
        if metrics.get('profit_factor', 0) < 1.2:
            recommendations['long_term_actions'].append({
                'action': 'strategy_refinement',
                'description': 'Profit factor below 1.2 indicates strategy needs fundamental improvement',
                'method': 'genetic_algorithm_tuning'
            })
        
        return recommendations
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate complete performance analysis report"""
        if df.empty:
            return {
                'status': 'insufficient_data',
                'message': 'No trading data available for analysis',
                'recommendations': {'immediate_actions': [{'action': 'collect_data', 'description': 'Run trading system to generate data'}]}
            }
        
        # Calculate all metrics
        metrics = self.calculate_performance_metrics(df)
        signal_analysis = self.analyze_signal_effectiveness(df)
        opportunities = self.identify_improvement_opportunities(df, metrics)
        recommendations = self.generate_learning_recommendations(metrics, signal_analysis, opportunities)
        
        # Create summary
        summary = {
            'status': 'success',
            'analysis_date': datetime.now().isoformat(),
            'data_period': f"{metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}",
            'metrics': metrics,
            'signal_analysis': signal_analysis,
            'opportunities': opportunities,
            'recommendations': recommendations,
            'summary_text': self._generate_summary_text(metrics, recommendations)
        }
        
        return summary
    
    def _generate_summary_text(self, metrics: Dict, recommendations: Dict) -> str:
        """Generate human-readable summary text"""
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        
        if total_trades == 0:
            return "No trades have been executed yet. Run the trading system to generate data."
        
        summary = f"Analyzed {total_trades} trades. "
        
        if win_rate > 0.5:
            summary += f"Win rate is strong at {win_rate:.1%}. "
        elif win_rate > 0.4:
            summary += f"Win rate is moderate at {win_rate:.1%}. Room for improvement. "
        else:
            summary += f"Win rate is low at {win_rate:.1%}. Immediate learning recommended. "
        
        if metrics.get('profit_factor', 0) > 1.5:
            summary += "Profit factor is excellent. "
        elif metrics.get('profit_factor', 0) > 1:
            summary += "Profit factor is positive but could be improved. "
        else:
            summary += "Profit factor is negative. Strategy needs adjustment. "
        
        priority = recommendations.get('learning_priority', 'low')
        summary += f"Learning priority: {priority.upper()}. "
        
        if recommendations.get('immediate_actions'):
            summary += f"{len(recommendations['immediate_actions'])} immediate actions recommended."
        
        return summary
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "charts/performance"):
        """Create performance visualization charts"""
        if df.empty:
            print("No data to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. PnL over time
        if 'pnl' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            df['cumulative_pnl'] = df['pnl'].cumsum()
            ax.plot(df['timestamp'], df['cumulative_pnl'], marker='o', linewidth=2)
            ax.set_title('Cumulative PnL Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative PnL')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'cumulative_pnl.png')
            plt.close()
        
        # 2. Signal effectiveness
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        signals = ['technical_score', 'sentiment_score', 'fundamental_score']
        
        for idx, signal in enumerate(signals):
            if signal in df.columns and 'pnl' in df.columns:
                ax = axes[idx // 2, idx % 2]
                ax.scatter(df[signal], df['pnl'], alpha=0.6)
                ax.set_xlabel(signal)
                ax.set_ylabel('PnL')
                ax.set_title(f'{signal} vs PnL')
                ax.grid(True, alpha=0.3)
        
        # 3. Win rate by signal quartile
        if all(s in df.columns for s in signals):
            ax = axes[1, 1]
            for signal in signals:
                df[f'{signal}_quartile'] = pd.qcut(df[signal], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                quartile_win_rate = df.groupby(f'{signal}_quartile').apply(lambda x: (x['pnl'] > 0).mean() if 'pnl' in x else 0)
                ax.plot(quartile_win_rate.values, label=signal, marker='o')
            ax.set_xlabel('Quartile (Q1=Lowest, Q4=Highest)')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate by Signal Strength')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'signal_analysis.png')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def print_report(self, report: Dict):
        """Print a formatted report to console"""
        print("\n" + "="*80)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        if report.get('status') != 'success':
            print(f"⚠️  {report.get('message', 'Analysis failed')}")
            return
        
        print(f"\n📈 Summary: {report['summary_text']}")
        print(f"\n📅 Period: {report['data_period']}")
        
        # Key metrics
        metrics = report['metrics']
        print(f"\n🎯 Key Metrics:")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Total PnL: {metrics.get('total_pnl', 0):.2f}")
        
        if 'avg_win' in metrics and 'avg_loss' in metrics:
            print(f"   Avg Win: {metrics.get('avg_win', 0):.2f}")
            print(f"   Avg Loss: {metrics.get('avg_loss', 0):.2f}")
        
        # Signal analysis
        if report['signal_analysis']:
            print(f"\n🔍 Signal Analysis:")
            for signal, analysis in report['signal_analysis'].items():
                if 'correlation_with_pnl' in analysis:
                    print(f"   {signal}: correlation = {analysis['correlation_with_pnl']:.3f}")
        
        # Recommendations
        recs = report['recommendations']
        print(f"\n💡 Learning Priority: {recs['learning_priority'].upper()}")
        
        if recs['immediate_actions']:
            print(f"\n🚨 Immediate Actions Required:")
            for action in recs['immediate_actions']:
                print(f"   • {action['description']}")
                print(f"     → {action['recommended_action']}")
        
        if recs['short_term_actions']:
            print(f"\n📋 Short-term Actions (1-7 days):")
            for action in recs['short_term_actions']:
                print(f"   • {action['description']}")
        
        if recs['long_term_actions']:
            print(f"\n🎯 Long-term Actions (2-4 weeks):")
            for action in recs['long_term_actions']:
                print(f"   • {action['description']}")
        
        print("\n" + "="*80 + "\n")

# For running standalone analysis
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # Load trades
    df = analyzer.load_trades(days_back=30)
    
    if not df.empty:
        # Generate report
        report = analyzer.generate_report(df)
        
        # Print to console
        analyzer.print_report(report)
        
        # Create visualizations
        analyzer.create_visualizations(df)
        
        # Save report to file
        report_path = Path("learning_results/performance_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Full report saved to {report_path}")
    else:
        print("No trading data found. Run the trading system first to generate data.")