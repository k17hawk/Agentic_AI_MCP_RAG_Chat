"""
Plot Generator - Creates visualizations for performance analysis
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from utils.logger import logger as logging

class PlotGenerator:
    """
    Plot Generator - Creates visualizations for performance analysis
    
    Features:
    - Equity curves
    - Drawdown charts
    - Return distributions
    - Performance attribution
    - Risk metrics visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Plot settings
        self.style = config.get("style", "seaborn-darkgrid")
        self.figsize = config.get("figsize", (12, 6))
        self.dpi = config.get("dpi", 100)
        self.output_dir = config.get("output_dir", "charts")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use(self.style)
        sns.set_palette("husl")
        
        logging.info(f"✅ PlotGenerator initialized")
    
    def plot_equity_curve(self, equity_curve: List[float], 
                          dates: List[datetime] = None,
                          title: str = "Equity Curve",
                          filename: str = None) -> str:
        """
        Plot equity curve
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = dates if dates else range(len(equity_curve))
        
        ax.plot(x, equity_curve, linewidth=2, color='blue', label='Portfolio')
        
        # Add horizontal line at initial equity
        ax.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
        
        # Fill below
        ax.fill_between(x, equity_curve, equity_curve[0], 
                        where=(np.array(equity_curve) > equity_curve[0]),
                        color='green', alpha=0.3, label='Above Initial')
        ax.fill_between(x, equity_curve, equity_curve[0],
                        where=(np.array(equity_curve) <= equity_curve[0]),
                        color='red', alpha=0.3, label='Below Initial')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates else 'Period')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "equity_curve.png")
    
    def plot_drawdown(self, equity_curve: List[float],
                     dates: List[datetime] = None,
                     title: str = "Drawdown Chart",
                     filename: str = None) -> str:
        """
        Plot drawdown chart
        """
        from analytics.performance_metrics.max_drawdown import MaxDrawdown
        dd_calc = MaxDrawdown(self.config)
        
        drawdowns = dd_calc.calculate_drawdown_curve(equity_curve)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        x = dates if dates else range(len(equity_curve))
        
        # Equity curve
        ax1.plot(x, equity_curve, linewidth=2, color='blue')
        ax1.set_title('Equity Curve', fontsize=12)
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(x, 0, drawdowns, color='red', alpha=0.5)
        ax2.set_title(title, fontsize=12)
        ax2.set_xlabel('Date' if dates else 'Period')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Mark maximum drawdown
        max_dd_idx = np.argmin(drawdowns)
        ax2.plot(x[max_dd_idx], drawdowns[max_dd_idx], 'ro', markersize=8,
                label=f'Max DD: {drawdowns[max_dd_idx]:.1f}%')
        ax2.legend()
        
        if dates:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "drawdown_chart.png")
    
    def plot_return_distribution(self, returns: List[float],
                                 title: str = "Return Distribution",
                                 filename: str = None) -> str:
        """
        Plot return distribution histogram
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]))
        
        # Histogram
        ax1.hist(returns, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(returns)*100:.2f}%')
        ax1.set_title('Return Distribution')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(returns, vert=True)
        ax2.set_title('Return Box Plot')
        ax2.set_ylabel('Return')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = (
            f"Mean: {np.mean(returns)*100:.2f}%\n"
            f"Std: {np.std(returns)*100:.2f}%\n"
            f"Skew: {float(np.mean((returns - np.mean(returns))**3) / np.std(returns)**3):.2f}\n"
            f"Kurtosis: {float(np.mean((returns - np.mean(returns))**4) / np.std(returns)**4 - 3):.2f}"
        )
        ax2.text(1.1, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "return_distribution.png")
    
    def plot_rolling_sharpe(self, returns: List[float],
                           window: int = 60,
                           dates: List[datetime] = None,
                           title: str = "Rolling Sharpe Ratio",
                           filename: str = None) -> str:
        """
        Plot rolling Sharpe ratio
        """
        from analytics.performance_metrics.sharpe_ratio import SharpeRatio
        sharpe_calc = SharpeRatio(self.config)
        
        rolling_sharpes = []
        rolling_dates = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            sharpe = sharpe_calc.calculate(window_returns)
            rolling_sharpes.append(sharpe.get('annualized_sharpe', 0))
            if dates:
                rolling_dates.append(dates[i])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = rolling_dates if dates else range(len(rolling_sharpes))
        
        ax.plot(x, rolling_sharpes, linewidth=2, color='purple')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (1.0)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates else 'Period')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "rolling_sharpe.png")
    
    def plot_monthly_returns_heatmap(self, returns: List[float],
                                     dates: List[datetime],
                                     title: str = "Monthly Returns Heatmap",
                                     filename: str = None) -> str:
        """
        Plot monthly returns heatmap
        """
        # Group returns by year and month
        years = []
        months = []
        monthly_returns = []
        
        for ret, date in zip(returns, dates):
            years.append(date.year)
            months.append(date.month)
            monthly_returns.append(ret * 100)  # Convert to percentage
        
        # Create pivot table
        import pandas as pd
        df = pd.DataFrame({
            'Year': years,
            'Month': months,
            'Return': monthly_returns
        })
        
        pivot = df.pivot_table(values='Return', index='Month', columns='Year', aggfunc='sum')
        
        fig, ax = plt.subplots(figsize=(self.figsize[0]*1.2, self.figsize[1]))
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(month_names)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "monthly_returns_heatmap.png")
    
    def plot_rolling_volatility(self, returns: List[float],
                               window: int = 20,
                               dates: List[datetime] = None,
                               title: str = "Rolling Volatility",
                               filename: str = None) -> str:
        """
        Plot rolling volatility
        """
        rolling_vol = []
        rolling_dates = []
        
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i]) * np.sqrt(252) * 100
            rolling_vol.append(vol)
            if dates:
                rolling_dates.append(dates[i])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = rolling_dates if dates else range(len(rolling_vol))
        
        ax.plot(x, rolling_vol, linewidth=2, color='orange')
        ax.axhline(y=np.mean(rolling_vol), color='red', linestyle='--',
                  label=f'Avg: {np.mean(rolling_vol):.1f}%')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates else 'Period')
        ax.set_ylabel('Volatility (% Annualized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "rolling_volatility.png")
    
    def plot_correlation_matrix(self, returns_dict: Dict[str, List[float]],
                               title: str = "Asset Correlation Matrix",
                               filename: str = None) -> str:
        """
        Plot correlation matrix of multiple assets
        """
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(returns_dict)
        
        # Calculate correlation
        corr = df.corr()
        
        fig, ax = plt.subplots(figsize=(self.figsize[0]*1.2, self.figsize[1]*1.2))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr), k=1)
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "correlation_matrix.png")
    
    def plot_performance_attribution(self, attribution_data: Dict[str, float],
                                    title: str = "Performance Attribution",
                                    filename: str = None) -> str:
        """
        Plot performance attribution as bar chart
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        items = list(attribution_data.keys())
        values = list(attribution_data.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(items, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, v in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{v:.1f}%', ha='center', va='bottom' if v > 0 else 'top')
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Contribution (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "performance_attribution.png")
    
    def plot_drawdown_periods(self, equity_curve: List[float],
                             dates: List[datetime] = None,
                             title: str = "Drawdown Periods",
                             filename: str = None) -> str:
        """
        Plot drawdown periods with recovery
        """
        from analytics.performance_metrics.max_drawdown import MaxDrawdown
        dd_calc = MaxDrawdown(self.config)
        
        drawdowns = dd_calc.calculate_drawdown_curve(equity_curve)
        periods = dd_calc.find_drawdown_periods(equity_curve, threshold=5)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = dates if dates else range(len(equity_curve))
        
        # Plot equity curve
        ax.plot(x, equity_curve, linewidth=2, color='blue', alpha=0.7, label='Equity')
        
        # Shade drawdown periods
        for period in periods:
            start = period['start_index']
            end = period.get('end_index', len(equity_curve) - 1)
            ax.axvspan(x[start], x[end], alpha=0.2, color='red')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date' if dates else 'Period')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, filename or "drawdown_periods.png")
    
    def _save_plot(self, fig, filename: str) -> str:
        """
        Save plot to file
        """
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"💾 Saved plot to {filepath}")
        return filepath