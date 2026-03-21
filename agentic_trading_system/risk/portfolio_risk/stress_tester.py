"""
Stress Tester - Simulates portfolio under extreme market conditions
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from agentic_trading_system.utils.logger import logger as logging

class StressTester:
    """
    Stress Tester - Simulates portfolio performance under stress scenarios
    
    Scenarios:
    - Market crash (2008 style)
    - Flash crash
    - Interest rate spike
    - Sector-specific shocks
    - Volatility spikes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default scenarios
        self.scenarios = config.get("scenarios", {
            "market_crash_2008": {"equity": -0.37, "bonds": 0.05},  # S&P 500 dropped 37% in 2008
            "flash_crash_2010": {"equity": -0.09, "bonds": 0.02},   # Flash crash: 9% drop in minutes
            "covid_crash_2020": {"equity": -0.34, "bonds": 0.10},    # COVID crash: 34% drop
            "interest_rate_shock": {"equity": -0.15, "bonds": -0.05}, # Rate hike scenario
            "volatility_spike": {"equity": -0.10, "bonds": 0.03}      # VIX spike scenario
        })
        
        logging.info(f"✅ StressTester initialized with {len(self.scenarios)} scenarios")
    
    def test_portfolio(self, positions: List[Dict], 
                      betas: Dict[str, float] = None,
                      sector_exposures: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Test portfolio under all stress scenarios
        """
        results = {}
        worst_case = {"scenario": None, "loss": 0}
        
        for scenario_name, scenario_returns in self.scenarios.items():
            result = self._apply_scenario(positions, scenario_name, scenario_returns, betas, sector_exposures)
            results[scenario_name] = result
            
            if result["total_loss"] < worst_case["loss"]:  # More negative = worse
                worst_case = {
                    "scenario": scenario_name,
                    "loss": result["total_loss"],
                    "description": result["description"]
                }
        
        # Calculate average stress loss
        avg_loss = np.mean([r["total_loss"] for r in results.values()])
        
        return {
            "scenario_results": results,
            "worst_case": worst_case,
            "average_stress_loss": float(avg_loss),
            "num_scenarios": len(self.scenarios),
            "portfolio_value": sum(p.get("value", 0) for p in positions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_scenario(self, positions: List[Dict], scenario_name: str,
                       scenario_returns: Dict[str, float],
                       betas: Dict[str, float] = None,
                       sector_exposures: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Apply a single stress scenario to portfolio
        """
        total_loss = 0
        position_impacts = []
        
        for position in positions:
            symbol = position.get("symbol")
            value = position.get("value", 0)
            sector = position.get("sector", "unknown")
            
            # Determine asset class impact
            if sector in ["Technology", "Communication Services"]:
                impact = scenario_returns.get("equity", -0.20)
            elif sector in ["Utilities", "Consumer Staples"]:
                impact = scenario_returns.get("bonds", -0.05)  # Defensive sectors
            elif sector in ["Financials"]:
                impact = scenario_returns.get("equity", -0.25)  # Banks hit harder in crashes
            else:
                impact = scenario_returns.get("equity", -0.15)
            
            # Adjust for beta if available
            if betas and symbol in betas:
                beta = betas[symbol]
                impact = impact * beta
            
            loss = value * impact
            total_loss += loss
            
            position_impacts.append({
                "symbol": symbol,
                "value": value,
                "impact_percent": float(impact * 100),
                "loss": float(loss),
                "sector": sector
            })
        
        # Sort by worst impact
        position_impacts.sort(key=lambda x: x["loss"])
        
        return {
            "scenario": scenario_name,
            "description": self._get_scenario_description(scenario_name),
            "total_loss": float(total_loss),
            "loss_percent": float(total_loss / sum(p.get("value", 0) for p in positions) * 100 if positions else 0),
            "position_impacts": position_impacts[:10],  # Top 10 worst
            "scenario_parameters": scenario_returns
        }
    
    def _get_scenario_description(self, scenario_name: str) -> str:
        """Get description for each scenario"""
        descriptions = {
            "market_crash_2008": "2008 Financial Crisis - severe market downturn",
            "flash_crash_2010": "2010 Flash Crash - rapid decline and recovery",
            "covid_crash_2020": "2020 COVID-19 Crash - pandemic selloff",
            "interest_rate_shock": "Interest Rate Shock - rapid rate increase",
            "volatility_spike": "Volatility Spike - VIX surge"
        }
        return descriptions.get(scenario_name, scenario_name)
    
    def add_scenario(self, name: str, returns: Dict[str, float]):
        """Add a custom stress scenario"""
        self.scenarios[name] = returns
        logging.info(f"➕ Added stress scenario: {name}")
    
    def remove_scenario(self, name: str):
        """Remove a stress scenario"""
        if name in self.scenarios:
            del self.scenarios[name]
            logging.info(f"➖ Removed stress scenario: {name}")