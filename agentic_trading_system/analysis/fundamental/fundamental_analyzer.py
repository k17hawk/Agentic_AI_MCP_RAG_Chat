"""
Fundamental Analyzer - Main orchestrator for fundamental analysis
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np

from utils.logger import logger as logging
from agents.base_agent import BaseAgent, AgentMessage

# Import all fundamental analyzers

from analysis.fundamental.valuation import ValuationAnalyzer
from analysis.fundamental.growth import GrowthAnalyzer
from analysis.fundamental.profitability import ProfitabilityAnalyzer
from analysis.fundamental.liquidity import LiquidityAnalyzer
from analysis.fundamental.solvency import SolvencyAnalyzer
from analysis.fundamental.efficiency import EfficiencyAnalyzer
from analysis.fundamental.discounted_cash_flow import DCFAnalyzer
from analysis.fundamental.fundamental_scorer import FundamentalScorer

class FundamentalAnalyzer(BaseAgent):
    """
    Main fundamental analysis agent
    Coordinates all fundamental analysis components
    
    Analysis Categories:
    - Valuation (P/E, P/B, P/S, EV/EBITDA)
    - Growth (Revenue, EPS, EBITDA growth)
    - Profitability (ROE, ROA, Margins)
    - Liquidity (Current Ratio, Quick Ratio)
    - Solvency (D/E, Interest Coverage)
    - Efficiency (Asset Turnover, Inventory Turnover)
    - DCF Valuation (Intrinsic Value)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Comprehensive fundamental analysis of companies",
            config=config
        )
        
        # Initialize all analyzers
        self.valuation = ValuationAnalyzer(config.get("valuation_config", {}))
        self.growth = GrowthAnalyzer(config.get("growth_config", {}))
        self.profitability = ProfitabilityAnalyzer(config.get("profitability_config", {}))
        self.liquidity = LiquidityAnalyzer(config.get("liquidity_config", {}))
        self.solvency = SolvencyAnalyzer(config.get("solvency_config", {}))
        self.efficiency = EfficiencyAnalyzer(config.get("efficiency_config", {}))
        self.dcf = DCFAnalyzer(config.get("dcf_config", {}))
        self.scorer = FundamentalScorer(config.get("scorer_config", {}))
        
        # Cache for fundamental data
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)  # Update daily
        
        # Sector benchmarks
        self.sector_benchmarks = self._load_sector_benchmarks()
        
        logging.info(f"✅ FundamentalAnalyzer initialized")
    
    def _load_sector_benchmarks(self) -> Dict[str, Dict]:
        """Load sector-specific benchmark ratios"""
        return {
            "Technology": {
                "pe_benchmark": 25,
                "pb_benchmark": 5,
                "ps_benchmark": 3,
                "roe_benchmark": 20,
                "de_benchmark": 0.5,
                "current_ratio_benchmark": 2.0
            },
            "Healthcare": {
                "pe_benchmark": 22,
                "pb_benchmark": 4,
                "ps_benchmark": 2.5,
                "roe_benchmark": 15,
                "de_benchmark": 0.6,
                "current_ratio_benchmark": 1.8
            },
            "Financial Services": {
                "pe_benchmark": 15,
                "pb_benchmark": 1.5,
                "ps_benchmark": 2,
                "roe_benchmark": 12,
                "de_benchmark": 2.0,
                "current_ratio_benchmark": 1.2
            },
            "Consumer Cyclical": {
                "pe_benchmark": 20,
                "pb_benchmark": 3,
                "ps_benchmark": 1.5,
                "roe_benchmark": 18,
                "de_benchmark": 0.8,
                "current_ratio_benchmark": 1.5
            },
            "Industrials": {
                "pe_benchmark": 18,
                "pb_benchmark": 2.5,
                "ps_benchmark": 1.2,
                "roe_benchmark": 15,
                "de_benchmark": 0.7,
                "current_ratio_benchmark": 1.6
            },
            "Energy": {
                "pe_benchmark": 12,
                "pb_benchmark": 1.2,
                "ps_benchmark": 0.8,
                "roe_benchmark": 10,
                "de_benchmark": 1.0,
                "current_ratio_benchmark": 1.3
            },
            "Utilities": {
                "pe_benchmark": 16,
                "pb_benchmark": 1.5,
                "ps_benchmark": 1.0,
                "roe_benchmark": 8,
                "de_benchmark": 1.5,
                "current_ratio_benchmark": 1.1
            },
            "Real Estate": {
                "pe_benchmark": 18,
                "pb_benchmark": 1.8,
                "ps_benchmark": 4,
                "roe_benchmark": 8,
                "de_benchmark": 1.2,
                "current_ratio_benchmark": 1.0
            },
            "Communication Services": {
                "pe_benchmark": 22,
                "pb_benchmark": 3,
                "ps_benchmark": 2,
                "roe_benchmark": 18,
                "de_benchmark": 0.8,
                "current_ratio_benchmark": 1.5
            },
            "Consumer Defensive": {
                "pe_benchmark": 20,
                "pb_benchmark": 3.5,
                "ps_benchmark": 1.8,
                "roe_benchmark": 25,
                "de_benchmark": 0.5,
                "current_ratio_benchmark": 1.8
            },
            "Basic Materials": {
                "pe_benchmark": 15,
                "pb_benchmark": 1.8,
                "ps_benchmark": 1.2,
                "roe_benchmark": 12,
                "de_benchmark": 0.6,
                "current_ratio_benchmark": 1.7
            }
        }
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process fundamental analysis requests
        """
        if message.message_type == "analysis_request":
            analysis_id = message.content.get("analysis_id")
            symbol = message.content.get("symbol")
            
            # Perform fundamental analysis
            score, details = await self.analyze(symbol)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="analysis_result",
                content={
                    "analysis_id": analysis_id,
                    "agent": self.name,
                    "score": score,
                    "details": details
                }
            )
        
        return None
    
    async def analyze(self, symbol: str) -> tuple[float, Dict]:
        """
        Perform comprehensive fundamental analysis
        """
        logging.info(f"📊 Analyzing fundamentals for {symbol}")
        
        # Check cache
        cache_key = f"fundamental_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached fundamental data for {symbol}")
                return cached_result["score"], cached_result["details"]
        
        try:
            # Fetch company data
            info = await self._fetch_company_info(symbol)
            
            if not info:
                return 0.5, {"error": "No fundamental data available"}
            
            # Get sector benchmarks
            sector = info.get("sector", "Unknown")
            benchmarks = self.sector_benchmarks.get(sector, self.sector_benchmarks.get("Technology"))
            
            # Run all analyses in parallel
            tasks = [
                self.valuation.analyze(info, benchmarks),
                self.growth.analyze(info, benchmarks),
                self.profitability.analyze(info, benchmarks),
                self.liquidity.analyze(info, benchmarks),
                self.solvency.analyze(info, benchmarks),
                self.efficiency.analyze(info, benchmarks),
                self.dcf.analyze(symbol, info)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            valuation_result = results[0] if not isinstance(results[0], Exception) else {"score": 0.5, "details": {}}
            growth_result = results[1] if not isinstance(results[1], Exception) else {"score": 0.5, "details": {}}
            profitability_result = results[2] if not isinstance(results[2], Exception) else {"score": 0.5, "details": {}}
            liquidity_result = results[3] if not isinstance(results[3], Exception) else {"score": 0.5, "details": {}}
            solvency_result = results[4] if not isinstance(results[4], Exception) else {"score": 0.5, "details": {}}
            efficiency_result = results[5] if not isinstance(results[5], Exception) else {"score": 0.5, "details": {}}
            dcf_result = results[6] if not isinstance(results[6], Exception) else {"score": 0.5, "details": {}}
            
            # Combine all results
            analysis_results = {
                "valuation": valuation_result,
                "growth": growth_result,
                "profitability": profitability_result,
                "liquidity": liquidity_result,
                "solvency": solvency_result,
                "efficiency": efficiency_result,
                "dcf": dcf_result
            }
            
            # Calculate overall score using scorer
            final_score, details = self.scorer.calculate_score(
                analysis_results,
                sector,
                benchmarks
            )
            
            # Add company info to details
            details["company_info"] = {
                "name": info.get("longName", symbol),
                "sector": sector,
                "industry": info.get("industry", "Unknown"),
                "country": info.get("country", "Unknown"),
                "website": info.get("website", ""),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0))
            }
            
            result = {
                "score": final_score,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            logging.info(f"✅ Fundamental analysis complete for {symbol}: score={final_score:.2f}")
            
            return final_score, details
            
        except Exception as e:
            logging.error(f"Error in fundamental analysis for {symbol}: {e}")
            return 0.5, {"error": str(e)}
    
    async def _fetch_company_info(self, symbol: str) -> Dict:
        """
        Fetch company information from yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info:
                return info
        except Exception as e:
            logging.error(f"Error fetching info for {symbol}: {e}")
        
        return {}