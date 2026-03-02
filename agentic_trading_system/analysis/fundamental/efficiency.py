"""
Efficiency Analyzer - Analyzes operational efficiency metrics
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as  logging

class EfficiencyAnalyzer:
    """
    Analyzes operational efficiency metrics:
    - Asset Turnover
    - Inventory Turnover
    - Receivables Turnover
    - Days Sales Outstanding (DSO)
    - Days Inventory Outstanding (DIO)
    - Cash Conversion Cycle
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Efficiency thresholds
        self.asset_turnover_ideal = config.get("asset_turnover_ideal", 1.0)
        self.inventory_turnover_ideal = config.get("inventory_turnover_ideal", 5.0)
        
        self.dso_ideal = config.get("dso_ideal", 30)  # Days
        self.dio_ideal = config.get("dio_ideal", 60)  # Days
        
        logging.info(f"✅ EfficiencyAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze efficiency metrics
        """
        results = {}
        scores = []
        
        # Asset Turnover
        asset_turnover = info.get("assetTurnover")
        if asset_turnover:
            analysis = self._analyze_asset_turnover(asset_turnover)
            results["asset_turnover"] = analysis
            scores.append(analysis["score"])
        
        # Inventory Turnover
        inventory_turnover = info.get("inventoryTurnover")
        if inventory_turnover:
            analysis = self._analyze_inventory_turnover(inventory_turnover)
            results["inventory_turnover"] = analysis
            scores.append(analysis["score"])
        
        # Receivables Turnover
        receivables_turnover = info.get("receivablesTurnover")
        if receivables_turnover:
            analysis = self._analyze_receivables_turnover(receivables_turnover)
            results["receivables_turnover"] = analysis
            scores.append(analysis["score"])
        
        # Days Sales Outstanding
        dso = self._calculate_dso(info)
        if dso:
            analysis = self._analyze_dso(dso)
            results["dso"] = analysis
            scores.append(analysis["score"])
        
        # Days Inventory Outstanding
        dio = self._calculate_dio(info)
        if dio:
            analysis = self._analyze_dio(dio)
            results["dio"] = analysis
            scores.append(analysis["score"])
        
        # Cash Conversion Cycle
        ccc = self._calculate_ccc(info, dso, dio)
        if ccc:
            analysis = self._analyze_ccc(ccc)
            results["ccc"] = analysis
            scores.append(analysis["score"])
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "very_efficient"
        elif composite_score >= 0.6:
            signal = "efficient"
        elif composite_score >= 0.4:
            signal = "average"
        else:
            signal = "inefficient"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_asset_turnover(self, turnover: float) -> Dict:
        """Analyze Asset Turnover ratio"""
        if turnover > self.asset_turnover_ideal * 1.5:
            score = 0.9
            signal = "exceptional"
            details = f"Asset turnover exceptional at {turnover:.2f}x"
        elif turnover > self.asset_turnover_ideal:
            score = 0.8
            signal = "strong"
            details = f"Asset turnover strong at {turnover:.2f}x"
        elif turnover > self.asset_turnover_ideal * 0.7:
            score = 0.7
            signal = "good"
            details = f"Asset turnover good at {turnover:.2f}x"
        elif turnover > self.asset_turnover_ideal * 0.5:
            score = 0.6
            signal = "fair"
            details = f"Asset turnover fair at {turnover:.2f}x"
        else:
            score = 0.4
            signal = "low"
            details = f"Asset turnover low at {turnover:.2f}x"
        
        return {
            "score": float(score),
            "value": float(turnover),
            "signal": signal,
            "details": details
        }
    
    def _analyze_inventory_turnover(self, turnover: float) -> Dict:
        """Analyze Inventory Turnover ratio"""
        if turnover > self.inventory_turnover_ideal * 1.5:
            score = 0.9
            signal = "exceptional"
            details = f"Inventory turnover exceptional at {turnover:.2f}x"
        elif turnover > self.inventory_turnover_ideal:
            score = 0.8
            signal = "strong"
            details = f"Inventory turnover strong at {turnover:.2f}x"
        elif turnover > self.inventory_turnover_ideal * 0.7:
            score = 0.7
            signal = "good"
            details = f"Inventory turnover good at {turnover:.2f}x"
        elif turnover > self.inventory_turnover_ideal * 0.4:
            score = 0.5
            signal = "fair"
            details = f"Inventory turnover fair at {turnover:.2f}x"
        else:
            score = 0.3
            signal = "low"
            details = f"Inventory turnover low at {turnover:.2f}x"
        
        return {
            "score": float(score),
            "value": float(turnover),
            "signal": signal,
            "details": details
        }
    
    def _analyze_receivables_turnover(self, turnover: float) -> Dict:
        """Analyze Receivables Turnover ratio"""
        if turnover > 15:
            score = 0.9
            signal = "exceptional"
            details = f"Receivables turnover exceptional at {turnover:.2f}x"
        elif turnover > 10:
            score = 0.8
            signal = "strong"
            details = f"Receivables turnover strong at {turnover:.2f}x"
        elif turnover > 8:
            score = 0.7
            signal = "good"
            details = f"Receivables turnover good at {turnover:.2f}x"
        elif turnover > 5:
            score = 0.6
            signal = "fair"
            details = f"Receivables turnover fair at {turnover:.2f}x"
        else:
            score = 0.4
            signal = "slow"
            details = f"Receivables turnover slow at {turnover:.2f}x"
        
        return {
            "score": float(score),
            "value": float(turnover),
            "signal": signal,
            "details": details
        }
    
    def _analyze_dso(self, dso: float) -> Dict:
        """Analyze Days Sales Outstanding"""
        if dso < self.dso_ideal:
            score = 0.9
            signal = "excellent"
            details = f"DSO excellent at {dso:.1f} days"
        elif dso < self.dso_ideal * 1.3:
            score = 0.8
            signal = "good"
            details = f"DSO good at {dso:.1f} days"
        elif dso < self.dso_ideal * 1.7:
            score = 0.6
            signal = "fair"
            details = f"DSO fair at {dso:.1f} days"
        elif dso < self.dso_ideal * 2:
            score = 0.4
            signal = "slow"
            details = f"DSO slow at {dso:.1f} days"
        else:
            score = 0.2
            signal = "very_slow"
            details = f"DSO very slow at {dso:.1f} days"
        
        return {
            "score": float(score),
            "value": float(dso),
            "signal": signal,
            "details": details
        }
    
    def _analyze_dio(self, dio: float) -> Dict:
        """Analyze Days Inventory Outstanding"""
        if dio < self.dio_ideal:
            score = 0.9
            signal = "excellent"
            details = f"DIO excellent at {dio:.1f} days"
        elif dio < self.dio_ideal * 1.3:
            score = 0.8
            signal = "good"
            details = f"DIO good at {dio:.1f} days"
        elif dio < self.dio_ideal * 1.7:
            score = 0.6
            signal = "fair"
            details = f"DIO fair at {dio:.1f} days"
        elif dio < self.dio_ideal * 2:
            score = 0.4
            signal = "slow"
            details = f"DIO slow at {dio:.1f} days"
        else:
            score = 0.2
            signal = "very_slow"
            details = f"DIO very slow at {dio:.1f} days"
        
        return {
            "score": float(score),
            "value": float(dio),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ccc(self, ccc: float) -> Dict:
        """Analyze Cash Conversion Cycle"""
        if ccc < 0:
            score = 0.9
            signal = "exceptional"
            details = f"Cash conversion cycle exceptional at {ccc:.1f} days (negative)"
        elif ccc < 30:
            score = 0.8
            signal = "excellent"
            details = f"Cash conversion cycle excellent at {ccc:.1f} days"
        elif ccc < 60:
            score = 0.7
            signal = "good"
            details = f"Cash conversion cycle good at {ccc:.1f} days"
        elif ccc < 90:
            score = 0.5
            signal = "fair"
            details = f"Cash conversion cycle fair at {ccc:.1f} days"
        elif ccc < 120:
            score = 0.3
            signal = "long"
            details = f"Cash conversion cycle long at {ccc:.1f} days"
        else:
            score = 0.1
            signal = "very_long"
            details = f"Cash conversion cycle very long at {ccc:.1f} days"
        
        return {
            "score": float(score),
            "value": float(ccc),
            "signal": signal,
            "details": details
        }
    
    def _calculate_dso(self, info: Dict) -> Optional[float]:
        """
        Calculate Days Sales Outstanding = (Accounts Receivable / Revenue) * 365
        """
        receivables = info.get("netReceivables")
        revenue = info.get("totalRevenue")
        
        if receivables and revenue and revenue > 0:
            return (receivables / revenue) * 365
        
        return None
    
    def _calculate_dio(self, info: Dict) -> Optional[float]:
        """
        Calculate Days Inventory Outstanding = (Inventory / COGS) * 365
        """
        inventory = info.get("inventory")
        cogs = info.get("costOfRevenue")
        
        if inventory and cogs and cogs > 0:
            return (inventory / cogs) * 365
        
        return None
    
    def _calculate_ccc(self, info: Dict, dso: Optional[float], dio: Optional[float]) -> Optional[float]:
        """
        Calculate Cash Conversion Cycle = DSO + DIO - DPO
        DPO = Days Payables Outstanding
        """
        # Calculate DPO
        payables = info.get("accountPayables")
        cogs = info.get("costOfRevenue")
        
        dpo = None
        if payables and cogs and cogs > 0:
            dpo = (payables / cogs) * 365
        
        # Calculate CCC
        if dso is not None and dio is not None and dpo is not None:
            return dso + dio - dpo
        
        return None