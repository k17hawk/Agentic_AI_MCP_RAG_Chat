"""
Hierarchical Risk Parity - Advanced risk parity using clustering
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from agentic_trading_system.utils.logger import logger as logging


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) - Uses clustering to build diversified portfolios
    
    Advantages over traditional methods:
    - More stable weights
    - Better out-of-sample performance
    - Handles illiquid assets better
    - No inversion of covariance matrix
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Clustering parameters
        self.linkage_method = config.get("linkage_method", "ward")
        self.correlation_method = config.get("correlation_method", "pearson")
        
        # Weight constraints
        self.max_weight = config.get("max_weight", 0.30)
        self.min_weight = config.get("min_weight", 0.01)
        
        logging.info(f"✅ HierarchicalRiskParity initialized")
    
    
    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        correlation = returns.corr()
        covariance = returns.cov() * 252
        
        n_assets = len(returns.columns)
        assets = list(returns.columns)
        
        distance_matrix = self._correlation_to_distance(correlation)
        
        # Pass cluster_registry back from _build_clusters
        root_cluster, cluster_registry = self._build_clusters(distance_matrix, assets)
        
        # Pass registry into recursive bisection
        weights_dict = self._recursive_bisection(root_cluster, cluster_registry, covariance, assets)
        
        weights_array = np.array([weights_dict.get(asset, 0) for asset in assets])
        weights_array = self._apply_constraints(weights_array)
        weights = {asset: float(weights_array[i]) for i, asset in enumerate(assets)}
        
        expected_returns = returns.mean() * 252
        weights_array = np.array([weights[a] for a in assets])
        portfolio_return = np.sum(expected_returns * weights_array)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(covariance.values, weights_array)))
        
        risk_contrib = self._calculate_risk_contributions(weights_array, covariance.values)
        total_risk = np.sum(risk_contrib)
        risk_contrib_pct = risk_contrib / total_risk if total_risk > 0 else risk_contrib
        
        return {
            "weights": weights,
            "risk_contributions": dict(zip(assets, risk_contrib)),
            "risk_contributions_pct": dict(zip(assets, risk_contrib_pct)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0),
            "method": "hierarchical_risk_parity",
            "assets": assets
        }


    
    def _correlation_to_distance(self, correlation: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix
        d = √(0.5 * (1 - ρ))
        """
        distance = np.sqrt(0.5 * (1 - correlation.values))
        return distance
    
    def _build_clusters(self, distance_matrix: np.ndarray, assets: List[str]) -> Tuple[Dict, Dict]:
        """Returns (root_cluster, cluster_registry)"""
        n_assets = len(assets)
        condensed = squareform(distance_matrix)
        linkage_matrix = linkage(condensed, method=self.linkage_method)
        root_cluster, registry = self._build_cluster_tree(linkage_matrix, n_assets, assets)
        return root_cluster, registry
    
    def _build_cluster_tree(self, linkage_matrix: np.ndarray, n_assets: int, assets: List[str]) -> Tuple[Dict, Dict]:
        """
        Build recursive cluster tree. Returns (root_cluster, full registry).
        """
        registry = {}
        
        # Leaf nodes
        for i in range(n_assets):
            registry[i] = {
                'is_leaf': True,
                'assets': [assets[i]],
                'size': 1,
                'children': []
            }
        
        current_id = n_assets
        for link in linkage_matrix:
            left_id = int(link[0])
            right_id = int(link[1])
            distance = link[2]
            
            left_cluster = registry[left_id]
            right_cluster = registry[right_id]
            
            registry[current_id] = {
                'is_leaf': False,
                'assets': left_cluster['assets'] + right_cluster['assets'],
                'children': [left_id, right_id],   # store IDs, not objects
                'distance': float(distance),
                'size': left_cluster['size'] + right_cluster['size']
            }
            current_id += 1
        
        root_id = current_id - 1
        return registry[root_id], registry
    
    def _create_cluster_from_list(self, clusters: Dict, node_id: int) -> Dict:
        """Create a cluster from a node ID that might not exist yet"""
        if node_id in clusters:
            return clusters[node_id]
        
        # If node is less than n_assets, it's a leaf
        n_assets = len([c for c in clusters.keys() if isinstance(clusters[c], dict) and clusters[c].get('is_leaf')])
        
        if node_id < n_assets:
            # This shouldn't happen if we initialized all leaves
            return {'is_leaf': True, 'assets': [f"asset_{node_id}"], 'size': 1, 'children': []}
        
        return {'is_leaf': False, 'assets': [], 'children': [], 'size': 0}
    
    def _recursive_bisection(self, cluster: Dict, registry: Dict, covariance: pd.DataFrame, assets: List[str]) -> Dict[str, float]:
        """
        Recursive bisection using the cluster registry to look up children.
        """
        if cluster['is_leaf']:
            return {cluster['assets'][0]: 1.0}
        
        left_id, right_id = cluster['children']
        left_cluster = registry[left_id]
        right_cluster = registry[right_id]
        
        left_variance = self._cluster_variance(left_cluster, covariance)
        right_variance = self._cluster_variance(right_cluster, covariance)
        
        total_variance = left_variance + right_variance
        if total_variance > 0:
            left_weight = right_variance / total_variance
            right_weight = left_variance / total_variance
        else:
            left_weight = right_weight = 0.5
        
        left_weights = self._recursive_bisection(left_cluster, registry, covariance, assets)
        right_weights = self._recursive_bisection(right_cluster, registry, covariance, assets)
        
        weights = {}
        for asset, w in left_weights.items():
            weights[asset] = w * left_weight
        for asset, w in right_weights.items():
            weights[asset] = w * right_weight
        
        return weights

    
    def _get_cluster_from_assets(self, cluster_id, covariance: pd.DataFrame, assets: List[str]) -> Dict:
        """
        Create a cluster from an ID
        """
        # For simplicity, treat any non-leaf as containing all assets not in the other branch
        # This is a simplified approach - in production, we'd track clusters properly
        return {
            'is_leaf': False,
            'assets': assets,  # Placeholder - would be actual assets in the cluster
            'children': [],
            'size': len(assets)
        }
    
    def _cluster_variance(self, cluster: Dict, covariance: pd.DataFrame) -> float:
        """
        Calculate cluster variance using the actual assets in the cluster.
        For a leaf, returns that asset's variance.
        For a non-leaf, returns the equal-weighted portfolio variance of all assets in the cluster.
        """
        cluster_assets = [a for a in cluster['assets'] if a in covariance.columns]
        
        if not cluster_assets:
            return 1e-6
        
        if len(cluster_assets) == 1:
            return float(covariance.loc[cluster_assets[0], cluster_assets[0]])
        
        n = len(cluster_assets)
        w = np.ones(n) / n
        cov_sub = covariance.loc[cluster_assets, cluster_assets].values
        return float(w @ cov_sub @ w)
    
    
    def _calculate_risk_contributions(self, weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset
        """
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        if portfolio_volatility == 0:
            return np.zeros_like(weights)
        
        marginal_contrib = np.dot(covariance, weights) / portfolio_volatility
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply weight constraints
        """
        # Clip weights to [min, max]
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Renormalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        
        return weights