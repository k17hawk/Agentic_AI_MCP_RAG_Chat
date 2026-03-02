"""
Hierarchical Risk Parity - Advanced risk parity using clustering
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from utils.logger import logger as logging

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
        """
        Perform hierarchical risk parity optimization
        """
        # Calculate correlation and covariance
        correlation = returns.corr()
        covariance = returns.cov() * 252
        
        # Calculate distance matrix
        distance_matrix = self._correlation_to_distance(correlation)
        
        # Perform hierarchical clustering
        clusters = self._build_clusters(distance_matrix)
        
        # Calculate weights using recursive bisection
        weights = self._recursive_bisection(clusters, covariance)
        
        # Apply weight constraints
        weights = self._apply_constraints(weights)
        
        # Calculate portfolio metrics
        expected_returns = returns.mean() * 252
        weights_array = np.array([weights[a] for a in returns.columns])
        portfolio_return = np.sum(expected_returns * weights_array)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(covariance.values, weights_array)))
        
        # Calculate risk contributions
        risk_contrib = self._calculate_risk_contributions(weights_array, covariance.values)
        total_risk = np.sum(risk_contrib)
        risk_contrib_pct = risk_contrib / total_risk if total_risk > 0 else risk_contrib
        
        return {
            "weights": weights,
            "risk_contributions": dict(zip(returns.columns, risk_contrib)),
            "risk_contributions_pct": dict(zip(returns.columns, risk_contrib_pct)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0),
            "clusters": clusters,
            "method": "hierarchical_risk_parity",
            "assets": list(returns.columns)
        }
    
    def _correlation_to_distance(self, correlation: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix
        d = √(0.5 * (1 - ρ))
        """
        distance = np.sqrt(0.5 * (1 - correlation.values))
        return distance
    
    def _build_clusters(self, distance_matrix: np.ndarray) -> Dict:
        """
        Build hierarchical clusters from distance matrix
        """
        # Condense distance matrix
        condensed = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed, method=self.linkage_method)
        
        # Build cluster tree
        clusters = self._build_cluster_tree(linkage_matrix)
        
        return clusters
    
    def _build_cluster_tree(self, linkage_matrix: np.ndarray) -> Dict:
        """
        Build recursive cluster tree from linkage matrix
        """
        n_assets = linkage_matrix.shape[0] + 1
        clusters = {}
        
        # Initialize leaf nodes
        for i in range(n_assets):
            clusters[i] = {
                'name': f'asset_{i}',
                'is_leaf': True,
                'children': []
            }
        
        # Build tree from bottom up
        cluster_id = n_assets
        for i, link in enumerate(linkage_matrix):
            left = int(link[0])
            right = int(link[1])
            distance = link[2]
            
            clusters[cluster_id] = {
                'name': f'cluster_{cluster_id}',
                'is_leaf': False,
                'children': [left, right],
                'distance': distance
            }
            cluster_id += 1
        
        # Root is the last cluster
        root_id = cluster_id - 1
        
        return clusters[root_id]
    
    def _recursive_bisection(self, cluster: Dict, covariance: pd.DataFrame) -> Dict[str, float]:
        """
        Recursive bisection algorithm to allocate weights
        """
        if cluster['is_leaf']:
            # Leaf node - single asset
            asset_idx = int(cluster['name'].split('_')[1])
            return {f"asset_{asset_idx}": 1.0}
        
        # Get child clusters
        left_cluster = self._get_cluster_by_id(cluster['children'][0])
        right_cluster = self._get_cluster_by_id(cluster['children'][1])
        
        # Calculate variance of each subcluster
        left_variance = self._cluster_variance(left_cluster, covariance)
        right_variance = self._cluster_variance(right_cluster, covariance)
        
        # Allocate weights inversely to variance
        total_variance = left_variance + right_variance
        if total_variance > 0:
            left_weight = right_variance / total_variance  # Inverse
            right_weight = left_variance / total_variance
        else:
            left_weight = right_weight = 0.5
        
        # Recursively allocate within clusters
        left_weights = self._recursive_bisection(left_cluster, covariance)
        right_weights = self._recursive_bisection(right_cluster, covariance)
        
        # Combine weights
        weights = {}
        for asset, w in left_weights.items():
            weights[asset] = w * left_weight
        for asset, w in right_weights.items():
            weights[asset] = w * right_weight
        
        return weights
    
    def _cluster_variance(self, cluster: Dict, covariance: pd.DataFrame) -> float:
        """
        Calculate variance of a cluster
        """
        if cluster['is_leaf']:
            asset_idx = int(cluster['name'].split('_')[1])
            return covariance.iloc[asset_idx, asset_idx]
        
        # Recursively calculate cluster variance
        left_cluster = self._get_cluster_by_id(cluster['children'][0])
        right_cluster = self._get_cluster_by_id(cluster['children'][1])
        
        left_variance = self._cluster_variance(left_cluster, covariance)
        right_variance = self._cluster_variance(right_cluster, covariance)
        
        # Simple approximation: average of child variances
        return (left_variance + right_variance) / 2
    
    def _get_cluster_by_id(self, cluster_id: int) -> Dict:
        """
        Get cluster by ID (placeholder - would need proper cluster storage)
        """
        # This is a simplified version - in production, maintain a cluster registry
        return {
            'is_leaf': cluster_id < 10,  # Assume first 10 are assets
            'name': f'asset_{cluster_id}' if cluster_id < 10 else f'cluster_{cluster_id}',
            'children': []
        }
    
    def _calculate_risk_contributions(self, weights: np.ndarray, 
                                     covariance: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset
        """
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        if portfolio_volatility == 0:
            return np.zeros_like(weights)
        
        marginal_contrib = np.dot(covariance, weights) / portfolio_volatility
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply weight constraints
        """
        # Clip weights to [min, max]
        for asset in weights:
            weights[asset] = max(self.min_weight, min(self.max_weight, weights[asset]))
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights