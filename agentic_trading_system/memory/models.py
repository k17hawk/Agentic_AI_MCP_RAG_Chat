"""
Models - Pydantic models for memory storage
"""
from typing import Dict, List, Optional, Any
from datetime import datetime,timedelta
from pydantic import BaseModel, Field, validator
import uuid

class Trade(BaseModel):
    """
    Trade model - Represents a completed trade
    """
    trade_id: str = Field(default_factory=lambda: f"trade_{uuid.uuid4().hex[:8]}")
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    total_value: float
    commission: float = 0.0
    slippage: float = 0.0
    
    # Analysis data
    confidence: float
    risk_score: float
    analysis_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Timing
    entry_time: datetime
    exit_time: Optional[datetime] = None
    hold_time_seconds: Optional[int] = None
    
    # Outcome
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    outcome: Optional[str] = None  # win, loss, breakeven
    
    # Metadata
    strategy: str = "default"
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    
    # System fields
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 1
    
    @validator('pnl', always=True)
    def calculate_pnl(cls, v, values):
        """Calculate P&L if not provided"""
        if v is not None:
            return v
        
        if 'exit_time' not in values or not values['exit_time']:
            return None
        
        if 'action' in values and 'quantity' in values and 'price' in values:
            if values['action'] == 'BUY':
                return values.get('exit_price', 0) - values['price']
            else:
                return values['price'] - values.get('exit_price', 0)
        
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Signal(BaseModel):
    """
    Signal model - Represents a trading signal
    """
    signal_id: str = Field(default_factory=lambda: f"signal_{uuid.uuid4().hex[:8]}")
    symbol: str
    signal_type: str  # price, volume, news, pattern, etc.
    confidence: float
    strength: float  # 0-1
    
    # Source information
    source: str  # trigger name
    source_version: str = "1.0.0"
    
    # Timing
    generated_at: datetime = Field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    # Signal data
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    processed_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistical significance
    z_score: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: Optional[int] = None
    
    # Market context
    market_regime: Optional[str] = None
    volatility: Optional[float] = None
    
    # Outcome tracking
    led_to_trade: bool = False
    trade_id: Optional[str] = None
    accuracy: Optional[float] = None  # How accurate was this signal type historically
    
    # Expiry
    expires_at: datetime = Field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PerformanceMetrics(BaseModel):
    """
    Performance Metrics - Trading performance statistics
    """
    period_id: str = Field(default_factory=lambda: f"perf_{datetime.now().strftime('%Y%m%d')}")
    start_date: datetime
    end_date: datetime
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # Volume and value
    total_volume: int = 0
    total_turnover: float = 0.0
    
    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    
    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    # Strategy breakdown
    by_strategy: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    by_symbol: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Market context
    market_regime: Optional[str] = None
    volatility_period: Optional[float] = None
    
    # System fields
    created_at: datetime = Field(default_factory=datetime.now)
    version: int = 1
    
    @validator('win_rate', always=True)
    def calculate_win_rate(cls, v, values):
        """Calculate win rate if not provided"""
        if v != 0.0:
            return v
        total = values.get('winning_trades', 0) + values.get('losing_trades', 0)
        if total > 0:
            return values.get('winning_trades', 0) / total
        return 0.0
    
    @validator('profit_factor', always=True)
    def calculate_profit_factor(cls, v, values):
        """Calculate profit factor if not provided"""
        if v != 0.0:
            return v
        if values.get('gross_loss', 0) > 0:
            return values.get('gross_profit', 0) / values['gross_loss']
        return float('inf') if values.get('gross_profit', 0) > 0 else 0.0

class ModelWeights(BaseModel):
    """
    Model Weights - Stores weights for ML models
    """
    weights_id: str = Field(default_factory=lambda: f"weights_{uuid.uuid4().hex[:8]}")
    model_name: str
    model_version: str
    
    # Weights by category
    technical_weights: Dict[str, float] = Field(default_factory=dict)
    fundamental_weights: Dict[str, float] = Field(default_factory=dict)
    sentiment_weights: Dict[str, float] = Field(default_factory=dict)
    timeframe_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Regime-based adjustments
    regime_adjustments: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Performance tracking
    training_date: datetime = Field(default_factory=datetime.now)
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    training_samples: int = 0
    features_used: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # System fields
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    notes: str = ""
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }