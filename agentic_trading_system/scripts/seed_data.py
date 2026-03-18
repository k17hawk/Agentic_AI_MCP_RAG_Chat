#!/usr/bin/env python3
"""
Seed the database with initial test data
"""
import sys
from pathlib import Path
import random
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import numpy as np

try:
    from memory.repositories.trade_repository import TradeRepository
    from memory.repositories.signal_repository import SignalRepository
    from memory.models import Trade, Signal
    from config.settings import settings
except ImportError as e:
    print(f"⚠️  Some imports failed: {e}")
    print("Running in mock mode - will create sample data files")

class DataSeeder:
    """Seed the system with initial test data"""
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"]
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def seed_all(self):
        """Run all seeding operations"""
        print("=" * 60)
        print("🌱 Seeding Trading System with Test Data")
        print("=" * 60)
        
        self.seed_raw_market_data()
        self.seed_processed_features()
        self.seed_trade_history()
        self.seed_signal_history()
        self.seed_model_weights()
        self.create_sample_reports()
        
        print("\n" + "=" * 60)
        print("✅ Seeding complete!")
        print("=" * 60)
    
    def seed_raw_market_data(self):
        """Download raw market data for symbols"""
        print("\n📊 Downloading raw market data...")
        
        for symbol in self.symbols[:3]:  # First 3 symbols
            try:
                print(f"  Fetching {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Download different timeframes
                data_1y = ticker.history(period="1y")
                data_6mo = ticker.history(period="6mo")
                
                # Save to raw directory
                filepath_1y = self.raw_dir / "market_data" / f"{symbol}_1y.csv"
                filepath_6mo = self.raw_dir / "market_data" / f"{symbol}_6mo.csv"
                
                data_1y.to_csv(filepath_1y)
                data_6mo.to_csv(filepath_6mo)
                
                # Save info
                info = ticker.info
                info_path = self.raw_dir / "fundamentals" / f"{symbol}_info.json"
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2, default=str)
                
                print(f"    ✅ Saved {len(data_1y)} days of data")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    def seed_processed_features(self):
        """Create processed features from raw data"""
        print("\n🔧 Creating processed features...")
        
        for symbol in self.symbols[:3]:
            try:
                # Load raw data
                filepath = self.raw_dir / "market_data" / f"{symbol}_6mo.csv"
                if not filepath.exists():
                    continue
                
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Calculate technical features
                features = pd.DataFrame(index=data.index)
                
                # Returns
                features['returns'] = data['Close'].pct_change()
                features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                
                # Moving averages
                features['sma_20'] = data['Close'].rolling(20).mean()
                features['sma_50'] = data['Close'].rolling(50).mean()
                features['ema_12'] = data['Close'].ewm(span=12).mean()
                
                # RSI
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # Volume
                features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
                
                # Volatility
                features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
                
                # Save processed features
                output_path = self.processed_dir / "features" / f"{symbol}_features.parquet"
                features.to_parquet(output_path)
                
                print(f"  ✅ Created {len(features)} features for {symbol}")
                
            except Exception as e:
                print(f"  ❌ Error processing {symbol}: {e}")
    
    def seed_trade_history(self):
        """Create sample trade history"""
        print("\n📈 Creating sample trade history...")
        
        trades = []
        start_date = datetime.now() - timedelta(days=90)
        
        for i, symbol in enumerate(self.symbols):
            # Generate random trades
            num_trades = random.randint(5, 15)
            
            for j in range(num_trades):
                trade_date = start_date + timedelta(days=random.randint(0, 90))
                entry_price = random.uniform(100, 500)
                exit_price = entry_price * (1 + random.uniform(-0.1, 0.15))
                quantity = random.randint(10, 100)
                
                pnl = (exit_price - entry_price) * quantity
                
                trade = {
                    "trade_id": f"trade_{i}_{j}",
                    "symbol": symbol,
                    "action": "BUY" if random.random() > 0.5 else "SELL",
                    "quantity": quantity,
                    "price": entry_price,
                    "exit_price": exit_price,
                    "total_value": entry_price * quantity,
                    "pnl": pnl,
                    "entry_time": trade_date.isoformat(),
                    "exit_time": (trade_date + timedelta(days=random.randint(1, 10))).isoformat(),
                    "outcome": "win" if pnl > 0 else "loss",
                    "strategy": random.choice(["momentum", "mean_reversion", "breakout"])
                }
                trades.append(trade)
        
        # Save to JSON
        trades_path = self.data_dir / "processed" / "sample_trades.json"
        with open(trades_path, 'w') as f:
            json.dump(trades, f, indent=2)
        
        print(f"  ✅ Created {len(trades)} sample trades")
    
    def seed_signal_history(self):
        """Create sample signal history"""
        print("\n📶 Creating sample signal history...")
        
        signals = []
        start_date = datetime.now() - timedelta(days=30)
        
        signal_types = ["price_surge", "volume_spike", "golden_cross", "rsi_oversold", "news_sentiment"]
        
        for i, symbol in enumerate(self.symbols):
            num_signals = random.randint(10, 30)
            
            for j in range(num_signals):
                signal_date = start_date + timedelta(hours=random.randint(0, 720))
                
                signal = {
                    "signal_id": f"signal_{i}_{j}",
                    "symbol": symbol,
                    "signal_type": random.choice(signal_types),
                    "confidence": random.uniform(0.6, 0.95),
                    "source": random.choice(["price_alert", "news_alert", "pattern_recognition"]),
                    "generated_at": signal_date.isoformat(),
                    "led_to_trade": random.random() > 0.7
                }
                signals.append(signal)
        
        # Save to JSON
        signals_path = self.data_dir / "processed" / "sample_signals.json"
        with open(signals_path, 'w') as f:
            json.dump(signals, f, indent=2)
        
        print(f"  ✅ Created {len(signals)} sample signals")
    
    def seed_model_weights(self):
        """Create sample model weights"""
        print("\n🤖 Creating sample model weights...")
        
        models_dir = self.data_dir / "models"
        
        # Technical model weights
        tech_weights = {
            "trend": random.uniform(0.3, 0.4),
            "momentum": random.uniform(0.2, 0.3),
            "volume": random.uniform(0.15, 0.25),
            "volatility": random.uniform(0.1, 0.2),
            "pattern": random.uniform(0.05, 0.15)
        }
        
        # Normalize
        total = sum(tech_weights.values())
        tech_weights = {k: v/total for k, v in tech_weights.items()}
        
        tech_path = models_dir / "technical" / "weights_v1.json"
        with open(tech_path, 'w') as f:
            json.dump({
                "model": "technical_ensemble",
                "version": "1.0.0",
                "weights": tech_weights,
                "created_at": datetime.now().isoformat(),
                "performance": {
                    "accuracy": random.uniform(0.6, 0.8),
                    "precision": random.uniform(0.6, 0.8),
                    "recall": random.uniform(0.6, 0.8)
                }
            }, f, indent=2)
        
        print(f"  ✅ Created sample model weights")
    
    def create_sample_reports(self):
        """Create sample report files"""
        print("\n📄 Creating sample reports...")
        
        reports_dir = self.data_dir / "reports"
        
        # Daily report
        daily_report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": {
                "total_pnl": random.uniform(-1000, 2000),
                "win_rate": random.uniform(0.4, 0.7),
                "total_trades": random.randint(5, 20),
                "sharpe_ratio": random.uniform(0.5, 2.0)
            },
            "trades": []
        }
        
        daily_path = reports_dir / "daily" / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
        with open(daily_path, 'w') as f:
            json.dump(daily_report, f, indent=2)
        
        print(f"  ✅ Created sample daily report")

if __name__ == "__main__":
    seeder = DataSeeder()
    seeder.seed_all()