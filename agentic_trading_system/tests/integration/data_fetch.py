#!/usr/bin/env python3
"""
TEST 1: Data Fetching - The Foundation
This must pass before anything else!
"""
import pytest
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class TestDataFetching:
    """Test the core data fetching functionality"""
    
    def test_yfinance_connection(self):
        """Test if we can connect to yFinance"""
        ticker = yf.Ticker("AAPL")
        assert ticker is not None
        print("✅ yFinance connection working")
    
    def test_fetch_historical_data(self):
        """Test fetching historical data"""
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="6mo")  
        
        assert not data.empty
        assert len(data) >= 60  # At least 60 days
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        
        print(f"✅ Fetched {len(data)} days of data")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    
    def test_fetch_company_info(self):
        """Test fetching company information"""
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        assert info is not None
        assert 'longName' in info
        assert 'sector' in info
        assert 'marketCap' in info
        
        print(f"✅ Company: {info.get('longName')}")
        print(f"   Sector: {info.get('sector')}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,}")
    
    def test_multiple_symbols(self):
        """Test fetching data for multiple symbols"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo")
            assert not data.empty
            print(f"✅ {symbol}: {len(data)} days")
    
    def test_data_quality(self):
        """Test data quality checks"""
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="6mo")
        
        # Check for gaps
        date_diff = data.index.to_series().diff().dt.days
        gaps = date_diff[date_diff > 3].count()
        assert gaps == 0, f"Found {gaps} data gaps"
        
        # Check for constant values (stale data)
        assert data['Close'].std() > 0.01, "Data might be stale"
        
        print(f"✅ Data quality: No gaps, reasonable variance")