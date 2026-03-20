import yfinance as yf

def test_yahoo_options():
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    
    # Get expiration dates
    expirations = stock.options
    print(f"Expiration dates: {expirations[:3]}")
    
    # Get options chain for nearest expiration
    if expirations:
        chain = stock.option_chain(expirations[0])
        print(f"Calls: {len(chain.calls)}")
        print(f"Puts: {len(chain.puts)}")
        
        # Show unusual activity (volume > open interest)
        unusual_calls = chain.calls[chain.calls['volume'] > chain.calls['openInterest'] * 2]
        print(f"Unusual calls: {len(unusual_calls)}")

if __name__ == "__main__":
    test_yahoo_options()