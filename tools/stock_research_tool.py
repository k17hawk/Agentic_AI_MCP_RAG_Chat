import yfinance as yf
from crewai.tools import tool

#registering the function as a CrewAI tool 
@tool("Live Stock information tool")
def get_stock_price(stock_symbol: str) -> str:
    """
    This function takes a stock symbol as input and returns the current stock price.
    It uses the yFinance library to fetch the stock data.
    Args:
        stock_symbol (str): The stock symbol for which to fetch the price (e.g., "AAPL" for Apple Inc.)
    Returns:
        str: A string containing the current stock price or an error message if the stock symbol is invalid
    """
    try:
        # Fetching the stock data using yFinance
        stock = yf.Ticker(stock_symbol)
        stock_info = stock.info
        
        # Extracting the current stock price
        current_price = stock_info.get("currentPrice", "N/A")
        change = stock_info.get("regularMarketChange", "N/A")
        percent_change = stock_info.get("regularMarketChangePercent", "N/A")
        currency = stock_info.get("currency", "N/A")
        
        if current_price != "N/A":
            return f"The current price of {stock_symbol} is: {current_price} {currency} (Change: {change} {currency}, Percent Change: {percent_change}%)"
        
        return (f"Stock:{stock_symbol.upper()} \n"
                f"Current Price: {current_price} {currency} \n"
                f"Change: {change} {currency} \n"
                f"Percent Change: {percent_change}%"
        )
        
    except Exception as e:
        return f"Error fetching data for stock symbol: {stock_symbol}. Please check the symbol and try again. Error details: {str(e)}"

# result = get_stock_price.run("AAPL")
# print(result)