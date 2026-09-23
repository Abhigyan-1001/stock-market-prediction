import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical OHLCV stock data for a given ticker between two dates.

    Parameters:
        ticker (str): Stock symbol, e.g. 'AAPL' or 'TCS.NS'
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: OHLCV data indexed by date
    """
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check if the ticker symbol is correct.")

    data = data.dropna()

    return data


# --- Quick manual test (only runs when you execute this file directly) ---
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g. AAPL, TCS.NS): ")
    start = input("Enter start date (YYYY-MM-DD): ")
    end = input("Enter end date (YYYY-MM-DD): ")

    df = fetch_stock_data(ticker, start, end)
    print(df.head())
    print(f"\nShape of data: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    
