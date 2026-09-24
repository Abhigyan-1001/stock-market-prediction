import plotly.graph_objects as go
import pandas as pd


def plot_closing_price(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Line chart of closing price over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'].values.flatten(),
        mode='lines',
        name='Close Price'
    ))
    fig.update_layout(
        title=f"{ticker} — Historical Closing Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig


def plot_candlestick(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick chart showing Open/High/Low/Close."""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'].values.flatten(),
        high=data['High'].values.flatten(),
        low=data['Low'].values.flatten(),
        close=data['Close'].values.flatten(),
        name=ticker
    )])
    fig.update_layout(
        title=f"{ticker} — Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig


def plot_volume(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Bar chart of trading volume."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'].values.flatten(),
        name='Volume'
    ))
    fig.update_layout(
        title=f"{ticker} — Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white"
    )
    return fig


def plot_moving_averages(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Closing price with 20-day and 50-day moving averages."""
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'].values.flatten(), mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'].values.flatten(), mode='lines', name='MA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'].values.flatten(), mode='lines', name='MA 50'))

    fig.update_layout(
        title=f"{ticker} — Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig


# --- Manual test ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_collection import fetch_stock_data

    df = fetch_stock_data("AAPL", "2022-01-01", "2024-01-01")

    fig1 = plot_closing_price(df, "AAPL")
    fig2 = plot_candlestick(df, "AAPL")
    fig3 = plot_volume(df, "AAPL")
    fig4 = plot_moving_averages(df, "AAPL")

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()