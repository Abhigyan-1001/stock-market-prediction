import streamlit as st
import numpy as np
import pandas as pd

from src.forecast import forecast_future
from src.data_collection import fetch_stock_data
from src.visualization import plot_closing_price, plot_candlestick, plot_volume, plot_moving_averages
from src.preprocessing import chronological_split, scale_data, create_sequences
from src.model import build_lstm_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Cached loading of model and scaler (runs once, not on every interaction) 
LOOKBACK = 60
TRAIN_EPOCHS = 10  # Reduced from 20 for faster in-app training


@st.cache_resource
def train_model_for_ticker(ticker: str, start: str, end: str):
    """
    Trains a fresh LSTM for the given ticker/date range.
    Cached by Streamlit: same (ticker, start, end) won't retrain.
    """
    df = fetch_stock_data(ticker, start, end)

    train_raw, test_raw = chronological_split(df, train_ratio=0.8)

    if len(train_raw) <= LOOKBACK or len(test_raw) <= LOOKBACK:
        raise ValueError(
            f"Not enough historical data to train a model with a {LOOKBACK}-day lookback. "
            f"Please choose a longer date range."
        )

    scaled_train, scaled_test, scaler = scale_data(train_raw, test_raw)
    X_train, y_train = create_sequences(scaled_train, lookback=LOOKBACK)

    model = build_lstm_model(input_shape=(LOOKBACK, 1))
    model.fit(X_train, y_train, epochs=TRAIN_EPOCHS, batch_size=32, verbose=0)

    return model, scaler, df

#  Sidebar: user inputs 
st.sidebar.title("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=__import__("datetime").date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=__import__("datetime").date(2024, 1, 1))

forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=10)

run_button = st.sidebar.button("Run Analysis")

# --- Main page ---
st.title("Stock Market Analysis & Price Prediction")

if run_button:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df = fetch_stock_data(ticker, str(start_date), str(end_date))
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.success(f"Data loaded: {len(df)} trading days for {ticker}")

    # Latest price metric
    latest_price = df['Close'].values.flatten()[-1]
    st.metric(label=f"Latest Closing Price ({ticker})", value=f"${latest_price:.2f}")

    # Charts
    st.subheader("Historical Price")
    st.plotly_chart(plot_closing_price(df, ticker), use_container_width=True)

    st.subheader("Candlestick Chart")
    st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)

    st.subheader("Trading Volume")
    st.plotly_chart(plot_volume(df, ticker), use_container_width=True)

    st.subheader("Moving Averages")
    st.plotly_chart(plot_moving_averages(df, ticker), use_container_width=True)

    st.subheader("Historical Data")
    st.dataframe(df.tail(50))

    # --- Model Training & Evaluation Section ---
    st.markdown("---")
    st.subheader("Model Prediction & Evaluation")
    st.caption(
        f"Training a lightweight LSTM ({TRAIN_EPOCHS} epochs) specifically for {ticker} on this data range. "
        f"This is a faster, smaller version of the model compared to the fully-trained offline version — "
        f"good for demonstration, not for production-grade accuracy."
    )

    try:
        with st.spinner(f"Training model for {ticker}... this may take a minute."):
            model, scaler, _ = train_model_for_ticker(ticker, str(start_date), str(end_date))

        train_raw, test_raw = chronological_split(df, train_ratio=0.8)
        scaled_test = scaler.transform(test_raw.reshape(-1, 1))
        X_test, y_test = create_sequences(scaled_test, lookback=LOOKBACK)

        scaled_predictions = model.predict(X_test, verbose=0)
        predicted_prices = scaler.inverse_transform(scaled_predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${mae:.2f}")
        col2.metric("RMSE", f"${rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual_prices.flatten(), mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(y=predicted_prices.flatten(), mode='lines', name='Predicted'))
        fig.update_layout(
            title=f"{ticker} — Actual vs Predicted Price (Test Set)",
            xaxis_title="Time Step (Test Period)",
            yaxis_title="Price ($)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Future Forecast Section ---
        st.markdown("---")
        st.subheader(f"Future Price Forecast ({forecast_days} Days)")

        st.warning(
            "**Important limitation:** this forecast is generated recursively — each predicted day "
            "is fed back in as input to predict the next. Errors compound with each step, so predictions "
            "become progressively less reliable the further into the future they go. Short-horizon forecasts "
            "(1-5 days) are more trustworthy than long ones. **This is not financial advice.**"
        )

        recent_prices = df['Close'].values.flatten()
        future_prices = forecast_future(model, scaler, recent_prices, n_days=forecast_days)

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=list(range(1, forecast_days + 1)),
            y=future_prices,
            mode='lines+markers',
            name='Forecasted Price'
        ))
        forecast_fig.update_layout(
            title=f"{ticker} — Next {forecast_days} Trading Days (Forecast)",
            xaxis_title="Days Ahead",
            yaxis_title="Predicted Price ($)",
            template="plotly_white"
        )
        st.plotly_chart(forecast_fig, use_container_width=True)

        forecast_df = pd.DataFrame({
            "Day": range(1, forecast_days + 1),
            "Forecasted Price": [f"${p:.2f}" for p in future_prices]
        })
        st.dataframe(forecast_df, hide_index=True)

    except ValueError as e:
        st.error(str(e))

else:
    st.info("Set your ticker and date range in the sidebar, then click **Run Analysis**.")



