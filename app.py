import streamlit as st
import joblib
from tensorflow.keras.models import load_model

from src.data_collection import fetch_stock_data
from src.visualization import plot_closing_price, plot_candlestick, plot_volume, plot_moving_averages

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# --- Cached loading of model and scaler (runs once, not on every interaction) ---
@st.cache_resource
def load_trained_model():
    model = load_model("models/lstm_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


# --- Sidebar: user inputs ---
st.sidebar.title("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=__import__("datetime").date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=__import__("datetime").date(2024, 1, 1))

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

else:
    st.info("Set your ticker and date range in the sidebar, then click **Run Analysis**.")
