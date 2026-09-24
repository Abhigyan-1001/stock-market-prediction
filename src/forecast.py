import numpy as np
import joblib
from tensorflow.keras.models import load_model

from src.data_collection import fetch_stock_data

LOOKBACK = 60
TICKER = "AAPL"


def forecast_future(model, scaler, recent_data, n_days: int = 10):
    """
    Recursively forecasts n_days into the future using the trained LSTM.

    Parameters:
        model: trained Keras LSTM model
        scaler: fitted MinMaxScaler (same one used in training)
        recent_data: 1D array of the most recent real closing prices (at least `lookback` long)
        n_days: how many future days to forecast

    Returns:
        List of predicted future prices (real scale, not normalized)
    """
    scaled_recent = scaler.transform(recent_data.reshape(-1, 1)).flatten()

    # Starting with the last `lookback` days as our sliding window
    current_window = list(scaled_recent[-LOOKBACK:])

    future_predictions_scaled = []

    for _ in range(n_days):
        input_seq = np.array(current_window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)

        next_scaled_pred = model.predict(input_seq, verbose=0)[0, 0]

        future_predictions_scaled.append(next_scaled_pred)

        # Slide the window forward: drop oldest, append the new prediction
        current_window.append(next_scaled_pred)

    # Convert scaled predictions back to real prices
    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions_scaled)

    return future_predictions.flatten()


#  Manual test 
if __name__ == "__main__":
    model = load_model("models/lstm_model.keras")
    scaler = joblib.load("models/scaler.pkl")

    df = fetch_stock_data(TICKER, "2015-01-01", "2024-01-01")
    recent_prices = df['Close'].values.flatten()

    forecast = forecast_future(model, scaler, recent_prices, n_days=10)

    print(f"\n--- {TICKER}: Next 10 Trading Days Forecast ---")
    for i, price in enumerate(forecast, start=1):
        print(f"Day {i}: ${price:.2f}")