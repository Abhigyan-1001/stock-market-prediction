import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from src.data_collection import fetch_stock_data
from src.preprocessing import chronological_split, create_sequences

LOOKBACK = 60
TICKER = "AAPL"


def evaluate_model():
    # 1. Load the saved model and scaler
    model = load_model("models/lstm_model.keras")
    scaler = joblib.load("models/scaler.pkl")

    # 2. Rebuild the exact same train/test split used during training
    df = fetch_stock_data(TICKER, "2015-01-01", "2024-01-01")
    train_raw, test_raw = chronological_split(df, train_ratio=0.8)

    # 3. Scale the test data using the ALREADY-FITTED scaler (no re-fitting!)
    scaled_test = scaler.transform(test_raw.reshape(-1, 1))

    # 4. Build test sequences
    X_test, y_test = create_sequences(scaled_test, lookback=LOOKBACK)

    # 5. Predict on the test set
    scaled_predictions = model.predict(X_test)

    # 6. Inverse-transform predictions AND true values back to real price scale
    predicted_prices = scaler.inverse_transform(scaled_predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 7. Compute metrics on REAL price scale
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    print(f"\n--- Evaluation on Test Set ({TICKER}) ---")
    print(f"MAE  : ${mae:.2f}")
    print(f"RMSE : ${rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    # 8. Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Price")
    plt.plot(predicted_prices, label="Predicted Price")
    plt.title(f"{TICKER} — Actual vs Predicted Price (Test Set)")
    plt.xlabel("Time Step (Test Period)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

    return actual_prices, predicted_prices, mae, rmse, mape


if __name__ == "__main__":
    evaluate_model()
    