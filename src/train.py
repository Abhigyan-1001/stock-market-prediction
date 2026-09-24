import os
import matplotlib.pyplot as plt

from src.data_collection import fetch_stock_data
from src.preprocessing import chronological_split, scale_data, create_sequences
from src.model import build_lstm_model

LOOKBACK = 60
TICKER = "AAPL"


def train_and_save_model():
    # 1. Get data
    df = fetch_stock_data(TICKER, "2015-01-01", "2024-01-01")

    # 2. Chronological split
    train_raw, test_raw = chronological_split(df, train_ratio=0.8)

    # 3. Scale (fit on train only)
    scaled_train, scaled_test, scaler = scale_data(train_raw, test_raw)

    # 4. Build sequences
    X_train, y_train = create_sequences(scaled_train, lookback=LOOKBACK)

    # 5. Build model
    model = build_lstm_model(input_shape=(LOOKBACK, 1))

    # 6. Train
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 7. Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    print("\nModel saved to models/lstm_model.keras")

    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    # 8. Plot training vs validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{TICKER} — Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()

    return model, history, scaler


if __name__ == "__main__":
    train_and_save_model()
    