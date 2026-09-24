import numpy as np
from sklearn.preprocessing import MinMaxScaler


def chronological_split(data, train_ratio: float = 0.8):
    """
    Splits a DataFrame into train/test sets IN ORDER (no shuffling).

    Parameters:
        data: DataFrame with a 'Close' column
        train_ratio: fraction of data to use for training (default 80%)

    Returns:
        train_data, test_data (both as 1D numpy arrays of closing prices)
    """
    close_prices = data['Close'].values.flatten()
    split_index = int(len(close_prices) * train_ratio)

    train_data = close_prices[:split_index]
    test_data = close_prices[split_index:]

    return train_data, test_data


def scale_data(train_data, test_data):
    """
    Fits MinMaxScaler ONLY on training data, then transforms both sets.
    This avoids data leakage from the test set into the scaler.

    Returns:
        scaled_train, scaled_test, fitted_scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_reshaped = train_data.reshape(-1, 1)
    test_reshaped = test_data.reshape(-1, 1)

    scaled_train = scaler.fit_transform(train_reshaped)
    scaled_test = scaler.transform(test_reshaped)

    return scaled_train, scaled_test, scaler


def create_sequences(data, lookback: int = 60):
    """
    Converts a 1D scaled price array into (X, y) sequences for LSTM training.

    Parameters:
        data: scaled 1D (or column-shaped) array of prices
        lookback: number of past days used to predict the next day

    Returns:
        X: shape (num_samples, lookback, 1)
        y: shape (num_samples,)
    """
    data = data.flatten()
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to (samples, timesteps, features) - required LSTM input shape
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


# --- Manual test using real stock data ---
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_collection import fetch_stock_data

    df = fetch_stock_data("AAPL", "2018-01-01", "2024-01-01")

    train_raw, test_raw = chronological_split(df, train_ratio=0.8)
    print(f"Train size: {len(train_raw)}, Test size: {len(test_raw)}")

    scaled_train, scaled_test, scaler = scale_data(train_raw, test_raw)
    print(f"Scaled train range: {scaled_train.min():.3f} to {scaled_train.max():.3f}")
    print(f"Scaled test range: {scaled_test.min():.3f} to {scaled_test.max():.3f}")

    X_train, y_train = create_sequences(scaled_train, lookback=60)
    X_test, y_test = create_sequences(scaled_test, lookback=60)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    