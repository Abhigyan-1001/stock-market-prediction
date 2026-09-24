from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Builds and compiles an LSTM model for stock price prediction.

    Parameters:
        input_shape: (timesteps, features), e.g. (60, 1)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# --- Manual test: build and inspect the model ---
if __name__ == "__main__":
    model = build_lstm_model(input_shape=(60, 1))
    model.summary()
    