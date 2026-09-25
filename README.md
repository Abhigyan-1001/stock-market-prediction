# LSTM-Based Stock Market Analysis & Price Prediction

An interactive web application that analyzes historical stock market data, visualizes trends, and forecasts future prices using an LSTM (Long Short-Term Memory) neural network. Built end-to-end with Python, TensorFlow/Keras, and Streamlit.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is **not** financial advice and should never be used as the sole basis for real trading or investment decisions. See [Limitations & Disclaimer](#limitations--disclaimer) below.

---

## Overview

This application lets a user pick any publicly traded stock ticker, then:

- Downloads and visualizes its historical price data
- Trains a fresh LSTM model on that specific stock, on the fly
- Evaluates the model's accuracy against real, unseen historical data
- Forecasts the next N days of prices using recursive multi-step forecasting

The goal is to demonstrate a complete, honest machine learning pipeline — data collection, preprocessing, model training, evaluation, and deployment — in a single working web app.

---

## Features

- Interactive ticker, date range, and forecast horizon selection
- Historical price line chart, candlestick chart, and trading volume chart
- 20-day and 50-day moving averages
- Per-ticker LSTM model, trained on demand (not a single fixed pre-trained model)
- Evaluation metrics: MAE, RMSE, and MAPE, computed on a held-out test set
- Actual vs. Predicted price chart for direct visual evaluation
- Recursive multi-step future price forecasting with an adjustable horizon (1–30 days)
- Clear in-app disclaimers about forecasting limitations
- Graceful error handling for invalid tickers, network issues, and insufficient data

---

## Architecture

```
User
 ↓
Streamlit Interface (ticker, date range, forecast horizon)
 ↓
yfinance Data Collection  →  raw OHLCV data
 ↓
Data Cleaning & Preprocessing  →  missing values handled, chronological split
 ↓
Exploratory Data Analysis  →  price, candlestick, volume, moving average charts
 ↓
Feature Preparation  →  MinMax scaling (fit on train only), 60-day sequence windows
 ↓
LSTM Model  →  Sequential([LSTM, Dropout, LSTM, Dropout, Dense])
 ↓
Training  →  per-ticker, cached by (ticker, start date, end date)
 ↓
Prediction  →  run on held-out test sequences
 ↓
Evaluation  →  MAE, RMSE, MAPE + Actual vs. Predicted chart
 ↓
Recursive Forecasting  →  predict next day, feed back in, repeat
 ↓
Interactive Visualization  →  Plotly charts rendered in the Streamlit dashboard
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| yfinance | Historical stock data source (Yahoo Finance) |
| Pandas / NumPy | Data handling and numerical operations |
| Scikit-learn | `MinMaxScaler`, evaluation metrics (MAE, RMSE) |
| TensorFlow / Keras | LSTM model definition and training |
| Plotly | Interactive charts |
| Matplotlib | Static plots during development (training loss curves) |
| Streamlit | Web application framework |
| Git & GitHub | Version control and hosting |

---

## Machine Learning Approach

The model is a 2-layer stacked LSTM network trained on a single stock's historical closing prices:

```
Input (60-day sequence)
 → LSTM(50 units, return_sequences=True)
 → Dropout(0.2)
 → LSTM(50 units, return_sequences=False)
 → Dropout(0.2)
 → Dense(1)  →  next-day predicted price
```

- **Lookback window:** 60 trading days of history are used to predict the next day
- **Scaling:** `MinMaxScaler` is fit only on training data and reused (never refit) on test data, to avoid data leakage
- **Train/test split:** chronological (80/20), never shuffled — shuffling time-series data would let the model "see the future" during training
- **Loss function:** Mean Squared Error
- **Optimizer:** Adam
- **In-app training:** 10 epochs per ticker (a faster, lighter-weight configuration than the 20-epoch offline training script, chosen to keep the interactive app responsive)

---

## Dataset

Historical OHLCV (Open, High, Low, Close, Volume) data is downloaded live from Yahoo Finance via the `yfinance` library for whatever ticker and date range the user selects — no static dataset is bundled with the project.

---

## Installation

**Prerequisites:** Python 3.11+ and Git.

```bash
git clone https://github.com/Abhigyan-1001/stock-market-prediction.git
cd stock-market-prediction

python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate      # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Project

**Run the Streamlit app:**

```bash
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`. Enter a ticker (e.g. `AAPL`, `MSFT`, `TCS.NS`), choose a date range and forecast horizon, and click **Run Analysis**.

**Optional — standalone scripts** (used during development, still runnable independently):

```bash
python src/data_collection.py     # Test data fetching
python src/visualization.py       # Test EDA charts
python src/preprocessing.py       # Test scaling/sequence pipeline
python src/model.py               # Inspect model architecture
python -m src.train                # Train and save a standalone AAPL model
python -m src.evaluate             # Evaluate the saved model on a test set
python -m src.forecast             # Generate a standalone future forecast
```

---

## Project Structure

```
stock-market-prediction/
│
├── app.py                   # Streamlit dashboard (main application)
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/                  # Saved model artifacts from standalone training
├── notebooks/                # Scratch/exploration scripts
│
└── src/
    ├── data_collection.py   # yfinance download logic
    ├── visualization.py     # Plotly chart functions
    ├── preprocessing.py     # Scaling and sequence-building
    ├── model.py              # LSTM architecture definition
    ├── train.py               # Standalone offline training script
    ├── evaluate.py           # Standalone evaluation script
    └── forecast.py           # Recursive future forecasting logic
```

---

## Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| LSTM (50 units) | (None, 60, 50) | 10,400 |
| Dropout (0.2) | (None, 60, 50) | 0 |
| LSTM (50 units) | (None, 50) | 20,200 |
| Dropout (0.2) | (None, 50) | 0 |
| Dense (1 unit) | (None, 1) | 51 |

**Total parameters:** ~30,651

---

## Evaluation

The model is evaluated on a chronologically held-out test set (the most recent ~20% of the selected date range), which it never saw during training. Metrics are computed on the real price scale (after inverse-transforming predictions), not on the internal [0,1] scaled values.

**Example (AAPL, 2015–2024, offline 20-epoch training):**

| Metric | Value | Meaning |
|---|---|---|
| MAE | $6.54 | Average absolute dollar error per prediction |
| RMSE | $7.56 | Root mean squared error — penalizes larger misses more |
| MAPE | 3.97% | Average percentage error, comparable across different stocks |

*Note: metrics generated in-app will vary by ticker, date range, and the reduced 10-epoch in-app training configuration.*

---

## Future Improvements

- Add technical indicators (RSI, MACD, Bollinger Bands)
- Allow saving/reusing trained models across sessions instead of retraining per browser session
- Support multi-feature input (Volume, High/Low) instead of Close price alone
- Add walk-forward validation for a more rigorous evaluation
- Compare LSTM performance against simpler baselines (e.g. ARIMA, moving average)
- Deploy the app publicly (e.g. Streamlit Community Cloud)

---

## Limitations & Disclaimer

- Stock prices are influenced by countless real-world factors (news, earnings, macroeconomic events) that a model trained only on historical price sequences cannot capture.
- Strong performance on historical test data does **not** guarantee accurate future predictions — markets are not fully predictable from price history alone.
- Recursive multi-step forecasting compounds error at each step: short-horizon forecasts (1–5 days) are more trustworthy than longer ones, and forecasts often appear artificially smooth compared to real price volatility.
- This project is intended to demonstrate ML engineering and software development skills. **It is not, and should not be used as, financial advice or a trading system.**
