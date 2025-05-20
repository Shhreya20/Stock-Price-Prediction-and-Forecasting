# Stock-Price-Prediction-and-Forecasting

# 📈 Stock Price Prediction with LSTM & Streamlit

This project uses an LSTM (Long Short-Term Memory) model to predict stock prices based on historical data. It features an interactive Streamlit app for entering stock ticker symbols and date ranges, visualizing trends, and forecasting future prices.

---

## 🚀 Features

- Input: Ticker symbol and custom date range
- Output: Graphs of all stock data columns (Open, High, Low, Close, Volume)
- Deep learning model: LSTM with dropout layers for regularization
- Evaluation metrics: RMSE and R² Score
- Visuals: Actual vs. predicted stock prices
- Forecasting: Next 7 days of future prices

---

## 🛠️ Technologies Used

- Python 3.x
- Streamlit
- YFinance
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run stock_predictor_app.py
```


## 📬 Author

Created by [Shreya Jain] – feel free to reach out with feedback or questions!
