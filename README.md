# 📈 Implied Volatility Visualizer

This project is an interactive Streamlit-based visualizer for exploring the implied volatility surface of stock options. It fetches real-time options data from Yahoo Finance using `yfinance`, calculates time to expiration, and renders a 3D surface plot of implied volatility against strike price and maturity.

## 🔧 Features

- 📊 Select a stock from a pre-defined list
- 🧮 Calculate time to expiration for each option
- 🖼️ Plot a 3D implied volatility surface (strike vs. time vs. IV)
- 🧠 Normalize moneyness for relative comparisons
- 🔄 Dynamically updates based on user stock selection

## 🛠️ Dependencies

- Python 3.7+
- streamlit
- yfinance
- pandas
- numpy
- matplotlib

Install them with:

```bash
