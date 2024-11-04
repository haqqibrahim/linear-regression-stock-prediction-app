import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

from Agent import model_analysis_agent

import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Title and description
st.title("📈 Stock Price Prediction using Linear Regression")
st.markdown("""
This app predicts stock prices using a Linear Regression model based on moving averages.
Select a stock symbol and customize the parameters to see the predictions.
""")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Stock symbol input
    ticker = st.text_input("Stock Symbol", "AAPL").upper()
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )
    
    # Moving average window selection
    ma_window = st.slider("Moving Average Window", 
                         min_value=5, 
                         max_value=200, 
                         value=50,
                         help="Number of days for calculating moving average")
    
    # Test size selection
    test_size = st.slider("Test Size", 
                         min_value=0.1, 
                         max_value=0.4, 
                         value=0.2,
                         help="Proportion of dataset to include in the test split")

try:
    # Fetch stock data
    @st.cache_data
    def load_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data['Date'] = data.index
        return data

    data = load_data(ticker, start_date, end_date)

    # Feature Engineering
    data[f'{ma_window}_day_MA'] = data['Close'].rolling(window=ma_window).mean()
    data = data.dropna()

    # Display raw data
    with st.expander("Show Raw Data"):
        st.dataframe(data)

    # Prepare features and target
    X = data[[f'{ma_window}_day_MA']].values
    y = data['Close'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"{float(mse):.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{float(rmse):.2f}")
    with col3:
        st.metric("R² Score", f"{float(r2):.2f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Close'], color='blue', label='Actual Prices')
    ax.plot(data['Date'].iloc[-len(y_test):], y_pred, 
            color='red', linestyle='dashed', label='Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'Linear Regression Model for {ticker} Stock Price Prediction')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display plot
    st.pyplot(fig)

    # Additional insights
    st.subheader("Model Insights")

    model_insight = model_analysis_agent(f"""
    - The model uses the {ma_window}-day moving average to predict stock prices
    - Coefficient: {float(model.coef_[0]):.4f}
    - Intercept: {float(model.intercept_):.4f}
    """)
    st.write(model_insight)

except Exception as e:
    st.error(f"""
    An error occurred: {str(e)}
    
    Please make sure:
    - The stock symbol is valid
    - The date range is valid
    - There is enough data for the selected moving average window
    """) 