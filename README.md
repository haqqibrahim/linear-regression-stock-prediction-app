# 📈 Stock Price Prediction App

A Streamlit web application that predicts stock prices using Linear Regression based on moving averages.

## Features

- Real-time stock data fetching using yfinance
- Customizable moving average window
- Interactive date range selection
- Model performance metrics (MSE, RMSE, R² Score)
- Visual representation of actual vs predicted prices
- AI-powered model analysis insights

## Installation

1. Clone the repository:
```bash
git clone https://github.com/haqqibrahim/linear-regression-stock-prediction-app.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your GROQ API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your GROQ API key: `GROQ_API_KEY = "your-api-key"`

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Access the app in your browser (typically at `http://localhost:8501`)

3. Configure the parameters:
   - Enter a stock symbol (e.g., AAPL, GOOGL)
   - Select date range
   - Adjust moving average window
   - Modify test size split

## Project Structure

```
├── streamlit_app.py    # Main application file
├── Agent.py            # AI analysis agent implementation
├── requirements.txt    # Project dependencies
└── .streamlit/
    └── secrets.toml    # Configuration secrets
```

## Dependencies

- streamlit
- yfinance
- numpy
- pandas
- matplotlib
- scikit-learn
- phi-agent
- groq

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```