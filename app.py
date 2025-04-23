import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

# Configure page settings
st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Market Prediction App")

# Add a sidebar for configuration
st.sidebar.header("Settings")

# Add sample data option
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=False)

# Technical Analysis Parameters
window_size_ma1 = st.sidebar.slider("Short MA Window", min_value=5, max_value=50, value=7)
window_size_ma2 = st.sidebar.slider("Long MA Window", min_value=10, max_value=200, value=21)
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)
prediction_days = st.sidebar.slider("Signal Prediction Days", min_value=1, max_value=10, value=3)

# Add information about the app
st.sidebar.header("About")
st.sidebar.info("""
This app predicts stock market movements using:
- Random Forest for Buy/Sell signals
- Linear Regression for price forecasting
""")

def load_and_validate_data(file):
    try:
        df = pd.read_csv(file)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check column names (case-insensitive)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.title()
        
        # Rename common column variations
        column_mappings = {
            'Shares Traded': 'Volume1',
            'Value': 'Volume2',
            'Trade': 'Volume3'
        }
        df = df.rename(columns=column_mappings)
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# File upload with error handling
if use_sample_data:
    uploaded_file = "niftyprediction.2.csv"
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using sample data")
else:
    uploaded_file = st.file_uploader("Upload your stock CSV", type="csv", key="stock_data_uploader")
    if uploaded_file:
        df = load_and_validate_data(uploaded_file)
    else:
        st.info("👆 Please upload a CSV file with stock data")
        st.stop()

if df is not None:
    try:
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Calculate technical indicators
        df[f'MA{window_size_ma1}'] = df['Close'].rolling(window=window_size_ma1, min_periods=1).mean()
        df[f'MA{window_size_ma2}'] = df['Close'].rolling(window=window_size_ma2, min_periods=1).mean()
        df['Future_Close'] = df['Close'].shift(-prediction_days)
        df['Signal'] = np.where(df['Future_Close'] > df['Close'], 1, 0)
        
        # Remove any remaining NaN values
        df.dropna(inplace=True)

        if len(df) < 30:
            st.error("❌ Not enough data points. Please upload a CSV with at least 30 days of data.")
            st.stop()

        # Show data preview
        st.subheader("Data Preview")
        st.write(df.head())
        
        # Random Forest Classification
        st.subheader("Buy/Sell Signal Prediction (Random Forest)")
        features = ['Open', 'High', 'Low', 'Close', 'Volume', f'MA{window_size_ma1}', f'MA{window_size_ma2}']
        X = df[features]
        y = df['Signal']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Display classification results
        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        
        with col2:
            st.write("Recent Predictions:")
            df_test = df.iloc[-len(y_test):].copy()
            df_test['Prediction'] = y_pred
            st.write(df_test[['Close', 'Prediction']].tail())

        # Plot Buy/Sell signals
        st.subheader("📈 Buy (1) / Sell (0) Signal Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_test.index, df_test['Close'], label='Close Price')
        ax.scatter(df_test[df_test['Prediction'] == 1].index, 
                  df_test[df_test['Prediction'] == 1]['Close'], 
                  color='green', label='Buy Signal', marker='^')
        ax.scatter(df_test[df_test['Prediction'] == 0].index, 
                  df_test[df_test['Prediction'] == 0]['Close'], 
                  color='red', label='Sell Signal', marker='v')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        st.pyplot(fig)
        plt.close()

        # Linear Regression Forecast
        st.subheader(f"🔮 {forecast_days}-Day Close Price Forecast")
        df[f'Target_Close_{forecast_days}d'] = df['Close'].shift(-forecast_days)
        df.dropna(inplace=True)

        Xf = df[features]
        yf = df[f'Target_Close_{forecast_days}d']

        # Prepare forecast data
        X_train_f, X_forecast = Xf[:-forecast_days], Xf[-forecast_days:]
        y_train_f = yf[:-forecast_days]

        # Train Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train_f, y_train_f)
        forecast = lr.predict(X_forecast)

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': df.index[-forecast_days:],
            'Actual_Close': df['Close'].iloc[-forecast_days:],
            'Forecast_Close': forecast
        }).set_index('Date')

        st.write(f"{forecast_days}-Day Forecast vs Actual:")
        st.write(forecast_df)

        # Plot forecast results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_df.index, forecast_df['Actual_Close'], label='Actual')
        ax.plot(forecast_df.index, forecast_df['Forecast_Close'], label='Forecast', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please check your data format and try again.")


