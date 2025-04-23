import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.title("📈 Stock Market Prediction App")

uploaded_file = st.file_uploader("Upload your stock CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(df.head())

    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['Future_Close'] = df['Close'].shift(-3)
        df['Signal'] = np.where(df['Future_Close'] > df['Close'], 1, 0)
        df.dropna(inplace=True)

        st.subheader("Buy/Sell Signal Prediction (Random Forest)")
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21']
        X = df[features]
        y = df['Signal']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        df_test = df.iloc[-len(y_test):].copy()
        df_test['Prediction'] = y_pred
        st.write(df_test[['Close', 'Prediction']].tail())

        st.subheader("📈 Buy (1) / Sell (0) Signal Plot")
        plt.figure(figsize=(10, 4))
        plt.plot(df_test.index, df_test['Close'], label='Close Price')
        plt.scatter(df_test.index, df_test['Prediction']*df_test['Close'], color='green', label='Buy Signal', marker='^')
        plt.legend()
        st.pyplot(plt)

        st.subheader("🔮 7-Day Close Price Forecast (Linear Regression)")
        df['Target_Close_7d'] = df['Close'].shift(-7)
        df.dropna(inplace=True)

        Xf = df[features]
        yf = df['Target_Close_7d']

        X_train_f, X_forecast = Xf[:-7], Xf[-7:]
        y_train_f = yf[:-7]

        lr = LinearRegression()
        lr.fit(X_train_f, y_train_f)
        forecast = lr.predict(X_forecast)

        forecast_df = pd.DataFrame({
            'Date': df.index[-7:],
            'Actual_Close': df['Close'].iloc[-7:],
            'Forecast_Close': forecast
        }).set_index('Date')

        st.write(forecast_df)

        st.subheader("📊 Forecast vs Actual")
        plt.figure(figsize=(10, 4))
        plt.plot(forecast_df.index, forecast_df['Actual_Close'], label='Actual')
        plt.plot(forecast_df.index, forecast_df['Forecast_Close'], label='Forecast')
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        
        import pandas as pd
import streamlit as st

# Read the CSV file uploaded by the user
uploaded_file = st.file_uploader("Upload your stock CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Show the first few rows to confirm
    st.subheader("Preview of Data")
    st.write(df.head())
    

    # Further processing like predictions or forecasts can go here


