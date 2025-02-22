import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Preprocessesing import MinMaxScaling 

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Load the scaler used during preprocessing
scaler = MinMaxScaling(feature_range= (0, 1))

# Define sequence length
sequence_length = 30

# Streamlit app
st.title("Stock Predictor")
st.write("Enter the past 30 days of stock data to predict the next day's closing price.")

# Input fields for past stock data
st.header("Input Past Stock Data")
input_data = []
for i in range(sequence_length):
    open_price = st.number_input(f"Day {i+1} Open Price", value=100.0)
    high_price = st.number_input(f"Day {i+1} High Price", value=105.0)
    low_price = st.number_input(f"Day {i+1} Low Price", value=95.0)
    close_price = st.number_input(f"Day {i+1} Close Price", value=102.0)
    adj_close = st.number_input(f"Day {i+1} Adj Close", value=102.0)
    volume = st.number_input(f"Day {i+1} Volume", value=1000000.0)
    input_data.append([open_price, high_price, low_price, close_price, adj_close, volume])

# Convert input to DataFrame
input_df = pd.DataFrame(input_data, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Add feature engineering (if needed)
input_df['Returns'] = input_df['Close'].pct_change()
input_df['MA_10'] = input_df['Close'].rolling(window=10).mean()
input_df['Volatility'] = input_df['Returns'].rolling(window=10).std()
input_df = input_df.dropna()

# Scale the input data
scaled_input = scaler.transform(input_df)

# Reshape for LSTM input
scaled_input = scaled_input.reshape(1, sequence_length, -1)  # Shape: (1, sequence_length, num_features)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]  # Inverse scaling
    st.success(f"Predicted Next Day's Closing Price: {predicted_price:.2f}")