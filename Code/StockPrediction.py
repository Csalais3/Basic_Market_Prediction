import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Load the trained LSTM model
model = load_model("/content/lstm_stock_predictor.keras")

# Load the scaler used during preprocessing
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define sequence length (number of days required)
sequence_length = 30

# Create default data: 30 rows with sample values.
# Each row: [open, high, low, close, adj close, volume]
data = pd.read_csv('/content/SPX.csv')
recent_features = data.tail(sequence_length)
default_data = [[row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume']] for _, row in recent_features.iterrows()]
def predict_next_close(stock_data):
    """
    Expects stock_data as a list-of-lists with columns:
    ['open', 'high', 'low', 'close', 'adj close', 'volume']
    for the past 30 days.
    """
    # Create DataFrame from input
    df = pd.DataFrame(stock_data, columns= ['open', 'high', 'low', 'close', 'adj close', 'volume'])

    # Feature engineering: create additional columns
    df['returns'] = df['close'].pct_change()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['volatility'] = df['returns'].rolling(window=10).std()

    # Fill missing values instead of dropping rows
    df.fillna(method= 'bfill', inplace= True)

    # Check if we have exactly 30 rows
    if len(df) != sequence_length:
        return "After preprocessing, the input data does not have enough rows.", None

    # Scale the input data (ensure the feature list is in the same order as during training)
    features = ['open', 'high', 'low', 'close', 'adj close', 'volume', 'returns', 'ma_10', 'volatility']
    scaled_input = scaler.transform(df[features])

    # Reshape for LSTM input: (1, sequence_length, num_features)
    scaled_input = scaled_input.reshape(1, sequence_length, -1)

    # Make prediction using the model
    prediction = model.predict(scaled_input)

    # To inverse transform the prediction, create a dummy array placing the predicted value in the proper column (index 3 corresponds to 'close')
    dummy = np.zeros((1, len(features)))
    dummy[0, 3] = prediction[0, 0]
    predicted_price = scaler.inverse_transform(dummy)[0, 3]

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot the past 30 days' closing prices
    ax.plot(range(1, sequence_length + 1), df['close'], marker= 'o', label= "Past 30 Days Closing Prices")
    # Instead of a horizontal line, plot an orange dot for the predicted value on day 31
    ax.scatter(sequence_length + 1, predicted_price, color='orange', s = 100, label="Predicted Next Day Close")
    ax.set_xlabel("Day")
    ax.set_ylabel("Stock Price")
    ax.set_title("Stock Price Trend")
    ax.legend()
    ax.grid(True)

    return predicted_price, fig

# Create a Gradio interface with the default data pre-populated
interface = gr.Interface(
    fn= predict_next_close,
    inputs= gr.Dataframe(
        headers= ["open", "high", "low", "close", "adj close", "volume"],
        value= default_data,
        row_count= sequence_length,
        col_count= 6,
        label= "Input Past 30 Days Stock Data"
    ),
    outputs=[
        gr.Number(label= "Predicted Next Day's Closing Price"),
        gr.Plot(label= "Stock Price Trend")
    ],
    title= "Stock Predictor",
    description= "Enter the past 30 days of stock data to predict the next day's closing price."
)

# Launch the interface (in Colab it will provide a shareable link)
interface.launch()