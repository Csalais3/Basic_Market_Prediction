import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Constants
SEQUENCE_LENGTH = 7        # Weekly pattern (can be adjusted)
FEATURE_RANGE = (0, 1)     # MinMax scaling range
TEST_SIZE = 0.2            # 80-20 train-test split
MODEL_PATH = 'lstm_stock_predictor.h5'

class StockPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        self.sequence_length = SEQUENCE_LENGTH
        
        # Initialize data structures
        self.raw_data = None
        self.cleaned_data = None
        self.scaled_data = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def load_and_preprocess(self):
        """Load and preprocess the stock data"""
        # Load raw data
        self.raw_data = pd.read_csv(self.data_path)
        
        # Data cleaning
        self.raw_data = self.raw_data.ffill()  # Forward fill missing values
        
        # Feature engineering
        self.raw_data['Returns'] = self.raw_data['Close'].pct_change()
        self.raw_data['MA_10'] = self.raw_data['Close'].rolling(window=10).mean()
        self.raw_data['Volatility'] = self.raw_data['Returns'].rolling(window=10).std()
        self.raw_data = self.raw_data.dropna()

        # Select features and scale
        features = self.raw_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                                 'Returns', 'MA_10', 'Volatility']]
        self.scaled_data = self.scaler.fit_transform(features)

    def create_sequences(self):
        """Create time sequences for LSTM training"""
        X, y = [], []
        for i in range(len(self.scaled_data) - self.sequence_length):
            X.append(self.scaled_data[i:i+self.sequence_length])
            y.append(self.scaled_data[i+self.sequence_length, 3])  # Predict 'Close' price
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        split = int(len(X) * (1 - TEST_SIZE))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

    def build_model(self):
        """Build and compile LSTM model"""
        self.model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, self.X_train.shape[2])),  # LSTM layer
            Dropout(0.3),  # Dropout layer
            Dense(32, activation='relu'),  # Dense layer
            Dense(1)  # Output layer
        ])
    
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
         )
    
    def train(self, epochs=50, batch_size=32):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self):
        """Evaluate model performance"""
        # Load best model weights
        self.model = load_model(MODEL_PATH)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Inverse scaling
        y_test_actual = self.scaler.inverse_transform(
            np.concatenate([np.zeros((len(self.y_test), self.scaled_data.shape[1]-1), 
                           self.y_test.reshape(-1,1))], axis=1))[:,3]
        
        y_pred_actual = self.scaler.inverse_transform(
            np.concatenate([np.zeros((len(y_pred), self.scaled_data.shape[1]-1), 
                           y_pred.reshape(-1,1))], axis=1))[:,3]

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        print(f"Test RMSE: {rmse:.2f}")
        
        # Plot results
        plt.figure(figsize=(12,6))
        plt.plot(y_test_actual, label='Actual Prices')
        plt.plot(y_pred_actual, label='Predicted Prices')
        plt.title('Stock Price Prediction Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run_pipeline(self):
        """Complete training pipeline"""
        self.load_and_preprocess()
        self.create_sequences()
        self.build_model()
        self.train()
        self.evaluate()

if __name__ == "__main__":
    predictor = StockPredictor("/Users/csalais3/Downloads/Basic_Market_Prediction/Data/SPX.csv")
    
    predictor.run_pipeline()