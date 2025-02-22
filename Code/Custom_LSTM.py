import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pickle # Import the pickle library


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
        
        # Clean column names: remove extra spaces and convert to lowercase
        self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
        print("Columns found in CSV:", self.raw_data.columns.tolist())
        
        # Data cleaning: forward-fill missing values
        self.raw_data = self.raw_data.ffill()
        
        # Feature engineering using the 'close' column (now lowercase)
        self.raw_data['returns'] = self.raw_data['close'].pct_change()
        self.raw_data['ma_10'] = self.raw_data['close'].rolling(window=10).mean()
        self.raw_data['volatility'] = self.raw_data['returns'].rolling(window=10).std()
        self.raw_data = self.raw_data.dropna()

        # Select features and scale
        features = self.raw_data[['open', 'high', 'low', 'close', 'adj close', 'volume', 
                                    'returns', 'ma_10', 'volatility']]
        self.scaled_data = self.scaler.fit_transform(features)

    def create_sequences(self):
        """Create time sequences for LSTM training"""
        X, y = [], []
        for i in range(len(self.scaled_data) - self.sequence_length):
            X.append(self.scaled_data[i:i+self.sequence_length])
            y.append(self.scaled_data[i+self.sequence_length, 3])  # Predict 'close' price (index 3)
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
            loss='mean_squared_error',
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
        
        # Inverse scaling: create dummy arrays to place predictions in the correct column.
        # The 'close' price is at index 3.
        y_test_expanded = np.zeros((len(self.y_test), self.scaled_data.shape[1]))
        y_test_expanded[:, 3] = self.y_test
        
        y_pred_expanded = np.zeros((len(y_pred), self.scaled_data.shape[1]))
        y_pred_expanded[:, 3] = y_pred[:, 0]
        
        y_test_actual = self.scaler.inverse_transform(y_test_expanded)[:, 3]
        y_pred_actual = self.scaler.inverse_transform(y_pred_expanded)[:, 3]

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
        
        # Save the fitted scaler using pickle
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f) 

if __name__ == "__main__":
    predictor = StockPredictor("")
    predictor.run_pipeline()
