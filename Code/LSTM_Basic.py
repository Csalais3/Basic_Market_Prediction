import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the preprocessed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate features and target
X_train = train_data.drop(columns=['Target']).values
y_train = train_data['Target'].values
X_test = test_data.drop(columns=['Target']).values
y_test = test_data['Target'].values

# Reshape the data for LSTM input: (samples, sequence_length, features)
sequence_length = 30  
num_features = X_train.shape[1] // sequence_length
X_train = X_train.reshape(-1, sequence_length, num_features)
X_test = X_test.reshape(-1, sequence_length, num_features)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, num_features)))  # LSTM layer
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')  # Compile the model

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the model
model.save('trained_stock_predictor.h5')
