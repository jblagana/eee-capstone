import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import torch

# Generate a sine wave dataset
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Reshape the data to fit the model
x = x.reshape((len(x), 1, 1))
y = y.reshape((len(y), 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model to the data
model.fit(x, y, epochs=200, verbose=0)

# Save the model
torch.save(model.state_dict(), 'LSTM.pt')
