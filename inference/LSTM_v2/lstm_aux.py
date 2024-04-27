import os
import sys
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

scaler = StandardScaler()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Single output neuron for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state for prediction (summary)
        out = self.sigmoid(out) # Apply sigmoid activation
        return out