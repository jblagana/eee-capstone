# app.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

scaler = StandardScaler()

# -------------------------------------
# Define LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # Additional LSTM layer
        self.fc1 = nn.Linear(hidden_size, 64)  # First dense layer
        self.fc2 = nn.Linear(64, 32)  # Second dense layer
        self.fc3 = nn.Linear(32, output_size)  # Third dense layer

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)  # Pass through the second LSTM layer
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    
def preprocess_input(new_data):
    # Assuming new_data is a numpy array with shape (batch_size, sequence_length, n_features)
    new_data_scaled = scaler.transform(new_data.reshape(-1, n_features)).reshape(-1, sequence_length, n_features)
    return torch.tensor(new_data_scaled, dtype=torch.float32)

# Function to make predictions
def predict_with_model(new_data):
    with torch.no_grad():
        new_data_tensor = preprocess_input(new_data)
        test_outputs = model(new_data_tensor)
        predicted_labels = (test_outputs >= 0.5).squeeze().cpu().numpy()
    return predicted_labels
    
#----------------------------------------

input_size = 3
hidden_size = 64
output_size = 1

n_features = 3
sequence_length = 10

# Load your trained model
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model.pt'))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (assuming JSON input)
        data = request.get_json()
        input_data = torch.tensor(data['input'], dtype=torch.float32)

        # Ensure input shape matches (10, 3)
        if input_data.shape != (10, 3):
            return jsonify({'error': 'Input shape must be (10, 3)'})

        # Make predictions
        with torch.no_grad():
            output = model(input_data.unsqueeze(0))  # Add batch dimension

        return jsonify({'output': output.item()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
