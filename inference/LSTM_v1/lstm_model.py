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
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # Additional LSTM layer
        self.fc1 = nn.Linear(hidden_size, 64)  # First dense layer
        self.fc2 = nn.Linear(64, 32)  # Second dense layer
        self.fc3 = nn.Linear(32, 1)  # Single output neuron
        self.fc4 = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        out, (h_n, c_n) = self.lstm1(x)
        out, _ = self.lstm2(out, (h_n, c_n))  # Pass through the second LSTM layer
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

    

class LSTM_Inference():
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_size = 64
        self.model = '' #trained and validated model
        self.threshold = 0 # optimal threshold
        self.f1 = 0 # max f1_score

    def scale(self, data):
        # scale data
        return scaler.fit_transform(data.reshape(-1, self.n_features)).reshape(-1, self.sequence_length, self.n_features)
    
    def array2tensor(self, array):
        return torch.tensor(array, dtype=torch.float32)
    
    def generate_training_data(self):
        # Generates random training data
        print("Generating random training dataset...")
        n_samples = 1000
        features = np.random.rand(n_samples, self.sequence_length, self.n_features)
        labels = np.random.randint(0, 2, size=n_samples)

        # Split data into train, validation, and test sets
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        X_val, self.X_test, y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train_scaled = self.scale(self.X_train)
        X_val_scaled = self.scale(X_val)
        self.X_test_scaled = self.scale(self.X_test)

        # Convert data to PyTorch tensors
        self.X_train_tensor = self.array2tensor(self.X_train_scaled)
        self.X_val_tensor = self.array2tensor(X_val_scaled)
        self.y_train_tensor = self.array2tensor(self.y_train)
        self.y_val_tensor = self.array2tensor(y_val)

        self.X_test_tensor = self.array2tensor(self.X_test_scaled)
        self.y_test_tensor = self.array2tensor(self.y_test)

    
    def train(self, training_data, n_epochs):

        self.generate_training_data() # generate random trining data

        # Instantiate the model
        self.model = LSTMModel(self.n_features, self.hidden_size)

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Initialize empty lists to store training and validation losses
        self.train_losses = []
        self.val_losses = []

        # Training loop
        for epoch in range(n_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor)
            loss = self.criterion(outputs, self.y_train_tensor.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val_tensor)
                val_loss = self.criterion(val_outputs, self.y_val_tensor.unsqueeze(1))

            # Append losses to lists
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss.item())

            # Training progress bar
            label = f"{epoch+1}/{n_epochs} epochs"
            progress = (epoch+1)/ n_epochs
            progress_bar_len = 50
            filled_len = int(progress_bar_len * progress)
            bar = '=' * filled_len + '-' * (progress_bar_len - filled_len)
            sys.stdout.write(f'\rTraining in progress: [{bar}] {progress * 100:.1f}% [{label}]')
            sys.stdout.flush()

        self.validate()
        
        print("\nFinished training and validation.")

        return self.model


    def load_model(self,model_path):
        self.model = LSTMModel(self.n_features, self.hidden_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.validate()


    def validate(self):
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            self.test_outputs = self.model(self.X_test_tensor)
            self.test_loss = self.criterion(self.test_outputs, self.y_test_tensor.unsqueeze(1))

        # Evaluation
        self.precision, self.recall, self.thresholds = precision_recall_curve(self.y_test_tensor, self.test_outputs)

        # Calculate F1 scores for each threshold
        self.f1_scores = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        # Find the optimal threshold that maximizes the F1 score
        self.threshold = self.thresholds[self.f1_scores.argmax()]
        self.f1 = self.f1_scores.max()


    def save_model(self):
        # Save the trained LSTM model
        torch.save(self.model.state_dict(), f'lstm_model_{self.threshold:.2f}.pt')
        print(f'Model saved at {os.getcwd()}')



    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot training and validation losses
        axs[0, 0].plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
        axs[0, 0].plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].set_title("Training and Validation Losses")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # Plot precision-recall curve
        axs[0, 1].plot(self.recall, self.precision, label='Precision-Recall Curve')
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].set_title('Precision-Recall Curve')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        # Plot F1 curve
        axs[1, 0].plot(self.thresholds, self.f1_scores[:-1], label='F1 Score')
        axs[1, 0].axvline(x=self.threshold, color='r', linestyle='--', label='Optimal Threshold')
        axs[1, 0].set_xlabel('Threshold')
        axs[1, 0].set_ylabel('F1 Score')
        axs[1, 0].set_title('F1 Curve')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # Plot confusion matrix
        test_outputs = (self.test_outputs >= self.threshold).float()
        cm = confusion_matrix(self.y_test_tensor, test_outputs)
        labels = ['No Robbery', 'Robbery']
        cm_transposed = cm.T
        sns.heatmap(cm_transposed, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=axs[1, 1])
        axs[1, 1].set_title('Confusion Matrix')
        axs[1, 1].set_xlabel('True Labels')
        axs[1, 1].set_ylabel('Predicted Labels')

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()



    def predict(self, i, input_data):
        try:
            # Preprocess data
            # input_data = self.scale(input_data).tolist()[0]
            # input_data = self.array2tensor(input_data)

            # Ensure input shape matches
            if input_data.shape != (self.sequence_length, self.n_features):
                print('error: ', f'Input shape must be {(self.sequence_length, self.n_features)}. Yours has shape {input_data.shape}')

            # Make predictions
            with torch.no_grad():
                output = self.model(input_data.unsqueeze(0))  # Add batch dimension
                prediction = (output).squeeze().cpu().numpy()

            print(f"prediction: {prediction} | actual: {self.y_train[i]}")


        except Exception as e:
            print('error: ', str(e))
    
