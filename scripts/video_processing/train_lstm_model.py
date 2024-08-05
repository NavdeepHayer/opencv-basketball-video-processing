import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedTrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=4):
        super(EnhancedTrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        return self.fc(h_lstm[:, -1, :])

def load_and_prepare_data(file_path='trajectory_data_with_features.npy'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")

    data = np.load(file_path, allow_pickle=True)
    X = []
    y = []
    for sequence in data:
        X.append(sequence[:-1])  # Features including center_x, center_y, width, height, and others
        y.append(sequence[-1])  # Predict full feature vector including position, size, and feature points

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def predict_future_features(lstm_model, trajectory_data):
    # Convert trajectory data to tensor and move to the model's device
    trajectory_tensor = torch.tensor(trajectory_data, dtype=torch.float32)
    
    # Ensure the tensor is on the same device as the model
    device = next(lstm_model.parameters()).device
    trajectory_tensor = trajectory_tensor.to(device).unsqueeze(0)
    
    prediction = lstm_model(trajectory_tensor)
    return prediction.detach().cpu().numpy().flatten()

def load_lstm_model(path, input_size, device='cpu'):
    # Load LSTM model from the given path
    model = EnhancedTrajectoryPredictor(input_size)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def save_model(model, path):
    # Save model state dict directly
    torch.save(model.state_dict(), path)
    logging.info(f'Model saved to {path}')

def main():
    # Use the new data loading function with updated feature set
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    input_size = X_train.shape[2]  # Updated input size to match new feature set
    output_size = y_train.shape[1]  # Adjust output size to predict all features
    model = EnhancedTrajectoryPredictor(input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        for inputs, labels in DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save model
    today_date = datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join('trained_models', today_date, 'model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model(model, save_path)

if __name__ == "__main__":
    main()

