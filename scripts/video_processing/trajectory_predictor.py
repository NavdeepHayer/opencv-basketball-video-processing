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
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
        super(EnhancedTrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        return self.fc(h_lstm[:, -1, :])

def load_and_prepare_data():
    # Dummy data loading function
    # Replace with actual data loading when available
    data = np.random.randn(1000, 10, 2)  # 1000 samples, 10 time steps, 2 features per step
    labels = np.random.randn(1000, 2)    # 1000 labels, 2 coordinates per label
    return train_test_split(data, labels, test_size=0.2, random_state=42)

def predict_future_position(lstm_model, trajectory_data):
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
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def save_model(model, path):
    # Save model state dict directly
    torch.save(model.state_dict(), path)
    logging.info(f'Model saved to {path}')

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = EnhancedTrajectoryPredictor(input_size=2)  # Adjust input_size if needed
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        for inputs, labels in train_loader:
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

