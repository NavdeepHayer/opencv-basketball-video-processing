import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bar
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedTrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
        super(EnhancedTrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        return self.fc(h_lstm[:, -1, :])

def backup_data(filename, base_path):
    today_date = datetime.now().strftime('%Y-%m-%d')
    backup_path = os.path.join(base_path, 'raw_training_data', today_date)
    os.makedirs(backup_path, exist_ok=True)

    try:
        shutil.copy(filename, backup_path)
        logging.info(f"Backup created for {filename} at {backup_path}")
    except Exception as e:
        logging.error(f"Error while backing up data: {e}")

def load_and_prepare_data_with_features(filename='trajectory_data_with_features.npy'):
    if not os.path.exists(filename):
        logging.error(f"Data file {filename} not found.")
        raise FileNotFoundError(f"Data file {filename} not found.")

    data = np.load(filename, allow_pickle=True)
    X = []
    y = []
    for sequence in data:
        X.append(sequence[:-1])
        y.append(sequence[-1][:2])  # Only predict positions, not features

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_enhanced_lstm_model(base_path):
    X_train, X_test, y_train, y_test = load_and_prepare_data_with_features()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[2]  # Updated input size to match position + feature data
    model = EnhancedTrajectoryPredictor(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    # Save model
    today_date = datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join(base_path, 'trained_models', today_date, 'enhanced_trajectory_predictor.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model(model, save_path)

