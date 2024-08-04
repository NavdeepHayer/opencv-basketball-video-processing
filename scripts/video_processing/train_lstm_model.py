import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bars

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(TrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

def load_and_prepare_data(filename='trajectory_data.npy'):
    data = np.load(filename, allow_pickle=True)
    X = []
    y = []
    for sequence in data:
        X.append(sequence[:-1])
        y.append(sequence[-1])

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_lstm_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'Train Loss': f'{train_loss / len(train_loader):.4f}'})
                pbar.update(1)

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

                test_outputs = model(X_batch)
                loss = criterion(test_outputs, y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), 'trajectory_predictor.pth')
    print("Model saved as trajectory_predictor.pth")

if __name__ == "__main__":
    train_lstm_model()

