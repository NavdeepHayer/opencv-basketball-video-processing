import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(TrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

def load_data(filename='trajectory_data.npy'):
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
    X_train, X_test, y_train, y_test = load_data()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = TrajectoryPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    torch.save(model.state_dict(), 'trajectory_predictor.pth')
    print("Model saved as trajectory_predictor.pth")

if __name__ == "__main__":
    train_lstm_model()
