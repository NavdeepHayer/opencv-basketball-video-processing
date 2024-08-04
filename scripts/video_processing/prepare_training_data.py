# prepare_training_data.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data_for_training(file_path='trajectory_data.npy', seq_length=10):
    """
    Prepare data for LSTM training by creating sequences and targets from trajectory data.

    Args:
    - file_path (str): Path to the file containing collected trajectory data.
    - seq_length (int): Length of the sequences to generate.

    Returns:
    - DataLoader: A DataLoader containing the prepared sequences and targets.
    """
    # Load collected data
    data = np.load(file_path, allow_pickle=True)

    # Prepare sequences and targets
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = [point[0] for point in data[i:i + seq_length]]
        target = data[i + seq_length][1]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # Create a DataLoader
    dataset = TensorDataset(torch.from_numpy(sequences), torch.from_numpy(targets))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return data_loader

if __name__ == "__main__":
    # Example usage
    train_loader = prepare_data_for_training('trajectory_data.npy')
    print(f"Prepared {len(train_loader.dataset)} sequences for training.")
