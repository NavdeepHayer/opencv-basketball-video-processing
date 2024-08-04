import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filename='trajectory_data.npy'):
    data = np.load(filename, allow_pickle=True)
    X = []
    y = []
    for sequence in data:
        X.append(sequence[:-1])  # Use all but the last position for input
        y.append(sequence[-1])   # Use the last position for output

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)
