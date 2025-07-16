import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_breast_cancer_dataset(file_path='dataset/data.csv', num_clients=5):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop ID column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Encode target: M = 1, B = 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features and target
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create one big dataset
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Split training dataset into federated clients
    client_data = split_dataset(train_dataset, num_clients)

    # Create client DataLoaders
    client_loaders = [DataLoader(data, batch_size=16, shuffle=True) for data in client_data]

    # Create test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return client_loaders, test_loader


def split_dataset(dataset, num_clients):
    data_len = len(dataset)
    client_len = data_len // num_clients
    client_data = []

    for i in range(num_clients):
        start = i * client_len
        end = (i + 1) * client_len if i < num_clients - 1 else data_len
        client_data.append(torch.utils.data.Subset(dataset, list(range(start, end))))
    return client_data
