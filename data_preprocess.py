import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class BreastCancerDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_breast_cancer_dataset(file_path='datasets/data.csv', num_clients=5):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop 'id' column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Drop columns with all missing values (e.g., 'Unnamed: 32')
    df = df.dropna(axis=1, how='all')

    # Drop rows with any missing values
    df = df.dropna()
    print(f"Rows after dropna: {df.shape[0]}")

    # Map diagnosis column
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Separate features and labels
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Shape before assert: {X.shape}")

    # Confirm the shape before proceeding
    assert X.shape[1] == 30, f"Expected 30 features, found {X.shape[1]}"

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Wrap into PyTorch datasets
    full_train_dataset = BreastCancerDataset(X_train, y_train.values)
    test_dataset = BreastCancerDataset(X_test, y_test.values)

    # Split into clients
    total_samples = len(full_train_dataset)
    client_size = total_samples // num_clients
    client_loaders = []

    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i < num_clients - 1 else total_samples
        indices = list(range(start, end))
        subset = Subset(full_train_dataset, indices)
        loader = DataLoader(subset, batch_size=16, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return client_loaders, test_loader
