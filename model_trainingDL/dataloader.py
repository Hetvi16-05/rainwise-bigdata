import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    """Custom Dataset for Flood prediction."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(file_path, batch_size=64, test_size=0.2):
    """Loads CSV, preprocesses, and returns DataLoaders."""
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.lower()
    
    # Selecting the same features as the ML model for a fair baseline
    features = [
        "rain_mm",
        "elevation_m",
        "distance_to_river_m",
        "lat",
        "lon"
    ]
    target = "flood"
    
    # Drop NAs
    df = df[features + [target]].dropna()
    
    X = df[features].values
    y = df[target].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Datasets
    train_dataset = FloodDataset(X_train, y_train)
    test_dataset = FloodDataset(X_test, y_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, len(features)

if __name__ == "__main__":
    # Test loading
    path = "data/processed/training_dataset_gujarat_advanced_labeled.csv"
    train_l, test_l, sc, in_dim = prepare_data(path)
    print(f"Data Loaded! Input dimension: {in_dim}")
    print(f"Train batches: {len(train_l)}")
