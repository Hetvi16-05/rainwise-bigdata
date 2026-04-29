import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    """Custom Dataset for Rainfall or Flood prediction."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Use float32 for regression, or ensure it's correct for BCE
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(file_path, task_type="classification", batch_size=64, test_size=0.2):
    """
    Loads CSV, preprocesses, and returns DataLoaders.
    task_type: 'classification' (Flood) or 'regression' (Rainfall)
    """
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.lower()
    
    # --- VIVA DEMO SPEEDUP ---
    # Downsample the massive 2.2 million row dataset to 150,000 rows
    # so the model retrains instantly during the presentation!
    if len(df) > 150000:
        df = df.sample(n=150000, random_state=42)
    
    # Add temporal awareness for Seasonal DL forecasting (Month of the year)
    if "date" in df.columns:
        # Extracts MM from YYYYMMDD integer format
        df["month"] = (df["date"] % 10000) // 100
    
    # Selecting the same features as the ML model for a fair baseline
    features = [
        "elevation_m",
        "distance_to_river_m",
        "lat",
        "lon",
        "population_2026"
    ]
    
    # Only append month if it was successfully extracted
    if "month" in df.columns:
        features.append("month")
        
    if task_type == "classification":
        # Additional features for flood classification
        features += ["rain_mm", "rain3_mm"]
        target = "flood"
        stratify_val = df[target] if target in df else None
    else:
        # Features for rainfall regression
        features += ["rain3_mm"] 
        target = "rain_mm" # Predicting daily rainfall
        stratify_val = None
    
    # Drop NAs
    df = df[features + [target]].dropna()
    
    X = df[features].values
    y = df[target].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_val
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
