"""
RAINWISE: Seasonal Rainfall Prediction Model
============================================
This model predicts daily rainfall for ANY date range using:
  - Cyclic day-of-year encoding (sin/cos) — handles seasonality for any year
  - City geography (lat, lon, elevation, distance to river)

No dependency on past rain values (rain3), so the model works
cleanly for any future date range the user specifies.
"""
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
DATA_PATH  = "data/processed/training_dataset_gujarat_advanced_labeled.csv"
MODEL_PATH = "DLmodels/seasonal_rainfall_dnn.pth"
SCALER_PATH= "DLmodels/seasonal_rainfall_scaler.pkl"
NROWS      = 300000     # fast enough for Viva (<90 sec on MPS)
BATCH_SIZE = 2048
EPOCHS     = 30
LR         = 0.001
PATIENCE   = 6
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==================== MODEL ====================
class SeasonalRainfallDNN(nn.Module):
    """
    Fully-connected DNN for daily rainfall regression.
    Inputs (7): sin_doy, cos_doy, lat, lon, elevation_m, distance_to_river_m, month
    Output (1): predicted rain_mm
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==================== DATASET ====================
class RainfallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ==================== TRAIN ====================
def train():
    os.makedirs("DLmodels", exist_ok=True)
    os.makedirs("outputs/dl", exist_ok=True)

    print(f"\n📂 Loading {NROWS:,} rows from training dataset...")
    cols = ["date", "lat", "lon", "elevation_m", "distance_to_river_m", "rain_mm"]
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False, nrows=NROWS*3)
    df.columns = df.columns.str.lower()
    df = df.dropna()

    # Convert integer date YYYYMMDD → proper date
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # Cyclic day-of-year features (key innovation: works for any future year)
    df["doy"]     = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["month"]   = df["date"].dt.month

    # Downsample for speed
    if len(df) > NROWS:
        df = df.sample(n=NROWS, random_state=42)

    features = ["sin_doy", "cos_doy", "lat", "lon", "elevation_m", "distance_to_river_m", "month"]
    target   = "rain_mm"

    X = df[features].values
    y = df[target].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved → {SCALER_PATH}")

    train_loader = DataLoader(RainfallDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(RainfallDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    model     = SeasonalRainfallDNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    print(f"🧠 Training SeasonalRainfallDNN on {DEVICE}  ({NROWS:,} rows, {EPOCHS} max epochs)...")
    best_loss, patience_ctr = float("inf"), 0
    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                v_loss += criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)).item()

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        history["train"].append(avg_t)
        history["val"].append(avg_v)
        scheduler.step(avg_v)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:02d}/{EPOCHS}] | Train MSE: {avg_t:.4f} | Val MSE: {avg_v:.4f}")

        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), MODEL_PATH)
            patience_ctr = 0
            print(f"  🚩 New best Val MSE: {avg_v:.4f} — model saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  🛑 Early stop at epoch {epoch+1}.")
                break

    print(f"\n✅ Training complete. Best Val MSE: {best_loss:.4f}")
    print(f"   Model → {MODEL_PATH}")

    # Save loss curves
    plt.figure(figsize=(9,4))
    plt.plot(history["train"], label="Train MSE")
    plt.plot(history["val"],   label="Val MSE")
    plt.yscale("log"); plt.grid(alpha=0.3); plt.legend()
    plt.title("SeasonalRainfallDNN — Loss Curves")
    plt.tight_layout()
    plt.savefig("outputs/dl/seasonal_rainfall_curves.png")
    print("📈 Loss curves → outputs/dl/seasonal_rainfall_curves.png")

if __name__ == "__main__":
    train()
