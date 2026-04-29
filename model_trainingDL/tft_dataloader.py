"""
RAINWISE — TFT Sequence Dataloader
====================================
Converts the Gujarat flat CSV into (static, temporal_sequence, target) triples
suitable for TFT training.

Sequence construction:
  - For each location (lat×lon grid point), sort by date.
  - Build rolling windows of SEQ_LEN consecutive days.
  - Target: rain_mm on the LAST day of the window.

Static features  (per location, time-invariant):
    lat, lon, elevation_m, distance_to_river_m

Temporal features (change every day, SEQ_LEN steps):
    sin_doy, cos_doy, month_sin, month_cos,
    rain3_mm, rain7_mm, precip_mm
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


# ─── Column names ───────────────────────────────────────────
STATIC_COLS   = ["lat", "lon", "elevation_m", "distance_to_river_m"]
TEMPORAL_COLS = ["sin_doy", "cos_doy", "month_sin", "month_cos",
                 "rain3_mm", "rain7_mm", "precip_mm"]
TARGET_COL    = "rain_mm"


# ─── Dataset ────────────────────────────────────────────────
class TFTSequenceDataset(Dataset):
    def __init__(self, static_arr, temporal_arr, target_arr):
        """
        static_arr  : (N, num_static)
        temporal_arr: (N, seq_len, num_temporal)
        target_arr  : (N,)
        """
        self.static   = torch.tensor(static_arr,   dtype=torch.float32)
        self.temporal = torch.tensor(temporal_arr, dtype=torch.float32)
        self.target   = torch.tensor(target_arr,   dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.static[idx], self.temporal[idx], self.target[idx]


# ─── Main preparation function ───────────────────────────────
def prepare_tft_data(
    file_path: str,
    seq_len:   int   = 14,
    nrows:     int   = 200_000,
    test_size: float = 0.15,
    batch_size: int  = 512,
    scaler_path: str = None,
):
    """
    Returns:
        train_loader, val_loader, static_scaler, temporal_scaler,
        num_static_vars, num_temporal_vars
    """
    print(f"  📂 Reading CSV ({nrows:,} rows cap)...")
    needed_cols = ["date", "lat", "lon", "elevation_m",
                   "distance_to_river_m", "rain_mm",
                   "rain3_mm", "rain7_mm", "precip_mm"]

    df = pd.read_csv(file_path, usecols=needed_cols,
                     low_memory=False, nrows=nrows * 4)
    df.columns = df.columns.str.lower()
    df = df.dropna()

    # ── Date features ────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"].astype(str),
                                format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    doy           = df["date"].dt.dayofyear
    month         = df["date"].dt.month
    df["sin_doy"] = np.sin(2 * np.pi * doy   / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy   / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Clip rainfall outliers (Gujarat max ~400mm/day) ──────
    df[TARGET_COL]    = df[TARGET_COL].clip(0, 400)
    df["rain3_mm"]    = df["rain3_mm"].clip(0, 1200)
    df["rain7_mm"]    = df["rain7_mm"].clip(0, 2800)
    df["precip_mm"]   = df["precip_mm"].clip(0, 400)

    # ── Build location ID for grouping ───────────────────────
    df["loc_id"] = (df["lat"].round(4).astype(str) + "_" +
                    df["lon"].round(4).astype(str))

    # ── Downsample if still over cap ─────────────────────────
    n_locs = df["loc_id"].nunique()
    if len(df) > nrows:
        # Keep the top `per_loc` days per location (sorted by date)
        per_loc  = max(seq_len + 5, nrows // n_locs)
        chunks   = []
        for loc, grp in df.groupby("loc_id", sort=False):
            chunks.append(grp.sort_values("date").head(per_loc))
        df = pd.concat(chunks, ignore_index=True)
        n_locs = df["loc_id"].nunique()

    print(f"  ✅ Loaded {len(df):,} rows | {n_locs} unique locations")

    # ── Scale static features ─────────────────────────────────
    static_scaler = StandardScaler()
    df[STATIC_COLS] = static_scaler.fit_transform(df[STATIC_COLS])

    # ── Scale temporal features ───────────────────────────────
    temporal_scaler = StandardScaler()
    df[TEMPORAL_COLS] = temporal_scaler.fit_transform(df[TEMPORAL_COLS])

    # ── Build sequences per location ─────────────────────────
    print(f"  🔧 Building {seq_len}-day sequences...")
    static_seqs   = []
    temporal_seqs = []
    targets       = []

    for loc_id, grp in df.groupby("loc_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < seq_len + 1:
            continue

        # Static values are constant for a location — take first row
        static_vals = grp[STATIC_COLS].iloc[0].values   # (num_static,)

        temp_vals = grp[TEMPORAL_COLS].values             # (T, num_temporal)
        rain_vals = grp[TARGET_COL].values                # (T,)

        for i in range(len(grp) - seq_len):
            window_temporal = temp_vals[i : i + seq_len]  # (seq_len, num_temporal)
            target_rain     = rain_vals[i + seq_len]       # scalar

            static_seqs.append(static_vals)
            temporal_seqs.append(window_temporal)
            targets.append(target_rain)

    static_arr   = np.array(static_seqs,   dtype=np.float32)   # (N, num_static)
    temporal_arr = np.array(temporal_seqs, dtype=np.float32)   # (N, seq_len, num_temporal)
    target_arr   = np.array(targets,       dtype=np.float32)   # (N,)

    print(f"  ✅ Sequences built: {len(target_arr):,} samples")
    print(f"     Static shape  : {static_arr.shape}")
    print(f"     Temporal shape: {temporal_arr.shape}")

    # ── Train / Val split (time-aware: last 15% for validation) ──
    split = int(len(target_arr) * (1 - test_size))
    # Shuffle within splits to avoid location bias
    idx = np.random.RandomState(42).permutation(len(target_arr))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = TFTSequenceDataset(
        static_arr[train_idx],
        temporal_arr[train_idx],
        target_arr[train_idx]
    )
    val_ds = TFTSequenceDataset(
        static_arr[val_idx],
        temporal_arr[val_idx],
        target_arr[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Optionally save scalers ───────────────────────────────
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump({"static": static_scaler, "temporal": temporal_scaler},
                    scaler_path)
        print(f"  💾 Scalers saved → {scaler_path}")

    return (train_loader, val_loader,
            static_scaler, temporal_scaler,
            len(STATIC_COLS), len(TEMPORAL_COLS))


if __name__ == "__main__":
    path = "../data/processed/training_dataset_gujarat_advanced_labeled.csv"
    train_l, val_l, ss, ts, ns, nt = prepare_tft_data(path, seq_len=14, nrows=100_000)
    s, t, y = next(iter(train_l))
    print(f"Batch static  : {s.shape}")
    print(f"Batch temporal: {t.shape}")
    print(f"Batch target  : {y.shape}")
