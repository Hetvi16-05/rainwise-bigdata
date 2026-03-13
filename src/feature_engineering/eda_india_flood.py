import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================

DATA_PATH = "data/processed/training_dataset_india_enhanced.csv"
df = pd.read_csv(DATA_PATH)

print("=================================")
print("Dataset Shape:", df.shape)
print("=================================")

# ==============================
# 1️⃣ SUMMARY STATISTICS
# ==============================

print("\n===== SUMMARY STATISTICS =====\n")
print(df.describe())

print("\n===== DATA TYPES =====\n")
print(df.dtypes)

print("\n===== MISSING VALUES =====\n")
print(df.isnull().sum())

# ==============================
# 2️⃣ VISUALIZATION SECTION
# ==============================

numeric_cols = df.select_dtypes(include=np.number).columns

# ---- HISTOGRAMS ----
for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ---- BOX PLOTS (Outlier Visual Check) ----
for col in numeric_cols:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    plt.show()

# ---- FLOOD CLASS DISTRIBUTION ----
if "flood" in df.columns:
    plt.figure()
    df["flood"].value_counts().plot(kind="bar")
    plt.title("Flood vs Non-Flood Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    print("\nFlood Class Counts:")
    print(df["flood"].value_counts())

# ==============================
# 3️⃣ CORRELATION ANALYSIS
# ==============================

print("\n===== CORRELATION MATRIX =====\n")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix")
plt.show()

# ==============================
# 4️⃣ OUTLIER DETECTION (IQR)
# ==============================

print("\n===== OUTLIER DETECTION USING IQR =====\n")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# ==============================
# 5️⃣ TIME SERIES VISUALIZATION
# ==============================

if "date" in df.columns and "rainfall" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    daily_avg = df.groupby("date")["rainfall"].mean()

    plt.figure()
    daily_avg.plot()
    plt.title("Average Rainfall Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Rainfall")
    plt.show()

print("\n===== EDA COMPLETE =====")
