import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


OUT = "plots"
os.makedirs(OUT, exist_ok=True)


FILE = "data/processed/training_dataset_gujarat_labeled.csv"


# =========================
# LOAD SAMPLE (fast)
# =========================

df = pd.read_csv(FILE).sample(
    200000,
    random_state=42
)

print("Shape:", df.shape)


# =========================
# DROP BAD COLUMNS
# =========================

drop_cols = [
    "date",
    "state",
    "lat_x",
    "lon_x",
    "river_distance"
]

df = df.drop(columns=drop_cols, errors="ignore")


# =========================
# keep numeric only
# =========================

df = df.select_dtypes("number")


# =========================
# remove columns with many missing
# =========================

df = df.dropna(axis=1, thresh=len(df) * 0.5)


# =========================
# remove constant columns
# =========================

for c in df.columns:
    if df[c].nunique() <= 1:
        df = df.drop(columns=[c])


print("Columns used:", df.columns)


# =========================
# choose rainfall column
# =========================

rain_col = None

for c in df.columns:
    if "rain7" in c or "rain_7" in c:
        rain_col = c

if rain_col is None:
    for c in df.columns:
        if "rain" in c or "precip" in c:
            rain_col = c

if rain_col is None:
    rain_col = df.columns[0]


print("Rain column:", rain_col)


# =========================
# BAR
# =========================

if "flood" in df.columns:

    df["flood"].value_counts().plot(kind="bar")

    plt.title("Flood vs Non Flood")

    plt.savefig(f"{OUT}/bar_flood.png")
    plt.close()


# =========================
# HISTOGRAM
# =========================

df[rain_col].hist(bins=50)

plt.title("Histogram")

plt.savefig(f"{OUT}/histogram.png")
plt.close()


# =========================
# BOXPLOT
# =========================

sns.boxplot(y=df[rain_col])

plt.title("Boxplot")

plt.savefig(f"{OUT}/boxplot.png")
plt.close()


# =========================
# SCATTER
# =========================

if "flood" in df.columns:

    plt.scatter(
        df[rain_col],
        df["flood"],
        s=5
    )

    plt.title("Scatter")

    plt.savefig(f"{OUT}/scatter.png")
    plt.close()


# =========================
# HEATMAP
# =========================

corr = df.corr()

sns.heatmap(corr, cmap="coolwarm")

plt.title("Heatmap")

plt.savefig(f"{OUT}/heatmap.png")
plt.close()


# =========================
# MODEL PLOTS
# =========================

if "flood" in df.columns:

    if df["flood"].nunique() > 1:

        X = df.drop("flood", axis=1)
        y = df["flood"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1
        )

        model.fit(X_train, y_train)


        # feature importance
        imp = model.feature_importances_

        pd.Series(
            imp,
            index=X.columns
        ).sort_values().plot(kind="barh")

        plt.title("Feature Importance")

        plt.savefig(
            f"{OUT}/feature_importance.png"
        )

        plt.close()


        # confusion
        pred = model.predict(X_test)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            pred
        )

        plt.savefig(
            f"{OUT}/confusion_matrix.png"
        )

        plt.close()


        # roc
        prob = model.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(
            y_test,
            prob
        )

        plt.savefig(
            f"{OUT}/roc_curve.png"
        )

        plt.close()


        # probability
        plt.hist(prob, bins=30)

        plt.title("Probability Histogram")

        plt.savefig(
            f"{OUT}/probability_hist.png"
        )

        plt.close()


print("CLEAN VISUALS CREATED")