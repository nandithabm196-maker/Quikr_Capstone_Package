"""
YUVA INTERN - WEEK 6 INTEGRATIVE CAPSTONE
Quikr Used-Car Price Prediction

Place the raw file named:
    Quikr car price prediction.csv
in the same folder as this script.

The script:
1. Loads and cleans the Quikr dataset.
2. Reproduces the Week 1 cleaning decisions.
3. Performs EDA and saves charts.
4. Builds supervised regression models.
5. Evaluates models using MAE, RMSE and R2.
6. Performs K-Means clustering as an optional unsupervised analysis.
7. Saves the cleaned data, metrics and charts.

IMPORTANT:
Do not invent model metrics in the report. Run this script on the actual dataset
and copy the generated metrics into the Week 6 report.
"""

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "Quikr car price prediction.csv"
OUT = BASE / "week6_outputs"
OUT.mkdir(exist_ok=True)

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce"
    )

def load_and_clean(path):
    df = pd.read_csv(path)

    # Normalize column names while preserving expected meanings.
    df.columns = [c.strip() for c in df.columns]

    # Remove exact duplicate listings.
    df = df.drop_duplicates(keep="first").copy()

    # Numeric conversion / validation.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["Price"] = clean_numeric(df["Price"])
    df["kms_driven"] = clean_numeric(df["kms_driven"])

    # Keep realistic years and rows with required numeric target/features.
    df.loc[(df["year"] < 1980) | (df["year"] > 2025), "year"] = np.nan
    df = df.dropna(subset=["year", "Price", "kms_driven"]).copy()

    df["year"] = df["year"].astype(int)

    # Fuel type: preserve known values and label the remaining missing value.
    df["fuel_type"] = df["fuel_type"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["fuel_type"].isin(["", "nan", "None"]), "fuel_type"] = "Unknown"

    # Project-level correction documented in Week 1.
    # Apply only to the known vehicle/listing condition if present.
    mask = (
        df["name"].astype(str).str.contains("Mahindra XUV500 W6", case=False, na=False)
        & (df["Price"] == 8500003)
    )
    df.loc[mask, "Price"] = 850000

    return df

def add_features(df):
    data = df.copy()
    data["vehicle_age"] = 2020 - data["year"]
    data["km_per_year"] = data["kms_driven"] / data["vehicle_age"].replace(0, 1)
    return data

def make_eda(df):
    # Price distribution
    plt.figure(figsize=(8,5))
    plt.hist(df["Price"], bins=30)
    plt.title("Used-Car Price Distribution")
    plt.xlabel("Price (₹)")
    plt.ylabel("Number of Listings")
    plt.tight_layout()
    plt.savefig(OUT / "01_price_distribution.png", dpi=200)
    plt.close()

    # Price vs mileage
    plt.figure(figsize=(8,5))
    plt.scatter(df["kms_driven"], df["Price"], alpha=0.6)
    plt.title("Price vs Kilometres Driven")
    plt.xlabel("Kilometres Driven")
    plt.ylabel("Price (₹)")
    plt.tight_layout()
    plt.savefig(OUT / "02_price_vs_kms.png", dpi=200)
    plt.close()

    # Average price by fuel
    fuel_price = df.groupby("fuel_type")["Price"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    fuel_price.plot(kind="bar")
    plt.title("Average Price by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Average Price (₹)")
    plt.tight_layout()
    plt.savefig(OUT / "03_average_price_by_fuel.png", dpi=200)
    plt.close()

    # Average price by year
    year_price = df.groupby("year")["Price"].mean()
    plt.figure(figsize=(9,5))
    year_price.plot(marker="o")
    plt.title("Average Price by Vehicle Year")
    plt.xlabel("Vehicle Year")
    plt.ylabel("Average Price (₹)")
    plt.tight_layout()
    plt.savefig(OUT / "04_average_price_by_year.png", dpi=200)
    plt.close()

def train_models(df):
    target = "Price"
    features = ["company", "year", "kms_driven", "fuel_type", "vehicle_age", "km_per_year"]

    X = df[features]
    y = df[target]

    categorical = ["company", "fuel_type"]
    numeric = ["year", "kms_driven", "vehicle_age", "km_per_year"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(OUT / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    plt.figure(figsize=(7,7))
    plt.scatter(y_test, best_pred, alpha=0.65)
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lims, lims)
    plt.title(f"Actual vs Predicted Price - {best_name}")
    plt.xlabel("Actual Price (₹)")
    plt.ylabel("Predicted Price (₹)")
    plt.tight_layout()
    plt.savefig(OUT / "05_actual_vs_predicted.png", dpi=200)
    plt.close()

    return results_df

def run_clustering(df):
    # Optional unsupervised analysis on numerical vehicle characteristics.
    cluster_features = df[["year", "kms_driven", "Price"]].copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(cluster_features)

    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)

    out = df[["name", "company", "year", "kms_driven", "Price", "fuel_type"]].copy()
    out["cluster"] = labels
    out.to_csv(OUT / "kmeans_clusters.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(Z)

    plt.figure(figsize=(8,5))
    plt.scatter(pcs[:,0], pcs[:,1], c=labels, alpha=0.65)
    plt.title("K-Means Vehicle Segmentation")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(OUT / "06_kmeans_clusters.png", dpi=200)
    plt.close()

    return out.groupby("cluster")[["year","kms_driven","Price"]].mean()

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place 'Quikr car price prediction.csv' beside this script."
        )

    raw = pd.read_csv(DATA_FILE)
    cleaned = load_and_clean(DATA_FILE)
    cleaned = add_features(cleaned)

    cleaned.to_csv(OUT / "cleaned_data_week6.csv", index=False)
    make_eda(cleaned)

    metrics = train_models(cleaned)
    cluster_summary = run_clustering(cleaned)

    print("\nRAW DATA SHAPE:", raw.shape)
    print("CLEANED DATA SHAPE:", cleaned.shape)
    print("\nMODEL COMPARISON:")
    print(metrics.to_string(index=False))
    print("\nCLUSTER SUMMARY:")
    print(cluster_summary)

if __name__ == "__main__":
    main()
