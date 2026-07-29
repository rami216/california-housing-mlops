import pandas as pd
from sklearn.datasets import fetch_california_housing

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

data = fetch_california_housing(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\n--- First rows ---")
print(df.head())

print("\n--- Statistics ---")
print(df.describe().T)

print("\n--- Missing values ---")
print(df.isna().sum())

print("\n--- Correlation with target ---")
print(df.corr()["MedHouseVal"].sort_values(ascending=False))

print("\n--- Most frequent target values ---")
print(df["MedHouseVal"].value_counts().head(5))