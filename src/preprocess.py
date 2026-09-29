import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
INPUT_PATH = BASE_DIR / "data" / "raw" / "vgsales.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "vgsales_cleaned.csv"


# Load raw dataset
df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)


# Remove duplicate records
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# Handle missing values
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])
df["Genre"] = df["Genre"].fillna(df["Genre"].mode()[0])


# Create processed folder if it does not exist
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)


print("Preprocessing completed successfully!")
print("Processed dataset saved at:", OUTPUT_PATH)
print("Final dataset shape:", df.shape)