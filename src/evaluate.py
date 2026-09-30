import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = BASE_DIR / "data" / "raw" / "vgsales.csv"
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove missing values
df = df.dropna(subset=[
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "Global_Sales"
])

# Features and target
X = df[["Platform", "Year", "Genre", "Publisher"]]
y = df["Global_Sales"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("------------------------")
print("MSE :", mse)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

print("\nPipeline completed successfully!")