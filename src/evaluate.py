import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
data_path = BASE_DIR / "data" / "raw" / "vgsales.csv"
model_path = BASE_DIR / "models" / "best_model.pkl"


# Load dataset
df = pd.read_csv(data_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# Features and target
features = ["Platform", "Year", "Genre", "Publisher"]
target = "Global_Sales"

X = df[features]
y = df[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Load trained best model
model = joblib.load(model_path)

print("Best model loaded successfully.")


# Predictions
predictions = model.predict(X_test)


# Evaluation metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


print("\n===== BEST MODEL EVALUATION =====")
print("MSE  :", mse)
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)