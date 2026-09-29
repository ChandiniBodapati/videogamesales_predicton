import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Processed dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "vgsales_cleaned.csv"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load processed dataset
df = pd.read_csv(DATA_PATH)

print("Processed dataset loaded successfully.")
print("Shape:", df.shape)


# Features and target
features = ["Platform", "Year", "Genre", "Publisher"]
target = "Global_Sales"

X = df[features]
y = df[target]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Categorical and numerical features
categorical_features = ["Platform", "Genre", "Publisher"]
numerical_features = ["Year"]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# Machine Learning models
models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting Regression": GradientBoostingRegressor(
        random_state=42
    )
}


# Store results
results = []

best_model = None
best_r2 = -np.inf
best_model_name = ""


# Train and evaluate each model
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("\n" + name)
    print("MSE :", mse)
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # Select best model based on highest R2
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_model_name = name


# Create model comparison dataframe
results_df = pd.DataFrame(results)

# Save model comparison
comparison_path = OUTPUT_DIR / "model_comparison.csv"
results_df.to_csv(comparison_path, index=False)


# Save best model
model_path = MODEL_DIR / "final_model.pkl"
joblib.dump(best_model, model_path)


print("\n--------------------------------")
print("Best Model:", best_model_name)
print("Best R2:", best_r2)
print("--------------------------------")

print("Model comparison saved to:", comparison_path)
print("Final model saved to:", model_path)