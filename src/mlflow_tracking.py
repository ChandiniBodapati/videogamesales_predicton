import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


DATA_PATH = "../data/processed/vgsales_cleaned.csv"

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Video_Game_Sales_Prediction")


def train_model(name, model):

    df = pd.read_csv(DATA_PATH)

    X = df[["Platform", "Year", "Genre", "Publisher"]]
    y = df["Global_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    categorical = ["Platform", "Genre", "Publisher"]
    numerical = ["Year"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model", name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

        print("\nModel:", name)
        print("MSE :", mse)
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)


if __name__ == "__main__":

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(alpha=1.0),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    for name, model in models.items():
        train_model(name, model)