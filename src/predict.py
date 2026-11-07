import pandas as pd
import joblib
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load final model
model_path = BASE_DIR / "models" / "final_model.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Final model not found at {model_path}")

model = joblib.load(model_path)
print("Final model loaded successfully.")

# New / unseen game data
new_game = pd.DataFrame({
    "Platform": ["PS4"],
    "Year": [2015],
    "Genre": ["Action"],
    "Publisher": ["Sony Computer Entertainment"]
})

# Make prediction
prediction = model.predict(new_game)
predicted_sales = float(prediction[0])

# Save prediction output
output_path = BASE_DIR / "outputs" / "new_game_prediction.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
result_df = new_game.copy()
result_df["Predicted_Global_Sales"] = predicted_sales
result_df.to_csv(output_path, index=False)

# Display result
print("\n===== NEW GAME PREDICTION =====")
print("\nNew Game Details:")
print(new_game)
print("\nPredicted Global Sales:")
print(f"{predicted_sales:.2f} million units")
print(f"Saved prediction to: {output_path}")
