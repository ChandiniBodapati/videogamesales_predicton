import pandas as pd
import joblib

from pathlib import Path


# Project root directory
BASE_DIR = Path('C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\src').resolve().parent.parent

# Load final model
model_path = BASE_DIR / "models"/"final_model.pkl"

model = joblib.load("C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\models\\final_model.pkl")

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

predicted_sales = prediction[0]


# Display result
print("\n===== NEW GAME PREDICTION =====")

print("\nNew Game Details:")
print(new_game)

print("\nPredicted Global Sales:")
print(f"{predicted_sales:.2f} million units")