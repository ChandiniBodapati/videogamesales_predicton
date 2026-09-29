import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Video Game Sales Prediction",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 Video Game Sales Prediction")

st.write(
    "Enter the details of a video game to predict its estimated "
    "Global Sales."
)

platform = st.selectbox(
    "Select Platform",
    ["PS4", "PS3", "X360", "XOne", "PC", "PS2", "Wii", "DS", "PSP"]
)

year = st.number_input(
    "Enter Release Year",
    min_value=1980,
    max_value=2026,
    value=2015,
    step=1
)

genre = st.selectbox(
    "Select Genre",
    [
        "Action",
        "Sports",
        "Shooter",
        "Role-Playing",
        "Platform",
        "Racing",
        "Misc",
        "Simulation",
        "Fighting",
        "Adventure",
        "Puzzle",
        "Strategy"
    ]
)

publisher = st.text_input(
    "Enter Publisher",
    "Sony Computer Entertainment"
)

if st.button("🔮 Predict Global Sales"):

    if publisher.strip() == "":
        st.warning("Please enter a publisher name.")

    else:

        new_game = pd.DataFrame({
            "Platform": [platform],
            "Year": [year],
            "Genre": [genre],
            "Publisher": [publisher]
        })

        prediction = model.predict(new_game)

        predicted_sales = prediction[0]

        st.success("Prediction completed successfully!")

        st.subheader("📊 Prediction Result")

        st.write(
            f"**Estimated Global Sales: "
            f"{predicted_sales:.2f} million units**"
        )
        st.write("### 🎮 Game Details")
        st.dataframe(new_game)