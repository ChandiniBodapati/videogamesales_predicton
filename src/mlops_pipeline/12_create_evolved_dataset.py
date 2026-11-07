import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data/processed/X_test_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/monitoring/drift/evolved_data.csv"

RANDOM_SEED = 42


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    if not SOURCE_PATH.exists():
        # Fallback to creating from raw if X_test_raw doesn't exist yet
        raw_df = pd.read_csv(PROJECT_ROOT / "data/raw/vgsales.csv")
        sample_df = raw_df[["Platform", "Year", "Genre", "Publisher"]].sample(n=3000, random_state=RANDOM_SEED)
    else:
        sample_df = pd.read_csv(SOURCE_PATH).copy()

    df = sample_df.copy()

    # 1. Numerical drift: Year (shift towards newer game releases)
    if "Year" in df.columns:
        df["Year"] = (
            pd.to_numeric(df["Year"], errors="coerce").fillna(2006) + rng.normal(5, 2, len(df))
        ).clip(lower=1980, upper=2024).round()

    # 2. Categorical drift: Platform (shift towards modern platforms: PS4, XOne, PC)
    if "Platform" in df.columns:
        modern_platforms = ["PS4", "XOne", "PC", "PS3", "X360", "WiiU"]
        p_dist = [0.40, 0.30, 0.15, 0.05, 0.05, 0.05]
        df["Platform"] = rng.choice(modern_platforms, size=len(df), p=p_dist)

    # 3. Categorical drift: Genre (shift towards Action and Shooter)
    if "Genre" in df.columns:
        modern_genres = ["Action", "Shooter", "Role-Playing", "Sports", "Adventure"]
        p_genres = [0.45, 0.30, 0.15, 0.05, 0.05]
        df["Genre"] = rng.choice(modern_genres, size=len(df), p=p_genres)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("EVOLVED DATASET CREATED")
    print("=" * 70)
    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Rows   : {len(df)}")
    print(f"Columns: {list(df.columns)}")
    return True


if __name__ == "__main__":
    create_evolved_dataset()
