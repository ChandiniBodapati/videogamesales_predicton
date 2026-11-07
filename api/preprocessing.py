import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw video game data before model feature transformation.
    """
    df = df.copy()

    # Drop identifiers and label columns if present
    drop_candidates = ["Rank", "Name", "Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    for col in drop_candidates:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Impute missing values consistently
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        median_val = df["Year"].median()
        df["Year"] = df["Year"].fillna(median_val if pd.notnull(median_val) else 2006.0)

    if "Publisher" in df.columns:
        df["Publisher"] = df["Publisher"].fillna("Unknown")

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("Other")

    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].fillna("Other")

    # Keep the expected feature columns
    features = ["Platform", "Year", "Genre", "Publisher"]
    available_cols = [c for c in features if c in df.columns]
    return df[available_cols]
