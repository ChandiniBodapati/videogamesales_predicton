from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, Check

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def simulate_corrupted_data(df):
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()

    # 1. Invalid numerical values (negative sales, invalid year)
    corrupted_df.loc[0, "Global_Sales"] = -10.5
    corrupted_df.loc[1, "Year"] = 1850

    # 2. Datatype mismatch (string in Global_Sales)
    corrupted_df.loc[2, "Global_Sales"] = "High Sales"

    # 3. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 999

    # 4. Missing required attribute (dropping a column)
    corrupted_df = corrupted_df.drop(columns=["Genre"])

    return corrupted_df


def run_corruption_simulation():
    data_path = PROJECT_ROOT / "data" / "raw" / "vgsales.csv"
    raw_df = pd.read_csv(data_path)

    corrupted_df = simulate_corrupted_data(raw_df)

    schema = pa.DataFrameSchema({
        "Rank": Column(int),
        "Name": Column(str),
        "Platform": Column(str),
        "Year": Column(float, Check.ge(1980), nullable=True),
        "Genre": Column(str),
        "Publisher": Column(str, nullable=True),
        "Global_Sales": Column(float, Check.ge(0.0)),
    })

    print("\n--- Running Validation on Corrupted Data ---")
    try:
        schema.validate(corrupted_df, lazy=True)
        print("[WARNING] Corrupted data unexpectedly passed validation!")
        return False
    except pa.errors.SchemaErrors as err:
        print("[SUCCESS] Schema Validation FAILED as expected. Corruptions detected.")
        failure_cases = err.failure_cases
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifacts_dir / "corrupted_simulation_errors.csv"
        failure_cases.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return True


if __name__ == "__main__":
    run_corruption_simulation()
