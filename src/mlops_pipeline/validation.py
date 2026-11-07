import pandas as pd
import pandera as pa
from pandera import Column, Check
from pathlib import Path


def get_vgsales_schema():
    return pa.DataFrameSchema(
        columns={
            "Rank": Column(int, Check.ge(1), nullable=True),
            "Name": Column(str, nullable=True),
            "Platform": Column(str, nullable=False),
            "Year": Column(float, Check.ge(1980), nullable=True),
            "Genre": Column(str, nullable=False),
            "Publisher": Column(str, nullable=True),
            "NA_Sales": Column(float, Check.ge(0.0), nullable=True),
            "EU_Sales": Column(float, Check.ge(0.0), nullable=True),
            "JP_Sales": Column(float, Check.ge(0.0), nullable=True),
            "Other_Sales": Column(float, Check.ge(0.0), nullable=True),
            "Global_Sales": Column(float, Check.ge(0.0), nullable=False),
        },
        coerce=True,
    )


def validate_schema(df: pd.DataFrame, output_report_name: str = "validation_report.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_vgsales_schema()
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failure_cases = err.failure_cases
        print(failure_cases)
        report_dir = Path("artifacts")
        report_dir.mkdir(parents=True, exist_ok=True)
        failure_cases.to_csv(report_dir / output_report_name, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_dir / output_report_name}'.")
        return False
