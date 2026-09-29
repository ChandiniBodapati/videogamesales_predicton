import pandas as pd
import pandera as pa
from pandera import Column, Check

DATA_PATH = "../data/raw/vgsales.csv"


def validate_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    schema = pa.DataFrameSchema({
        "Rank": Column(int),
        "Name": Column(str),
        "Platform": Column(str),
        "Year": Column(float, nullable=True),
        "Genre": Column(str),
        "Publisher": Column(str, nullable=True),
        "NA_Sales": Column(float, Check.ge(0)),
        "EU_Sales": Column(float, Check.ge(0)),
        "JP_Sales": Column(float, Check.ge(0)),
        "Other_Sales": Column(float, Check.ge(0)),
        "Global_Sales": Column(float, Check.ge(0))
    })

    try:
        schema.validate(df, lazy=True)
        print("\nData validation successful!")

    except pa.errors.SchemaErrors as e:
        print("\nData validation failed!")
        print(e.failure_cases)


if __name__ == "__main__":
    validate_data()