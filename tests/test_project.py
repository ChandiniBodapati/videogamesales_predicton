import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def test_dataset_exists():
    assert (BASE_DIR / "data" / "raw" / "vgsales.csv").exists()

def test_processed_dataset_exists():
    assert (BASE_DIR / "data" / "processed" / "vgsales_cleaned.csv").exists()

def test_final_model_exists():
    assert (BASE_DIR / "models" / "final_model.pkl").exists()

def test_prediction_output_exists():
    assert (BASE_DIR / "outputs" / "new_game_prediction.csv").exists()
