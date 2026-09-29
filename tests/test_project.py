import os
def test_dataset_exists():
    assert os.path.exists("C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\data\\raw\\vgsales.csv")
def test_processed_dataset_exists():
    assert os.path.exists("C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\data\\processed\\vgsales_cleaned.csv")
def test_final_model_exists():
    assert os.path.exists("C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\models\\final_model.pkl")
def test_prediction_output_exists():
    assert os.path.exists("C:\\Users\\HP 640 G5\\Desktop\\videogamesales_predicton\\outputs\\new_game_prediction.csv")    