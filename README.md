# Video Game Sales Prediction

## Project Overview

This project is an end-to-end Machine Learning project that predicts the
global sales of video games using the Video Game Sales dataset.

The project was initially developed as part of CLA-1 and is extended in
CLA-2 by implementing MLOps concepts such as data validation, experiment
tracking, model registration, and an automated ML pipeline.

## Problem Statement

The objective is to predict the Global Sales of a video game based on
features such as:

- Platform
- Year
- Genre
- Publisher

## Dataset

Dataset: Video Game Sales Dataset

Target Variable:
- `Global_Sales`

Input Features:
- `Platform`
- `Year`
- `Genre`
- `Publisher`

The regional sales columns are not used as input features because they
could cause target leakage.

## Machine Learning Models

The following regression models were implemented:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Gradient Boosting Regressor was selected as the final model after model
comparison and hyperparameter tuning.

## Project Structure

```text
videogamesales_prediction/
│
├── data/
│   ├── raw/
│   │   └── vgsales.csv
│   └── processed/
│
├── notebooks/
│   └── vgsales.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── validate_data.py
│   ├── mlflow_tracking.py
│   ├── model_registry.py
│   └── pipeline.py
│
├── models/
│   └── final_model.pkl
│
├── outputs/
│   ├── model_performance.csv
│   └── new_game_prediction.csv
│
├── tests/
│   └── test_project.py
│
├── reports/
├── configs/
├── logs/
│
├── app.py
├── requirements.txt
└── README.md