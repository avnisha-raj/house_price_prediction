# House Price Prediction Model 

## Overview

This project presents an end-to-end machine learning solution for predicting house prices. It uses a `RandomForestRegressor` model built with Python and `scikit-learn`. The entire workflow, from data cleaning to real-time prediction, is automated within a modular pipeline, making it robust and easy to use.

---

## Features

- Preprocessing: Utilizes a `scikit-learn` Pipeline to  handle missing data, scale numerical features, and encode categorical features.
- Interactive Interface: A user-friendly command-line interface (CLI) allows for on-demand price predictions.
- Smart Input Handling: If a user skips an input, the program automatically uses a learned default (mean/mode) to prevent errors and ensure a prediction can always be made.


---
## About Dataset

- This dataset provides key features for predicting house prices, including area, bedrooms, bathrooms, stories, amenities like air         conditioning and parking, and information on furnishing status. It enables analysis and modelling to understand the factors impacting house prices and develop accurate predictions in real estate markets.
- Source: [Kagggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- File: Housing.csv


## Project Workflow

The project follows a standard machine learning pipeline:

1.  Data Loading: Loads the dataset from a `.csv` file.
2.  User Input: asks the user to specify the target column (e.g., 'price').
3.  Preprocessing: An automated pipeline performs:
    - Imputation: Fills missing values (mean for numerical, mode for categorical).
    - Scaling: Standardizes numerical features using `StandardScaler`.
    - Encoding: Converts categorical features to numerical format using `OneHotEncoder`.
4.  Model Training: Trains a `RandomForestRegressor` model on the preprocessed data.
5.  Interactive Prediction: Launches a CLI for the user to input house features and receive a price estimate.

---

## Technologies Used

- Python
- pandas: For data manipulation and loading.
- numpy:For numerical operations.
- Scikit-learn: For building the machine learning pipeline and models.

---

##  How to Run the Project
1. Clone this repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt

Run the script:
app.py
Enter your feature value to get real-time predictions
(Type exit to quit the CLI)
