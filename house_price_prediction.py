
# ===============================================
# 1. Install Required Packages (Colab safe)
# ===============================================
!pip install scikit-learn pandas numpy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from google.colab import files
import io

# ===============================================
# 2. Upload Dataset Manually
# ===============================================
print("Please upload your dataset (CSV format)...")
uploaded = files.upload()

# Load dataset
file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]))
print("\nDataset loaded successfully!")
print(df.head())

# ===============================================
# 3. Basic Data Setup
# ===============================================
target_column = input("\nEnter the name of the target column (price column): ").strip()

X = df.drop(columns=[target_column])
y = df[target_column]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ===============================================
# 4. Preprocessing Pipelines
# ===============================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===============================================
# 5. Train Model
# ===============================================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel trained! MAE:", mean_absolute_error(y_test, y_pred))

# Store default values for missing inputs
default_values = {}
for col in numeric_features:
    default_values[col] = X_train[col].mean()
for col in categorical_features:
    default_values[col] = X_train[col].mode()[0]

# ===============================================
# 6. User Prediction Interface
# ===============================================
print("\n=== Price Prediction Interface ===")
while True:
    user_data = {}
    for col in X.columns:
        value = input(f"Enter value for '{col}' (Press Enter to skip): ").strip()
        if value == "":
            user_data[col] = default_values[col]
        else:
            if col in numeric_features:
                user_data[col] = float(value)
            else:
                user_data[col] = value

    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)[0]
    print(f"\nPredicted Price: {prediction:.2f}")

    cont = input("\nDo you want to predict again? (yes/no): ").strip().lower()
    if cont != "yes":
        print("Exiting prediction interface.")
        break
