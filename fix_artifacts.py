import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# This script fixes the "Feature names mismatch" by strictly scaling ONLY numeric columns
# and ensuring artifacts are consistent with app.py

print("🔄 Repairing model artifacts...")

# Load raw data
try:
    df = pd.read_csv('product_success_dataset-2.csv')
except:
    print("❌ Error: product_success_dataset-2.csv not found.")
    exit()

# Basic cleaning
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(df['Price'].median())
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce').fillna(df['Discount'].median())

# Preprocessing
features = ['Category', 'Seasonality', 'Region', 'Price', 'Discount']
X = df[features]
y = df['Success']

# One-hot encoding
X = pd.get_dummies(X, columns=['Category', 'Seasonality', 'Region'], drop_first=False)
model_columns = list(X.columns)

# Split first to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SCALE ONLY NUMERIC COLUMNS (Robust Way)
num_cols = ['Price', 'Discount']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Save consistent artifacts
joblib.dump(model, 'product_success_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

category_values = {
    'Category': list(df['Category'].unique()),
    'Seasonality': list(df['Seasonality'].unique()),
    'Region': list(df['Region'].unique())
}
joblib.dump(category_values, 'category_values.pkl')

print("✅ Repair Complete! Scaler now expects 2 features (Price, Discount).")
print("✅ Model now expects total columns:", len(model_columns))
