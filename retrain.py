import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('product_success_dataset-2.csv')

# Preprocessing
features = ['Category', 'Seasonality', 'Region', 'Price', 'Discount']
X = df[features]
y = df['Success']

# One-hot encoding
X = pd.get_dummies(X, columns=['Category', 'Seasonality', 'Region'], drop_first=False)
model_columns = list(X.columns)

# Scaling
scaler = StandardScaler()
X[['Price', 'Discount']] = scaler.fit_transform(X[['Price', 'Discount']])

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X, y)

# Save artifacts
joblib.dump(model, 'product_success_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

category_values = {
    'Category': list(df['Category'].unique()),
    'Seasonality': list(df['Seasonality'].unique()),
    'Region': list(df['Region'].unique())
}
joblib.dump(category_values, 'category_values.pkl')

print("✅ Model retrained and artifacts saved for scikit-learn 1.8.0")
