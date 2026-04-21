from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model artifacts
model = joblib.load('product_success_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')
category_values = joblib.load('category_values.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', categories=category_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Create a zero-filled DataFrame with model columns
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Numeric inputs
        input_df['Price'] = float(data.get('Price', 0))
        input_df['Discount'] = float(data.get('Discount', 0))
        
        # Scaling numeric inputs
        input_df[['Price', 'Discount']] = scaler.transform(input_df[['Price', 'Discount']])
        
        # Categorical inputs - One Hot Encoding manually
        cat_col = f"Category_{data.get('Category')}"
        season_col = f"Seasonality_{data.get('Seasonality')}"
        region_col = f"Region_{data.get('Region')}"
        
        if cat_col in input_df.columns: input_df[cat_col] = 1
        if season_col in input_df.columns: input_df[season_col] = 1
        if region_col in input_df.columns: input_df[region_col] = 1
        
        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        result = "SUCCESS" if prediction == 1 else "FAILURE"
        prob_percentage = round(probability * 100, 2)
        
        # Short business message
        if prediction == 1:
            message = "High potential product! Consider stocking a healthy quantity."
        else:
            message = "Risky investment. Consider a smaller trial batch or rethink strategy."
            
        return render_template('index.html', 
                               categories=category_values,
                               prediction=result,
                               probability=prob_percentage,
                               message=message,
                               form_data=data)
                               
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
