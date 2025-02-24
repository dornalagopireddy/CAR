from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("car_price.pkl")

# Define categorical mappings (from dataset analysis)
fuel_map = {0: 'Petrol', 1: 'Diesel', 2: 'CNG'}
transmission_map = {0: 'Manual', 1: 'Automatic'}
seller_map = {0: 'Dealer', 1: 'Individual'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {
            'Year': int(request.form['Year']),
            'Kms_Driven': int(request.form['Kms_Driven']),
            'Fuel_Type': fuel_map[int(request.form['Fuel'])],
            'Transmission': transmission_map[int(request.form['Transmission'])],
            'Owner': int(request.form['Owner']),
            'Seller_Type': seller_map[int(request.form['Seller_Type'])]
        }

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_df, columns=['Fuel_Type', 'Transmission', 'Seller_Type'], drop_first=True)

        # Add missing columns (if any) with 0s
        expected_columns = [
            'Year', 'Kms_Driven', 'Owner',
            'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
            'Transmission_Manual',
            'Seller_Type_Individual'
        ]
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Ensure column order matches training
        input_encoded = input_encoded[expected_columns]

        # Predict
        prediction = model.predict(input_encoded)
        return render_template('index.html', prediction_text=f'Predicted Car Price: ₹{prediction[0]:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)