from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("car_price.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [int(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        
        # Predict price
        prediction = model.predict(final_features)
        return render_template('index.html', prediction_text=f'Predicted Car Price: ₹{prediction[0]:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)