import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
data = pd.read_csv("car_prediction_data.csv")

# Select features and target
X = data[['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Owner', 'Seller_Type']]
y = data['Selling_Price']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['Fuel_Type', 'Transmission', 'Seller_Type'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "car_price.pkl")
print("modle Trained successfully")