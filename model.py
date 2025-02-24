# Import required libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
try:
    data = pd.read_csv("car_prediction_data.csv")
except FileNotFoundError:
    print("Error: Dataset file 'car_data.csv' not found. Please check the file path.")
    exit()

# Data Preprocessing (Selecting required features)
try:
    X = data[['Year','Kms_Driven', 'Fuel_Type', 'Transmission', 'Owner','Seller_Type']]
    y = data['Selling_Price']
except KeyError as e:
    print(f"Error: Missing column {e}. Please check the dataset.")
    exit()

# Convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)

# Ensure all categories are the same
X=pd.get_dummies(X,drop_first=True)
print(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)


# Ensure all categories are the same
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0  # Add missing columns with default value 0

X_test = X_test[X_train.columns]  # Reorder columns
# Train the Linear Regression model
model = LinearRegression()
print(model.fit(X_train, y_train))


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse:.2f}")

# Save the trained model
joblib.dump(model, "car_price.pkl")
print("Model saved as 'car_price_model.pkl'.")