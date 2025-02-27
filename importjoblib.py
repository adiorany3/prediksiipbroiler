import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Function to interpret IP values
def interpret_ip(ip_value):
    if ip_value < 300:
        return "Kurang"
    elif 301 <= ip_value <= 325:
        return "Cukup"
    elif 326 <= ip_value <= 350:
        return "Baik"
    elif 351 <= ip_value <= 400:
        return "Sangat Baik"
    elif 401 <= ip_value <= 500:
        return "Istimewa"
    else:
        return "Perlu pengecekan data"

# Try loading existing data or create sample data if not available
try:
    # Replace 'data_kandang.csv' with your actual data file path if available
    data = pd.read_csv('data_kandang.csv')
    print("Data loaded successfully")
    
    # Define features and target
    X = data[['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']]
    y = data['IP']
    
except FileNotFoundError:
    print("Creating sample training data...")
    # Create sample data (you should replace this with your actual data)
    n_samples = 100
    X_train = np.random.rand(n_samples, 6)  # 6 features now
    # Column names for reference: Age, Total_Body_Weight, FCR, Live_Bird, Ayam_Dipelihara, persen_Live_Bird
    y_train = np.random.rand(n_samples) * 400  # IP values typically between 0-500
    
    # Convert to DataFrame for better handling
    X = pd.DataFrame(X_train, columns=['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird'])
    y = pd.Series(y_train)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance - MSE: {mse:.2f}, R²: {r2:.2f}")

# Save the model
joblib.dump(rf_model, 'poultry_rf_model.joblib')
print("Model saved successfully")

# Load the model
model = joblib.load('poultry_rf_model.joblib')

# Example: Make a prediction with sample data
# For real use, replace these values with actual data
age = 35
total_body_weight = 2100  # kg
fcr = 1.5
ayam_dipelihara = 10000  # birds
persen_live_bird = 95.0  # %
live_bird = (persen_live_bird/100) * ayam_dipelihara

# Create input data for prediction
input_data = [[age, total_body_weight, fcr, live_bird, ayam_dipelihara, persen_live_bird]]

# Perform prediction
prediction = model.predict(input_data)[0]

# Calculate actual IP using the formula from reference code
actual_ip = ((persen_live_bird * (total_body_weight/live_bird)) / (fcr * age)) * 100

# Output predictions with interpretation
print(f"Predicted IP broiler: {prediction:.2f} ({interpret_ip(prediction)})")
print(f"Calculated IP broiler: {actual_ip:.2f} ({interpret_ip(actual_ip)})")

# Log the prediction to a CSV file
today = datetime.date.today()
culling = ayam_dipelihara * (100 - persen_live_bird) / 100
adg_actual = (total_body_weight / live_bird) * 1000 if live_bird > 0 else 0
feed = fcr * total_body_weight

# Create a dataframe with the new data
new_data = pd.DataFrame([[
    age, today, total_body_weight, live_bird, ayam_dipelihara, persen_live_bird, 
    actual_ip, prediction, culling, adg_actual, feed, fcr
]], columns=[
    'Age', 'Date', 'Total_Body_Weight', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird',
    'IP_actual', 'IP', 'Culling', 'ADG_actual', 'Feed', 'FCR_actual'
])

# Save or append to prediction log
try:
    existing_data = pd.read_csv('prediction_log.csv')
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv('prediction_log.csv', index=False)
    print("Prediction added to log")
except FileNotFoundError:
    new_data.to_csv('prediction_log.csv', index=False)
    print("Prediction log created")