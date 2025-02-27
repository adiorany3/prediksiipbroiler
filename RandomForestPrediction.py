from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib  # Tambahkan import joblib untuk menyimpan model

# Load the data
file_path = '/Users/macbookpro/Documents/Backup/OneDriveBack/Python Script/RandomForest/data_kandang.csv'
poultry_data = pd.read_csv(file_path)

# Drop rows with missing target, separate target from predictors
poultry_data.dropna(axis=0, subset=['IP'], inplace=True)
y = poultry_data.IP
poultry_data.drop(['IP'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = poultry_data.select_dtypes(include=['int64', 'float64']).columns
X = poultry_data[numeric_cols]

# Split data into training and validation data, for both features and target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model
model.fit(imputed_X_train, y_train)

# Simpan model ke file
model_filename = '/Users/macbookpro/Documents/Backup/OneDriveBack/Python Script/RandomForest/poultry_rf_model.joblib'
joblib.dump(model, model_filename)
print(f"Model berhasil disimpan ke: {model_filename}")

# Get validation predictions
preds = model.predict(imputed_X_valid)

# Calculate mean squared error
mse = mean_squared_error(y_valid, preds)

# Calculate R^2 score
r2 = r2_score(y_valid, preds)

# Output the mean squared error and R^2 score
print(f'Mean Squared Error (MSE): {mse}')
print(f'R^2 Score: {r2}')

# Add interpretation of the measurement results
print("\n--- INTERPRETASI HASIL PENGUKURAN ---")

# MSE Interpretation
y_mean = y_valid.mean()
y_std = y_valid.std()
y_var = y_valid.var()
rmse = np.sqrt(mse)
print(f"\nInterpretasi MSE:")
print(f"- MSE: {mse:.4f}")
print(f"- RMSE: {rmse:.4f} (akar dari MSE, dalam satuan yang sama dengan target)")
print(f"- Rata-rata nilai target: {y_mean:.4f}")
print(f"- Standar deviasi target: {y_std:.4f}")
print(f"- Variansi target: {y_var:.4f}")

if rmse < 0.25 * y_std:
    print("- RMSE sangat rendah dibandingkan dengan standar deviasi target, menunjukkan model memiliki akurasi yang sangat baik.")
elif rmse < 0.5 * y_std:
    print("- RMSE cukup rendah dibandingkan dengan standar deviasi target, menunjukkan model memiliki akurasi yang baik.")
elif rmse < 0.75 * y_std:
    print("- RMSE mendekati standar deviasi target, menunjukkan model memiliki akurasi yang cukup.")
elif rmse < y_std:
    print("- RMSE lebih kecil dari standar deviasi target, menunjukkan model memberikan prediksi yang lebih baik daripada menggunakan nilai rata-rata.")
else:
    print("- RMSE lebih besar dari standar deviasi target, menunjukkan model tidak memberikan prediksi yang lebih baik daripada menggunakan nilai rata-rata.")

# R^2 Interpretation
print(f"\nInterpretasi R^2:")
print(f"- R^2: {r2:.4f}")
if r2 > 0.9:
    print("- R^2 > 0.9: Model sangat baik, menjelaskan lebih dari 90% variabilitas dalam data target.")
elif r2 > 0.7:
    print("- 0.7 < R^2 < 0.9: Model baik, menjelaskan antara 70%-90% variabilitas dalam data target.")
elif r2 > 0.5:
    print("- 0.5 < R^2 < 0.7: Model cukup, menjelaskan antara 50%-70% variabilitas dalam data target.")
elif r2 > 0.3:
    print("- 0.3 < R^2 < 0.5: Model kurang optimal, hanya menjelaskan antara 30%-50% variabilitas dalam data target.")
elif r2 > 0:
    print("- 0 < R^2 < 0.3: Model memiliki kekuatan prediktif yang rendah, menjelaskan kurang dari 30% variabilitas.")
else:
    print("- R^2 ≤ 0: Model tidak lebih baik dari penggunaan nilai rata-rata target sebagai prediksi.")

# Feature importance
print("\nFitur paling penting (5 teratas):")
feature_importances = pd.DataFrame(model.feature_importances_, 
                                  index=imputed_X_train.columns,
                                  columns=['importance']).sort_values('importance', ascending=False)
for i, (index, row) in enumerate(feature_importances.head(5).iterrows()):
    print(f"{i+1}. {index}: {row['importance']:.4f}")

print("\nKesimpulan:")
if r2 > 0.7 and rmse < 0.5 * y_std:
    print("Model memiliki performa yang baik dan dapat digunakan untuk memprediksi IP dengan tingkat kepercayaan yang tinggi.")
elif r2 > 0.5 and rmse < 0.75 * y_std:
    print("Model memiliki performa yang cukup dan dapat digunakan untuk memprediksi IP, namun masih ada ruang untuk perbaikan.")
else:
    print("Model memiliki performa yang kurang memuaskan. Pertimbangkan untuk melakukan perbaikan seperti menambah data, menggunakan fitur tambahan, atau mencoba algoritma lain.")