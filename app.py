import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import requests

# Page configuration
st.set_page_config(page_title="Hitung IP Broiler dengan mudah", page_icon="🐔")

# Title and header
col1, col2 = st.columns([1, 2])

with col1:
    try:
        image = Image.open("header.jpeg")
        st.image(image, use_container_width=True, width=120)
    except FileNotFoundError:
        st.warning("header.jpeg tidak ditemukan.")

with col2:
    st.title("Tool Menghitung Indeks Performans (IP) Broiler")

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

# Add this function after the interpret_ip function (around line 30)
def generate_ip_recommendations(ip_value, fcr, mortality_rate, age):
    """Generate personalized recommendations based on IP and farm parameters"""
    # Base recommendations dictionary organized by IP interpretation category
    recommendations = {
        "Kurang": [
            "Tingkatkan manajemen ventilasi dan kualitas udara dalam kandang",
            "Evaluasi kembali program pemberian pakan dan nutrisi",
            "Periksa kepadatan kandang, mungkin terlalu tinggi",
            "Evaluasi program pencahayaan dalam kandang",
            "Pastikan air minum cukup bersih dan selalu tersedia",
            "[Cek Penyakit](https://deteksiayam.streamlit.app/) - Periksa kemungkinan masalah kesehatan pada ayam Anda"
        ],
        "Cukup": [
            "Fokus pada perbaikan FCR dengan manajemen pakan yang lebih baik",
            "Optimalkan program vaksinasi dan biosecurity",
            "Pertimbangkan kontrol suhu kandang yang lebih baik",
            "Periksa kembali kualitas DOC (Day Old Chick)",
            "Evaluasi jadwal pemberian pakan",
            "[Cek Penyakit](https://deteksiayam.streamlit.app/) - Periksa apakah ada gejala penyakit yang mempengaruhi performa"
        ],
        "Baik": [
            "Pertahankan praktek manajemen yang sudah baik",
            "Optimalkan formula pakan untuk efisiensi lebih baik",
            "Pertimbangkan perbaikan kecil pada kualitas air",
            "Monitor pertumbuhan harian lebih ketat",
            "Evaluasi potensi peningkatan kepadatan kandang"
        ],
        "Sangat Baik": [
            "Dokumentasikan praktek manajemen untuk replikasi di periode berikutnya",
            "Fokus pada detail kecil untuk perbaikan minor",
            "Pertimbangkan ujicoba suplemen nutrisi untuk hasil optimal",
            "Review program kesehatan untuk menekan angka kematian",
            "Evaluasi potensi perpanjangan masa panen untuk bobot optimal"
        ],
        "Istimewa": [
            "Dokumentasikan semua praktek untuk standardisasi",
            "Bagikan praktek terbaik dengan farm lain",
            "Pertimbangkan pengembangan kapasitas produksi",
            "Evaluasi aspek ekonomi untuk optimasi profit",
            "Pertahankan sistem manajemen yang sudah sangat baik"
        ]
    }
    
    # Get interpretation of IP
    interpretation = interpret_ip(ip_value)
    
    # Select base recommendations
    base_recs = recommendations.get(interpretation, ["Perlu evaluasi menyeluruh pada manajemen farm"])
    
    # Add specific recommendations based on metrics
    specific_recs = []
    
    # FCR-specific recommendations
    if fcr > 2.0:
        specific_recs.append("FCR tinggi (>2.0): Evaluasi kualitas pakan dan program pemberian pakan")
    elif fcr > 1.7:
        specific_recs.append("FCR cukup tinggi: Pertimbangkan penyesuaian formulasi pakan")
    elif fcr < 1.4:
        specific_recs.append("FCR sangat baik: Pertahankan manajemen pakan yang sudah optimal")
        
    # Mortality-specific recommendations (100 - persen_live_bird = mortality rate)
    mortality_rate = 100 - mortality_rate  # Convert live bird % to mortality %
    if mortality_rate > 5:
        specific_recs.append(f"Angka kematian tinggi ({mortality_rate:.1f}%): Evaluasi program kesehatan dan biosecurity")
    elif mortality_rate > 3:
        specific_recs.append(f"Angka kematian perlu perhatian ({mortality_rate:.1f}%): Periksa kondisi kandang dan manajemen")
    else:
        specific_recs.append(f"Angka kematian rendah ({mortality_rate:.1f}%): Pertahankan program kesehatan yang baik")
        
    # Age-specific recommendations
    if ip_value < 350 and age > 35:
        specific_recs.append("Umur panen yang lebih panjang tidak menghasilkan IP optimal: Evaluasi waktu panen")
        
    # Combine and return recommendations
    all_recommendations = base_recs + specific_recs
    return all_recommendations

# Cache data loading for better performance
@st.cache_data
def load_data(url, verbose=False):
    # Define standard column structure we want to maintain
    standard_columns = ['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 
                       'Ayam_Dipelihara', 'persen_Live_Bird', 'IP']
    
    # Lists to hold our data sources
    data_sources = []
    
    # Try loading main dataset
    try:
        main_data = pd.read_csv(url, thousands=',')
        if verbose:
            st.success(f"Data {url} berhasil dimuat: {len(main_data)} baris")
        
        # Verify required columns exist
        missing_cols = [col for col in standard_columns if col not in main_data.columns]
        if missing_cols and verbose:
            st.warning(f"Kolom yang tidak ditemukan di {url}: {', '.join(missing_cols)}")
        else:
            data_sources.append(('Main data', main_data))
    except FileNotFoundError:
        if verbose:
            st.warning(f"File {url} tidak ditemukan.")
    
    # Try loading prediction history
    try:
        pred_data = pd.read_csv('prediksi.csv')
        if verbose:
            st.success(f"Data prediksi sebelumnya berhasil dimuat: {len(pred_data)} baris")
        
        # Map columns from prediction data to standard names
        column_mapping = {
            'IP_actual': 'IP',  # Use actual IP for training
            'FCR_actual': 'FCR'
        }
        
        # Apply mapping
        for old_col, new_col in column_mapping.items():
            if old_col in pred_data.columns and new_col in standard_columns:
                pred_data[new_col] = pred_data[old_col]
        
        # Check if we have required columns
        has_required = all(col in pred_data.columns for col in standard_columns)
        if not has_required and verbose:
            missing = [col for col in standard_columns if col not in pred_data.columns]
            st.warning(f"Kolom yang tidak ditemukan di prediksi.csv: {', '.join(missing)}")
        else:
            data_sources.append(('Prediction data', pred_data))
    except FileNotFoundError:
        if verbose:
            st.info("Tidak ada data prediksi sebelumnya.")
    
    # Combine the datasets
    if data_sources:
        # Create a new combined dataframe with standardized structure
        combined_data = pd.DataFrame(columns=standard_columns)
        
        for source_name, source_df in data_sources:
            # Extract only the columns we need, ignore others
            source_subset = pd.DataFrame()
            for col in standard_columns:
                if col in source_df.columns:
                    source_subset[col] = source_df[col]
                else:
                    source_subset[col] = np.nan
            
            # Filter out rows with critical NaN values
            valid_rows = ~source_subset[['Age', 'FCR', 'IP']].isna().any(axis=1)
            valid_subset = source_subset[valid_rows]
            
            # Append to the combined data
            if not valid_subset.empty:
                if verbose:
                    st.info(f"Menambahkan {len(valid_subset)} baris valid dari {source_name}")
                combined_data = pd.concat([combined_data, valid_subset], ignore_index=True)
            else:
                if verbose:
                    st.warning(f"Tidak ada data valid dari {source_name}")
    
        if len(combined_data) >= 10:
            if verbose:
                st.success(f"Total data untuk pelatihan: {len(combined_data)} baris")
            return combined_data
        else:
            if verbose:
                st.warning("Data gabungan terlalu sedikit. Menambahkan data sampel.")
    else:
        if verbose:
            st.warning("Tidak ada sumber data valid. Menggunakan data sampel.")
    
    # If we reach here, we need sample data
    sample_data = generate_sample_data()
    return sample_data

def generate_sample_data():
    n_samples = 100
    X_train = np.random.rand(n_samples, 6)
    X_train[:, 0] *= 45  # Age (1-45 days)
    X_train[:, 1] *= 23000  # Total_Body_Weight (0-23000 kg)
    X_train[:, 2] = 1 + X_train[:, 2]  # FCR (1.0-2.0)
    X_train[:, 3] *= 30000  # Live_Bird (0-10000 birds)
    X_train[:, 4] *= 30000  # Ayam_Dipelihara (0-30000 birds)
    X_train[:, 5] *= 100  # persen_Live_Bird (0-100%)
    
    # Calculate IP values
    y_train = np.zeros(n_samples)
    for i in range(n_samples):
        if X_train[i, 0] > 0 and X_train[i, 2] > 0:
            y_train[i] = ((X_train[i, 5] * (X_train[i, 1]/X_train[i, 3])) / (X_train[i, 2] * X_train[i, 0])) * 100
    
    df = pd.DataFrame(
        X_train,
        columns=['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']
    )
    df['IP'] = y_train
    
    return df

# Load data
DATA_URL = 'data_kandang.csv'
data = load_data(DATA_URL, verbose=False)

# Model training function
@st.cache_data(show_spinner=False)
def train_model(data):
    required_features = ['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']
    target = 'IP'
    
    # Check if all required columns exist
    missing_cols = [col for col in required_features + [target] if col not in data.columns]
    if missing_cols:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_cols)}")
        return None, 0, 0
    
    # Select features and target, handle missing values
    X = data[required_features].copy()
    y = data[target].copy()
    
    # Drop rows with NaN values
    valid_rows = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_rows]
    y = y[valid_rows]
    
    if len(X) < 10:
        st.warning("Data terlalu sedikit untuk pelatihan model yang baik. Menambahkan data sampel.")
        sample_data = generate_sample_data()
        X = pd.concat([X, sample_data[required_features]], ignore_index=True)
        y = pd.concat([y, sample_data[target]], ignore_index=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, 'poultry_rf_model.joblib')
    
    return model, mse, r2

# Better approach to use environment variables
import os

bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

# Function to toggle Telegram bot status
def toggle_telegram_bot(enable=None):
    """Toggle or set the Telegram bot status"""
    status_file = 'telegram_bot_status.txt'
    
    # If enable is None, we're toggling the current status
    if enable is None:
        # Read current status
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                current_status = f.read().strip() == 'enabled'
            # Toggle the status
            new_status = not current_status
        else:
            # Default to disabled if no status file exists
            new_status = False
    else:
        # Set status directly to the provided value
        new_status = enable
    
    # Save the new status
    with open(status_file, 'w') as f:
        f.write('enabled' if new_status else 'disabled')
    
    return new_status

# Get current Telegram bot status
def get_telegram_bot_status():
    """Get the current Telegram bot status (enabled/disabled)"""
    status_file = 'telegram_bot_status.txt'
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return f.read().strip() == 'enabled'
    else:
        # Default to disabled if no status file exists
        return False

# Modify the send_to_telegram function to support sending multiple files
def send_to_telegram(message, files=None):
    """Send notification and optionally files to Telegram bot"""
    # Check if the Telegram bot is enabled
    if not get_telegram_bot_status():
        # Bot is disabled, do nothing
        return None
    
    try:
        # Use the global variables instead of hardcoded values
        api_url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
        params = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(api_url, params=params)
        
        # If we have files to send, upload them as documents
        if files:
            print(f"Attempting to send message to Telegram with {len(files) if files else 0} files")
            for file_name, caption in files:
                if os.path.exists(file_name):
                    api_url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
                    
                    # Prepare the file for upload
                    with open(file_name, 'rb') as file:
                        files = {'document': file}
                        data = {'chat_id': chat_id, 'caption': caption}
                        
                        # Upload the file
                        file_response = requests.post(api_url, data=data, files=files)
        
        return response.json()
    except Exception as e:
        # Log error but don't crash the app
        print(f"Error sending Telegram notification: {str(e)}")
        return None

# Replace the model loading section with this auto-retraining implementation

# Load data first
DATA_URL = 'data_kandang.csv'
data = load_data(DATA_URL)

# Check if we need to retrain the model
retrain_needed = False
model_path = 'poultry_rf_model.joblib'
last_modified_time = None
prediksi_modified_time = None

try:
    # Check if model exists and when it was last modified
    if os.path.exists(model_path):
        last_modified_time = os.path.getmtime(model_path)
        model_modified_date = datetime.datetime.fromtimestamp(last_modified_time).date()
        
        # Check if prediction data has been modified since last model training
        if os.path.exists('prediksi.csv'):
            prediksi_modified_time = os.path.getmtime('prediksi.csv')
            prediksi_modified_date = datetime.datetime.fromtimestamp(prediksi_modified_time).date()
            
            # Retrain if prediction data is newer than model
            if prediksi_modified_time > last_modified_time:
                st.info(f"Data prediksi telah diperbarui sejak model terakhir dilatih. Model akan dilatih ulang.")
                retrain_needed = True
    else:
        # Model doesn't exist, needs training
        retrain_needed = True
        
    # Try loading the model if no retraining is needed
    if not retrain_needed:
        model = joblib.load(model_path)
        expected_features = len(['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird'])
        
        # Check feature compatibility
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != expected_features:
            st.warning(f"Model membutuhkan {model.n_features_in_} fitur, tetapi kita memerlukan {expected_features} fitur. Model akan dilatih ulang.")
            retrain_needed = True
        else:
            st.success(f"Model terbaru berhasil dimuat pada tanggal {model_modified_date.strftime('%d %B %Y')}.")
            
except (FileNotFoundError, ValueError) as e:
    st.warning(f"Tidak dapat memuat model: {str(e)}. Model akan dilatih ulang.")
    retrain_needed = True

# After model training, store R² in session state for later use
if retrain_needed:
    with st.spinner("Melatih model dengan data terbaru..."):
        model, mse, r2 = train_model(data)
        st.success("Model baru berhasil dilatih!")
        st.info(f"Performa model - MSE: {mse:.2f}, R²: {r2:.2f}")
        
        # Store R² in session state
        if 'model_r2' not in st.session_state:
            st.session_state.model_r2 = r2
        else:
            st.session_state.model_r2 = r2
            
        # Add this code to send notification and files if R² is high enough
        if r2 >= 0.98:
            host_ip = requests.get('https://api.ipify.org?format=json').json()['ip']
            message = f"""<b>🎉 Model Unggul Terdeteksi!</b>
            
Model dengan performa tinggi telah dihasilkan:
- R² Score: {r2:.4f}
- MSE: {mse:.4f}
- Server: {host_ip}
- Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model dan file prediksi telah disimpan dan dikirimkan."""
            
            # Send both message and files
            files_to_send = [
                ('poultry_rf_model.joblib', f"Model file (R² score: {r2:.4f})"),
                ('prediksi.csv', f"Prediction data file (R² score: {r2:.4f})")
            ]
            send_to_telegram(message, files=files_to_send)
else:
    # If we didn't retrain, use a stored R² or default value
    if 'model_r2' not in st.session_state:
        # Load the model and evaluate it if possible, otherwise use a default
        try:
            # Quick evaluation on sample data
            sample = generate_sample_data()
            X = sample[['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']]
            y = sample['IP']
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            st.session_state.model_r2 = r2
        except:
            # Default to a very low score to be safe
            st.session_state.model_r2 = 0.0

# Custom CSS for green button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #00FF00 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# Input form
st.sidebar.header("Masukkan Parameter Produksi Broiler")

# Add the retrain button to the sidebar
if st.sidebar.button("Cek Model dengan Data Terbaru"):
    with st.spinner("Sedang melatih model dengan data terbaru..."):
        # Force reload of data (bypass cache)
        st.cache_data.clear()
        fresh_data = load_data(DATA_URL, verbose=False)
        model, mse, r2 = train_model(fresh_data)
        st.success("Model berhasil diperbarui dengan data terbaru! Terimakasih atas kontribusi Anda.")
        st.info(f"Performa model baru - MSE: {mse:.2f}, R²: {r2:.2f}. Data terbaru telah dimuat.")
        
        # Add this code to send notification and model file if R² is high enough
        if r2 >= 0.90:
            host_ip = requests.get('https://api.ipify.org?format=json').json()['ip']
            message = f"""<b>🎉 Model Unggul Terdeteksi!</b>
            
Model dengan performa tinggi telah dihasilkan:
- R² Score: {r2:.4f}
- MSE: {mse:.4f}
- Server: {host_ip}
- Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model ini telah disimpan dan siap digunakan. File model dilampirkan."""
            # Send both message and model file
            send_to_telegram(message, files=[('poultry_rf_model.joblib', f"Model file (R² score: {r2:.4f})")])
            
            # Second location - when "Cek Model dengan Data Terbaru" button is pressed
            if r2 >= 0.90:
                host_ip = requests.get('https://api.ipify.org?format=json').json()['ip']
                message = f"""<b>🎉 Model Unggul Terdeteksi!</b>
                
Model dengan performa tinggi telah dihasilkan:
- R² Score: {r2:.4f}
- MSE: {mse:.4f}
- Server: {host_ip}
- Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model dan file prediksi telah disimpan dan dikirimkan."""
                
                # Send both message and files
                files_to_send = [
                    ('poultry_rf_model.joblib', f"Model file (R² score: {r2:.4f})"),
                    ('prediksi.csv', f"Prediction data file (R² score: {r2:.4f})")
                ]
                send_to_telegram(message, files=files_to_send)

age = st.sidebar.number_input("Umur Ayam (Hari)", min_value=1, max_value=45, 
                             help="Masukkan umur ayam antara 1-45 hari (periode pemeliharaan broiler standar)")
fcr = st.sidebar.number_input("FCR", min_value=0.0, max_value=3.0, 
                             help="Feed Conversion Ratio: Rasio pakan terhadap pertambahan bobot. Nilai normal antara 1.0-3.0, semakin rendah semakin baik")
ayam_dipelihara = st.sidebar.number_input("Jumlah Ayam Dipelihara (ekor)", min_value=0,
                                         help="Jumlah total ayam yang dipelihara sejak awal")
persen_live_bird = st.sidebar.number_input("Persentase Ayam Hidup (%)", min_value=50, max_value=100,
                                          help="Persentase ayam yang bertahan hidup hingga panen (50-100%)")
total_body_weight = st.sidebar.number_input("Total Berat Badan Panen (kg)", min_value=0,
                                           help="Berat total seluruh ayam yang dipanen dalam kilogram")

# Predict button - MOVED ABOVE Status Synchronisasi
if st.sidebar.button("Hitung Indeks Performans"):
    # Calculate Live_Bird
    live_bird = (persen_live_bird / 100) * ayam_dipelihara
    
    # Validate inputs
    if age == 0 or fcr == 0 or live_bird == 0:
        st.warning("Mohon isi semua parameter dengan nilai yang valid (tidak boleh nol).")
    else:
        # Create input data for prediction
        input_data = pd.DataFrame([[age, total_body_weight, fcr, live_bird, ayam_dipelihara, persen_live_bird]],
                                  columns=['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate actual IP
        actual_ip = ((persen_live_bird * (total_body_weight/live_bird)) / (fcr * age)) * 100
        
        # Display predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"IP Prediksi: {prediction:.2f}")
            interpretasi_prediksi = interpret_ip(prediction)
            st.success(f"Interpretasi IP Prediksi: {interpretasi_prediksi}")
            
        with col2:
            st.success(f"IP Aktual: {actual_ip:.2f}")
            interpretasi_aktual = interpret_ip(actual_ip)
            st.success(f"Interpretasi IP Aktual: {interpretasi_aktual}")
        
        # Compare interpretations
        if interpretasi_aktual != interpretasi_prediksi:
            st.info("Perbedaan interpretasi antara IP Aktual dan IP Prediksi: System memerlukan lebih banyak data untuk meningkatkan akurasi dugaan. Silahkan tambahkan data yang lain, agar Machine Learning dapat terus belajar")
        
        # Additional calculated values
        today = datetime.date.today()
        culling = ayam_dipelihara * (100 - persen_live_bird) / 100
        adg_actual = (total_body_weight / live_bird) * 1000 if live_bird > 0 else 0
        feed = fcr * total_body_weight
        
        # Create record for logging
        new_data = pd.DataFrame([[
            age, today, total_body_weight, live_bird, ayam_dipelihara, persen_live_bird, 
            actual_ip, prediction, culling, adg_actual, feed, fcr
        ]], columns=[
            'Age', 'Date', 'Total_Body_Weight', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird',
            'IP_actual', 'IP', 'Culling', 'ADG_actual', 'Feed', 'FCR_actual'
        ])
        
        # Save data to CSV - only if model quality is good enough
        try:
            # Check if model R² is high enough to add this data point
            model_quality_sufficient = st.session_state.get('model_r2', 0.0) >= 0.90
            
            if model_quality_sufficient:
                existing_data = pd.read_csv('prediksi.csv')
                
                # Rename columns in existing data if needed
                if 'actual_ip' in existing_data.columns:
                    existing_data.rename(columns={'actual_ip': 'IP_actual'}, inplace=True)
                if 'prediction' in existing_data.columns:
                    existing_data.rename(columns={'prediction': 'IP'}, inplace=True)
                if 'FCR' in existing_data.columns and 'FCR_actual' not in existing_data.columns:
                    existing_data.rename(columns={'FCR': 'FCR_actual'}, inplace=True)
                
                # Ensure new_data has the same columns as existing_data
                for col in existing_data.columns:
                    if col not in new_data.columns:
                        new_data[col] = np.nan
                
                new_data = new_data[existing_data.columns]
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Remove duplicates based on key operational fields
                # We consider entries duplicate if all these key parameters match
                duplicate_check_columns = ['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 
                                          'Ayam_Dipelihara', 'persen_Live_Bird', 'IP_actual']
                
                # Keep only the first occurrence of each unique combination
                combined_data = combined_data.drop_duplicates(
                    subset=duplicate_check_columns, 
                    keep='first'
                )
                
                # Save deduplicated data
                combined_data.to_csv('prediksi.csv', index=False)
                st.success("Data akan dipertimbangkan menjadi update (duplikasi dihapus)")
            else:
                st.warning(f"Data tidak ditambahkan ke database karena kualitas model saat ini (R² = {st.session_state.get('model_r2', 0.0):.2f}) kurang dari 0.90")
                
        except FileNotFoundError:
            # For first-time use, create file if model is good enough
            if st.session_state.get('model_r2', 0.0) >= 0.90:
                new_data.to_csv('prediksi.csv', index=False)
                st.success("Database berhasil diperbarui")
            else:
                st.warning(f"Data tidak ditambahkan ke database karena kualitas model saat ini (R² = {st.session_state.get('model_r2', 0.0):.2f}) kurang dari 0.90")
        
        # Show summary
        st.success(f"Berikut data IP di kandang Anda, berdasarkan perhitungan maka nilainya {actual_ip:.2f} ({interpretasi_aktual}), dan berdasarkan prediksi dari system kami nilainya {prediction:.2f} ({interpretasi_prediksi})")

        # Generate and display AI recommendations
        st.subheader("Rekomendasi untuk Peningkatan Performa")
        recommendations = generate_ip_recommendations(
            ip_value=actual_ip, 
            fcr=fcr, 
            mortality_rate=persen_live_bird,
            age=age
        )
        
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to 5 recommendations
            st.write(f"{i}. {rec}")
        
        with st.expander("Lihat semua rekomendasi"):
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
                
        st.info("Rekomendasi ini dibuat berdasarkan analisis data performa dan praktik terbaik dalam budidaya broiler. Selalu konsultasikan dengan ahli peternakan untuk penerapan spesifik.")

# Add this code after the recommendations section (around line 673)

# Add visualization section
if os.path.exists('prediksi.csv'):
    st.write("---")
    st.subheader("Grafik Performa Produksi")
    
    # Load historical data
    hist_data = pd.read_csv('prediksi.csv')
    
    # Ensure column names are standardized
    column_mapping = {
        'actual_ip': 'IP_actual',
        'prediction': 'IP',
        'FCR': 'FCR_actual'
    }
    
    # Rename columns if needed
    for old_col, new_col in column_mapping.items():
        if old_col in hist_data.columns and new_col not in hist_data.columns:
            hist_data.rename(columns={old_col: new_col}, inplace=True)
    
    # Check if we have enough data to plot
    if len(hist_data) >= 3 and 'IP_actual' in hist_data.columns and 'IP' in hist_data.columns:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Perbandingan IP Aktual vs Prediksi", "Tren Performa", "Distribusi IP"])
        
        with tab1:
            # Scatter plot of Actual vs Predicted IP
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(hist_data['IP_actual'], hist_data['IP'], alpha=0.7)
            
            # Add perfect prediction line
            min_val = min(hist_data['IP_actual'].min(), hist_data['IP'].min())
            max_val = max(hist_data['IP_actual'].max(), hist_data['IP'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            ax.set_xlabel('IP Aktual')
            ax.set_ylabel('IP Prediksi')
            ax.set_title('Perbandingan IP Aktual vs Prediksi')
            ax.grid(True, alpha=0.3)
            
            # Annotate with R² value
            if 'model_r2' in st.session_state:
                r2 = st.session_state.model_r2
                ax.annotate(f'Model R² = {r2:.2f}', 
                          xy=(0.05, 0.95), xycoords='axes fraction',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            st.pyplot(fig)
            st.caption("Grafik perbandingan nilai IP aktual dengan nilai prediksi. Semakin dekat titik-titik dengan garis merah putus-putus, semakin akurat model.")
        
        with tab2:
            # Check if Date column exists
            date_col = 'Date' if 'Date' in hist_data.columns else None
            
            if date_col and not hist_data[date_col].isna().all():
                # Convert to datetime if needed
                hist_data[date_col] = pd.to_datetime(hist_data[date_col])
                hist_data = hist_data.sort_values(by=date_col)
                
                # Time series plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(hist_data[date_col], hist_data['IP_actual'], 'o-', label='IP Aktual')
                ax.plot(hist_data[date_col], hist_data['IP'], 's--', label='IP Prediksi')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Indeks Performans (IP)')
                ax.set_title('Tren IP Seiring Waktu')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                # Use record number as x-axis
                fig, ax = plt.subplots(figsize=(10, 6))
                x_values = range(len(hist_data))
                ax.plot(x_values, hist_data['IP_actual'], 'o-', label='IP Aktual')
                ax.plot(x_values, hist_data['IP'], 's--', label='IP Prediksi')
                ax.set_xlabel('Urutan Record')
                ax.set_ylabel('Indeks Performans (IP)')
                ax.set_title('Tren IP Seiring Waktu')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
            
            st.caption("Grafik menunjukkan perubahan nilai IP aktual dan prediksi seiring waktu.")
            
            # Add FCR trend if available
            if 'FCR_actual' in hist_data.columns:
                st.subheader("Tren FCR Seiring Waktu")
                fig, ax = plt.subplots(figsize=(10, 6))
                if date_col and not hist_data[date_col].isna().all():
                    ax.plot(hist_data[date_col], hist_data['FCR_actual'], 'o-', color='green')
                    ax.set_xlabel('Tanggal')
                else:
                    ax.plot(range(len(hist_data)), hist_data['FCR_actual'], 'o-', color='green')
                    ax.set_xlabel('Urutan Record')
                ax.set_ylabel('Feed Conversion Ratio (FCR)')
                ax.set_title('Tren FCR')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Tren FCR seiring waktu. Nilai FCR yang lebih rendah menunjukkan efisiensi pakan yang lebih baik.")
        
        with tab3:
            # Distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use transparency to show overlap
            ax.hist(hist_data['IP_actual'], alpha=0.6, bins=10, label='IP Aktual')
            ax.hist(hist_data['IP'], alpha=0.6, bins=10, label='IP Prediksi')
            
            ax.set_xlabel('Indeks Performans (IP)')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Nilai IP')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            st.caption("Distribusi nilai IP aktual dan prediksi. Grafik ini menunjukkan seberapa sering nilai IP tertentu muncul.")
            
            # Calculate stats
            mean_actual = hist_data['IP_actual'].mean()
            mean_pred = hist_data['IP'].mean()
            median_actual = hist_data['IP_actual'].median()
            
            # Display summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rata-rata IP Aktual", f"{mean_actual:.2f}")
                st.metric("Rata-rata IP Prediksi", f"{mean_pred:.2f}")
            with col2:
                st.metric("Median IP Aktual", f"{median_actual:.2f}")
                st.metric("Error Rata-rata", f"{abs(mean_actual - mean_pred):.2f}")
    else:
        st.info("Belum cukup data untuk menampilkan grafik perbandingan. Minimal dibutuhkan 3 record dengan nilai IP Aktual dan Prediksi.")

# Add Telegram bot controls with password protection
with st.sidebar.expander("Pengaturan"):
    # Initialize session state for authentication if not already done
    if 'bot_authenticated' not in st.session_state:
        st.session_state.bot_authenticated = False
    
    # Password verification section
    if not st.session_state.bot_authenticated:
        password = st.text_input("Masukkan password admin:", type="password", key="admin_password")
        
        # Check password from secrets file instead of hardcoding
        correct_password = st.secrets.get("ADMIN_PASSWORD", "AdmiNyA")  # Fallback to default if not in secrets
        
        if st.button("Verifikasi", key="verify_button"):
            if password == correct_password:
                st.session_state.bot_authenticated = True
                st.success("Verifikasi berhasil!")
                st.rerun()
            else:
                st.error("Password salah!")
    
    # Only show bot controls if authenticated
    if st.session_state.bot_authenticated:
        bot_status = get_telegram_bot_status()
        st.write(f"Status Bot: {'Aktif' if bot_status else 'Nonaktif'}")
        
        if st.button("Aktifkan Bot" if not bot_status else "Nonaktifkan Bot", key="toggle_bot_button"):
            new_status = toggle_telegram_bot()
            st.success(f"Bot sekarang {'Aktif' if new_status else 'Nonaktif'}")
            st.rerun()
        
        # Add threshold setting for high-performing model
        st.subheader("Pengaturan Model")
        
        if 'r2_threshold' not in st.session_state:
            st.session_state.r2_threshold = 0.90
            
        r2_threshold = st.slider(
            "Ambang R² untuk notifikasi model unggul", 
            min_value=0.80, 
            max_value=0.99, 
            value=st.session_state.r2_threshold,
            step=0.01,
            key="r2_threshold_slider"
        )
        st.session_state.r2_threshold = r2_threshold
        
        st.write(f"Model akan dianggap unggul jika R² ≥ {r2_threshold:.2f}")
        
        # Add checkbox to enable/disable auto-notification
        if 'auto_notify' not in st.session_state:
            st.session_state.auto_notify = True
            
        auto_notify = st.checkbox(
            "Kirim notifikasi otomatis untuk model unggul", 
            value=st.session_state.auto_notify,
            key="auto_notify_checkbox"
        )
        st.session_state.auto_notify = auto_notify
        
        # Test notification button
        if st.button("Uji Notifikasi", key="test_notification"):
            message = f"""<b>🧪 Uji Notifikasi Bot</b>
            
Bot berhasil dikonfigurasi dan berjalan dengan baik.
            
- Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Ambang R²: {r2_threshold:.2f}
- Auto-notifikasi: {'Aktif' if auto_notify else 'Nonaktif'}
"""
            send_result = send_to_telegram(message)
            if send_result:
                st.success("Notifikasi uji berhasil dikirim!")
            else:
                st.error("Gagal mengirim notifikasi uji. Periksa token dan chat ID.")
            
        if st.button("Keluar", key="logout_button"):
            st.session_state.bot_authenticated = False
            st.rerun()
        
        # Add data management section
        st.subheader("Manajemen Data")
        
        # Add button to clean prediksi.csv file
        if st.button("Bersihkan Duplikasi Data", key="clean_data"):
            try:
                if os.path.exists('prediksi.csv'):
                    # Read the file
                    df = pd.read_csv('prediksi.csv')
                    original_count = len(df)
                    
                    # Map possible column names to standardized names
                    column_mapping = {
                        'actual_ip': 'IP_actual',
                        'prediction': 'IP',
                        'FCR': 'FCR_actual'
                    }
                    
                    # Rename columns if needed
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns and new_col not in df.columns:
                            df.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Identify columns that exist in the dataframe for deduplication
                    duplicate_check_columns = []
                    all_possible_columns = ['Age', 'Total_Body_Weight', 'FCR', 'FCR_actual', 'Live_Bird', 
                                          'Ayam_Dipelihara', 'persen_Live_Bird', 'IP_actual']
                    
                    for col in all_possible_columns:
                        if col in df.columns:
                            duplicate_check_columns.append(col)
                    
                    if not duplicate_check_columns:
                        st.error("Tidak ada kolom yang sesuai untuk pemeriksaan duplikasi.")
                        st.stop()  # Use st.stop() instead of return
                        
                    # Remove duplicates based on available key fields
                    st.info(f"Menghapus duplikasi berdasarkan kolom: {', '.join(duplicate_check_columns)}")
                    df = df.drop_duplicates(subset=duplicate_check_columns, keep='first')
                    new_count = len(df)
                    
                    # Save cleaned data
                    df.to_csv('prediksi.csv', index=False)
                    
                    if original_count > new_count:
                        st.success(f"Pembersihan berhasil! {original_count - new_count} duplikasi telah dihapus.")
                    else:
                        st.info("Tidak ditemukan duplikasi data.")
                else:
                    st.warning("File prediksi.csv tidak ditemukan.")
            except Exception as e:
                st.error(f"Error saat membersihkan data: {str(e)}")
                st.info("Detail struktur file: " + ", ".join(df.columns) if 'df' in locals() and hasattr(df, 'columns') else "File tidak dapat dibaca")

# Information section
st.write("---")
st.write("Keterangan:")
st.write("1. IP (Indeks Performans) adalah nilai yang menggambarkan performa produksi broiler.")
st.write("2. Masukkan data produksi broiler Anda pada sidebar, dan system juga akan memberikan interpretasi IP yang dihasilkan, sekaligus memberikan prediksi IP berdasarkan data yang Anda masukkan, saat Anda klik tombol [Hitung Indeks Performans].")
st.write(f"Data prediksi akan semakin presisi jika Anda sering menggunakan system ini untuk menghitung IP Broiler ")

# Footer
st.markdown("---")
current_year = datetime.datetime.now().year

# Add IP detection and visitor counting
try:
    import requests
    
    # Function to log visitor and get count
    def log_visitor(ip_address):
        """Log visitor IP and return total unique visitor count"""
        visitors_file = 'visitors.csv'
        
        try:
            # Try to load existing visitor data
            if os.path.exists(visitors_file):
                visitors_df = pd.read_csv(visitors_file)
            else:
                # Create new dataframe if file doesn't exist
                visitors_df = pd.DataFrame(columns=['ip', 'timestamp'])
            
            # Add new visitor if not already in today's logs
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Filter to see if this IP has visited today
            today_visits = visitors_df[visitors_df['timestamp'].str.startswith(today)]
            if ip_address not in today_visits['ip'].values:
                # Add new row with current IP and timestamp
                new_row = pd.DataFrame({
                    'ip': [ip_address],
                    'timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                visitors_df = pd.concat([visitors_df, new_row], ignore_index=True)
                visitors_df.to_csv(visitors_file, index=False)
            
            # Count unique visitors
            unique_visitors = len(visitors_df['ip'].unique())
            return unique_visitors
            
        except Exception:
            # If any error occurs, return 1 (at least this visitor)
            return 1
    
    response = requests.get('https://api.ipify.org?format=json', timeout=3)
    ip_address = response.json()['ip']
    visitor_count = log_visitor(ip_address)
    
    st.text(f"© {current_year} Developed by: Galuh Adi Insani with ❤️. All rights reserved.")
    st.text(f"Visitor IP: {ip_address} | Total Visitors: {visitor_count}")
except Exception as e:
    st.text(f"© {current_year} Developed by: Galuh Adi Insani with ❤️. All rights reserved.")
    # Uncomment to show errors during debugging
    # st.text(f"Could not detect IP: {str(e)}")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)