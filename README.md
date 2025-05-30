# Poultry Performance Index (IP) Calculator and Predictor 🐔

This project provides a comprehensive tool for poultry farmers to monitor, analyze, and predict the Performance Index (IP) of their broiler chickens. Beyond just calculating and predicting IP, this application offers personalized recommendations to enhance farm management practices and visualizes key performance indicators over time. The goal is to empower farmers with data-driven insights to optimize productivity and improve decision-making in their broiler operations.

## Author

This project was created by Galuh Adi Insani.

## Features

*   **IP Calculation:** Accurate calculation of actual Performance Index (IP).
*   **IP Prediction:** Prediction of future IP using a Random Forest machine learning model.
*   **IP Interpretation:** Clear interpretation of IP scores (e.g., Kurang, Cukup, Baik, Sangat Baik, Istimewa).
*   **Personalized Recommendations:** Actionable advice based on IP scores and farm parameters to improve performance.
*   **Data Logging & History:** Saves user inputs and prediction results to `prediksi.csv` for tracking.
*   **Automatic Model Retraining:** The system can automatically retrain the prediction model when new data is available or if the existing model is outdated/corrupted.
*   **Data Visualization:** Interactive charts and graphs displaying performance trends, actual vs. predicted IP, data distributions, and parameter correlations.
*   **Telegram Bot Notifications:** Sends alerts for significant events like high-performing model training or data management tasks (password-protected).
*   **Visitor Tracking:** Logs unique visitors to the application.
*   **Admin Section:** Password-protected area for:
    *   Managing Telegram bot settings (enable/disable, API credentials).
    *   Setting R² thresholds for model notifications.
    *   Manually triggering model retraining.
    *   Cleaning duplicate entries in `prediksi.csv`.
    *   Toggling graph visibility for users.

## How to Run the Application

**Prerequisites:**
*   Python 3.x installed.
*   `pip` (Python package installer).

**Steps:**
1.  Clone the repository:
    ```bash
    git clone https://github.com/adiorany3/your-repo-name.git 
    ```
    (Replace the URL with the actual repository URL if available, or instruct users to clone the current repository they are viewing).
2.  Navigate to the project directory:
    ```bash
    cd your-repo-name 
    ```
    (Replace `your-repo-name` with the actual directory name).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

**Initial Setup Notes:**
*   The file `data_kandang.csv` can be used for initial model training if the pre-trained model file, `poultry_rf_model.joblib`, is not present.
*   `poultry_rf_model.joblib` is the file where the trained Random Forest machine learning model is stored.
*   For Telegram bot integration (notifications) and admin panel access, you will need to configure a `secrets.toml` file within a `.streamlit` directory. Detailed configuration for this will be covered in a subsequent section or a dedicated configuration guide.

## File Structure

*   `app.py`: The main Streamlit web application script containing the user interface and core logic.
*   `RandomForestPrediction.py`: Python script responsible for training the Random Forest machine learning model.
*   `poultry_rf_model.joblib`: The primary file storing the trained Random Forest model. This is loaded by `app.py` for predictions.
*   `modelawal/poultry_rf_model.joblib`: Contains an initial or backup version of the trained model.
*   `data_kandang.csv`: Sample or initial dataset that can be used for model training if `poultry_rf_model.joblib` is not available.
*   `prediksi.csv`: CSV file used to log user inputs, calculated actual IP, and predicted IP values for historical tracking and potential model retraining.
*   `requirements.txt`: Lists all Python dependencies required to run the project (e.g., Streamlit, Pandas, Scikit-learn).
*   `README.md`: This documentation file, providing information about the project.
*   `.streamlit/secrets.toml`: (User-created) Configuration file for Streamlit Cloud deployment, storing sensitive information like Telegram bot token, chat ID, and the admin panel password.
*   `header.jpeg`: Image file (likely a banner or logo) used in the application's user interface.
*   `visitors.csv`: Logs IP addresses and timestamps of unique visitors to the application for analytics.
*   `test_app.py`, `test_RandomForestPrediction.py`, `test_app_calculations.py`, `test_setup.py`: Python scripts containing unit tests to ensure the functionality of different parts of the application.
*   `importjoblib.py`: (Likely a utility script, purpose might need further clarification if essential for users, otherwise can be omitted if it's a dev tool).

## Model Details

The core of the prediction functionality lies in a machine learning model trained to estimate the Poultry Performance Index (IP).

*   **Algorithm:** The prediction model utilizes a **Random Forest Regressor**, a robust ensemble learning method implemented in the `scikit-learn` library.
*   **Target Variable:** The model is trained to predict the **`IP`** (Indeks Performa).
*   **Key Features:** The primary input features used by the model for prediction are:
    *   `Age` (Umur Panen - days)
    *   `Total_Body_Weight` (Total Bobot Badan - kg)
    *   `FCR` (Feed Conversion Ratio)
    *   `Live_Bird` (Jumlah Ayam Hidup - heads)
    *   `Ayam_Dipelihara` (Jumlah Ayam Awal Dipelihara - heads)
    *   `persen_Live_Bird` (Persentase Ayam Hidup - %)
*   **Training Data:** The model is initially trained using the data provided in `data_kandang.csv`. It is designed to be periodically retrained using the accumulated data from user inputs and results, which are logged in `prediksi.csv`. This allows the model to adapt and improve over time.
*   **Evaluation Metrics:** The performance of the Random Forest model is evaluated using the following standard regression metrics:
    *   **Mean Squared Error (MSE):** Measures the average squared difference between the actual and predicted IP values. Lower values indicate a better fit.
    *   **R² Score (Coefficient of Determination):** Represents the proportion of the variance in the dependent variable (IP) that is predictable from the independent variables (features). Values closer to 1 indicate a better model performance.
*   **Model Persistence:** The trained machine learning model is serialized and saved to the file `poultry_rf_model.joblib` using `joblib`. This file is then loaded by the application to make predictions without needing to retrain the model every time the application starts.

## Telegram Integration

This application features an optional integration with a Telegram bot, allowing for automated notifications and file sharing directly to your Telegram account. This functionality is managed and configured via the password-protected Admin Section within the application.

**Key Bot Capabilities:**

*   **Model Performance Alerts:** Receive notifications when a newly retrained machine learning model achieves an R² score above a user-defined threshold (set in the Admin Section).
*   **File Dispatch:** Send important files, such as the trained model (`poultry_rf_model.joblib`) or the prediction history (`prediksi.csv`), directly to your Telegram chat.
*   **Data Management Notifications:** Get alerts regarding data management activities, for instance, before and after the `prediksi.csv` file is cleaned of duplicate entries.
*   **Connectivity Testing:** A test notification feature is available in the Admin Section to verify that the Telegram bot is correctly configured and connected.

**Configuration:**

To enable Telegram integration, you must provide your Telegram Bot Token and Chat ID. This is done by creating or editing the `secrets.toml` file located in the `.streamlit` directory of your project. Add the following lines to this file:

```toml
TELEGRAM_BOT_TOKEN = "your_actual_bot_token_here"
TELEGRAM_CHAT_ID = "your_actual_chat_id_here"
```

**Important:**
*   Replace `"your_actual_bot_token_here"` with the token you receive from BotFather on Telegram.
*   Replace `"your_actual_chat_id_here"` with your personal or group chat ID where you want to receive messages.
*   The Telegram bot functionality will remain inactive if these secrets are not correctly configured in `secrets.toml` or if the "Enable Telegram Bot" option is turned off in the Admin Section of the application.

## Admin Section

The application includes a dedicated "Pengaturan" (Settings) area, accessible from the sidebar, which serves as the admin panel. This section is password-protected and provides administrators with tools to manage the application, model, and user-facing features.

**Accessing the Admin Section:**

To access the admin panel, you must first set an admin password. This is configured in the `.streamlit/secrets.toml` file by adding the following entry:

```toml
ADMIN_PASSWORD = "your_secure_password_here"
```
Replace `"your_secure_password_here"` with a strong, unique password. If this password is not set in the `secrets.toml` file, the Admin Section will not be accessible.

**Admin Panel Functionalities:**

The Admin Section offers the following management capabilities:

*   **Telegram Bot Management:**
    *   Enable or disable the Telegram bot integration for sending notifications and files.
    *   Send a test notification to verify the Telegram bot configuration (Bot Token and Chat ID must be set in `secrets.toml` as detailed in the "Telegram Integration" section).

*   **Model & Notification Settings:**
    *   Define the R² score threshold that determines if a newly retrained model is considered "high-performing."
    *   Enable or disable automatic Telegram notifications for when a model meets this high-performance R² threshold.

*   **Model Training & Data Management:**
    *   Manually initiate the model retraining process. This typically uses new data accumulated in `prediksi.csv` along with the initial `data_kandang.csv`.
    *   Perform data cleaning by removing duplicate entries from the `prediksi.csv` file to ensure data quality for retraining.
    *   Option to trigger model retraining using the complete dataset, combining all records from both `data_kandang.csv` and `prediksi.csv`.

*   **User Interface (UI) Settings:**
    *   Control the visibility of performance graphs and data visualizations for regular users of the application.

*   **Security:**
    *   A "Logout" button is provided to securely end the admin session and protect the admin panel from unauthorized access.

## Contributing

Contributions to enhance this project are highly welcome! Whether you have ideas for improvements, new features, or have identified bugs, your input is valuable.

Here are a few ways you can contribute:

*   **Report Bugs:** If you encounter any errors or unexpected behavior, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the bug.
*   **Suggest Features:** Have an idea for a new functionality or an enhancement to an existing one? Open an issue to discuss its feasibility and how it might be implemented.
*   **Improve Documentation:** If you find areas in this README or other documentation that could be clearer or more comprehensive, feel free to suggest changes or submit a pull request with your improvements.
*   **Submit Pull Requests:** For code changes, bug fixes, or new features, you can fork the repository, make your changes in a separate branch, and then submit a pull request.

**Guidelines:**

*   Before making significant changes, it's a good idea to **open an issue first** to discuss your proposed contribution. This helps ensure that your work aligns with the project's goals and avoids duplication of effort.
*   Please try to follow the existing code style and conventions used throughout the project.
*   Write clear and concise commit messages that explain the purpose of your changes.

We appreciate your help in making this tool better for all users!
