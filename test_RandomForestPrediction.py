import unittest
import os
import pandas as pd
import RandomForestPrediction # Import the refactored script

class TestRandomForestPrediction(unittest.TestCase):

    def setUp(self):
        """Create a dummy data_kandang.csv for testing."""
        self.dummy_csv_path = 'data_kandang.csv'
        self.model_path = 'poultry_rf_model.joblib'
        
        data = {
            'Age': [30, 35, 28, 32, 38],
            'Total_Body_Weight': [2000, 2200, 1900, 2100, 2300],
            'FCR': [1.5, 1.6, 1.45, 1.55, 1.65],
            'Live_Bird': [950, 1900, 960, 1920, 940], # Corrected: ensure this is less than Ayam_Dipelihara
            'Ayam_Dipelihara': [1000, 2000, 1000, 2000, 1000],
            'persen_Live_Bird': [95.0, 95.0, 96.0, 96.0, 94.0],
            'IP': [350, 360, 355, 365, 340]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dummy_csv_path, index=False)

    def test_run_script_and_model_creation(self):
        """Test if the script runs and creates the model file."""
        # Ensure the model file does not exist before running
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            
        RandomForestPrediction.main() # Call the main function from the script
        
        self.assertTrue(os.path.exists(self.model_path), 
                        f"Model file '{self.model_path}' was not created.")

    def tearDown(self):
        """Clean up dummy CSV and model file."""
        if os.path.exists(self.dummy_csv_path):
            os.remove(self.dummy_csv_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
