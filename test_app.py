import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from app import load_data, train_model, interpret_ip

# test_app.py

class TestApp(unittest.TestCase):

    def setUp(self):
        # Generate sample data
        self.data = pd.DataFrame({
            'Age': np.random.randint(1, 46, 100),
            'Total_Body_Weight': np.random.uniform(0, 23000, 100),
            'FCR': np.random.uniform(1.0, 2.0, 100),
            'Live_Bird': np.random.randint(0, 30000, 100),
            'Ayam_Dipelihara': np.random.randint(0, 30000, 100),
            'persen_Live_Bird': np.random.uniform(0, 100, 100),
            'IP': np.random.uniform(0, 500, 100)
        })

    def test_train_model(self):
        model, mse, r2 = train_model(self.data)
        self.assertIsNotNone(model)
        self.assertGreater(r2, 0.0)

    def test_model_performance(self):
        # Split data
        X = self.data[['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']]
        y = self.data['IP']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_class = [interpret_ip(ip) for ip in y_pred]
        y_test_class = [interpret_ip(ip) for ip in y_test]

        # Calculate metrics
        accuracy = accuracy_score(y_test_class, y_pred_class)
        precision = precision_score(y_test_class, y_pred_class, average='weighted')
        recall = recall_score(y_test_class, y_pred_class, average='weighted')
        f1 = f1_score(y_test_class, y_pred_class, average='weighted')

        # Assert metrics
        self.assertGreater(accuracy, 0.0)
        self.assertGreater(precision, 0.0)
        self.assertGreater(recall, 0.0)
        self.assertGreater(f1, 0.0)

if __name__ == '__main__':
    unittest.main()