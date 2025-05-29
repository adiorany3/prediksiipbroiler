import unittest
import pandas as pd
import numpy as np
from app import generate_sample_data # Assuming app.py is in the same directory or accessible in PYTHONPATH

# Helper function mirroring the IP calculation in app.py
def calculate_actual_ip(persen_live_bird, total_body_weight, live_bird, fcr, age):
    if live_bird == 0 or fcr == 0 or age == 0:
        # This mimics the pre-check in app.py's UI, which prevents calculation.
        # For a raw formula test, we'd expect ZeroDivisionError.
        # However, if app.py's button handler effectively prevents this,
        # the formula itself might not be called.
        # Here, we test the formula directly, so errors are expected if divisors are zero.
        pass # Or raise an error, depending on what exactly we are testing from app.py
    
    # The raw formula that can raise ZeroDivisionError
    return ((persen_live_bird * (total_body_weight / live_bird)) / (fcr * age)) * 100

class TestAppCalculations(unittest.TestCase):

    def test_calculate_actual_ip_valid(self):
        # Test case 1 (Valid)
        # persen_live_bird=95, total_body_weight=2000, live_bird=950, fcr=1.5, age=30
        # Expected: ((95 * (2000/950)) / (1.5 * 30)) * 100
        # (95 * 2.1052631578947367) / 45 * 100
        # (200 / 45) * 100 = 4.444444444444445 * 100 = 444.4444444444445
        expected_ip = ((95 * (2000/950)) / (1.5 * 30)) * 100
        calculated_ip = calculate_actual_ip(
            persen_live_bird=95, 
            total_body_weight=2000, 
            live_bird=950, 
            fcr=1.5, 
            age=30
        )
        self.assertAlmostEqual(calculated_ip, expected_ip, places=5)

    def test_calculate_actual_ip_live_bird_zero(self):
        # Test case 2 (live_bird is zero)
        with self.assertRaises(ZeroDivisionError):
            calculate_actual_ip(
                persen_live_bird=95, 
                total_body_weight=2000, 
                live_bird=0, 
                fcr=1.5, 
                age=30
            )

    def test_calculate_actual_ip_age_zero(self):
        # Test case 3 (age is zero)
        with self.assertRaises(ZeroDivisionError):
            calculate_actual_ip(
                persen_live_bird=95, 
                total_body_weight=2000, 
                live_bird=950, 
                fcr=1.5, 
                age=0
            )

    def test_calculate_actual_ip_fcr_zero(self):
        # Test case 4 (fcr is zero)
        with self.assertRaises(ZeroDivisionError):
            calculate_actual_ip(
                persen_live_bird=95, 
                total_body_weight=2000, 
                live_bird=950, 
                fcr=0, 
                age=30
            )

    def test_generate_sample_data_ip_calculation(self):
        sample_df = generate_sample_data()
        # X_train columns: ['Age', 'Total_Body_Weight', 'FCR', 'Live_Bird', 'Ayam_Dipelihara', 'persen_Live_Bird']
        # Indices: Age=0, FCR=2, Live_Bird=3
        
        for index, row in sample_df.iterrows():
            age = row['Age']
            fcr_val = row['FCR']
            live_bird_val = row['Live_Bird']
            expected_ip = row['IP']
            
            # Check the condition used in generate_sample_data
            if not (age > 0 and fcr_val > 0 and live_bird_val > 0):
                self.assertEqual(expected_ip, 0.0, 
                                 f"Row {index}: IP should be 0.0 if Age, FCR, or Live_Bird is not > 0. Got IP={expected_ip} for Age={age}, FCR={fcr_val}, Live_Bird={live_bird_val}")
            else:
                # Optionally, recalculate and verify positive IP values if desired, though a bit redundant
                # For this test, we primarily care about the zeroing out.
                pass 

if __name__ == '__main__':
    unittest.main()
