import unittest
from main import *

class TestMain(unittest.TestCase):

    def test_load_dataframe(self):
        # Test with valid file
        file = 'test_data.csv'
        df, numeric_cols, categorical_cols, target_column, regression_type = load_dataframe(file)
        self.assertIsNotNone(df)
        self.assertIsInstance(numeric_cols, list)
        self.assertIsInstance(categorical_cols, list)
        self.assertIsInstance(target_column, str)
        self.assertIn(regression_type, ['classification', 'regression'])

        # Test with invalid file
        file = 'invalid.csv'
        df, numeric_cols, categorical_cols, target_column, regression_type = load_dataframe(file)
        self.assertIsNone(df)

    def test_load_url(self):
        # Test with valid URL
        url = 'https://test-data.csv'
        df, numeric_cols, categorical_cols, target_column, regression_type = load_url(url)
        self.assertIsNotNone(df)
        self.assertIsInstance(numeric_cols, list)
        self.assertIsInstance(categorical_cols, list)
        self.assertIsInstance(target_column, str)
        self.assertIn(regression_type, ['classification', 'regression'])
        
        # Test with invalid URL
        url = 'invalidurl'
        df, numeric_cols, categorical_cols, target_column, regression_type = load_url(url)
        self.assertIsNone(df)

    def test_random_forest(self):
        # Test with valid inputs
        df = 'dummy_df'
        numeric_cols = ['num1', 'num2']
        categorical_cols = ['cat1', 'cat2']
        target_column = 'target'
        regression_type = 'classification'
        
        model = random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
        self.assertIsNotNone(model)
        
        # Test with invalid inputs
        df = None
        model = random_forest(df, numeric_cols, categorical_cols, target_column, regression_type)
        self.assertIsNone(model)
        
