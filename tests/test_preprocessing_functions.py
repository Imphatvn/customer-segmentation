import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import preprocessing_functions as preprocessing


class NegativePredictor:
    def __init__(self, **kwargs):
        pass

    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.full(len(features), -3.0)


class PreprocessingSafeguardTests(unittest.TestCase):
    def test_vegetable_imputation_floors_negative_predictions(self):
        customers = pd.DataFrame({
            "spend_per_year_vegetables": [12.0, np.nan],
            "lifetime_spend_vegetables": [120.0, 0.0],
            "customer_lifetime": [10, 5],
        })

        with patch.object(preprocessing, "XGBRegressor", NegativePredictor):
            result = preprocessing.impute_vegetables_with_boosting(customers)

        self.assertEqual(result.loc[1, "spend_per_year_vegetables"], 0.0)
        self.assertTrue(pd.isna(customers.loc[1, "spend_per_year_vegetables"]))

    def test_distinct_store_imputation_rejects_invalid_counts(self):
        invalid_counts = ([1, 0], [1, 1.5])
        for values in invalid_counts:
            with self.subTest(values=values):
                customers = pd.DataFrame({"distinct_stores_visited": values})
                with self.assertRaises(ValueError):
                    preprocessing.impute_distinct_stores(customers)

    def test_customer_validation_reports_invalid_values(self):
        customers = pd.DataFrame({
            "customer_id": [1, 1],
            "pct_prod_in_promotion": [0.2, 1.2],
            "customer_lifetime": [5, -1],
            "spend_per_year_vegetables": [10.0, -2.0],
            "spend_per_year_groceries": [4.0, np.nan],
        })

        checks = preprocessing.validate_customer_data(customers)

        self.assertFalse(checks["unique_customer_id"])
        self.assertFalse(checks["promotion_within_range"])
        self.assertFalse(checks["nonnegative_customer_lifetime"])
        self.assertFalse(checks["no_missing_annual_spend"])
        self.assertFalse(checks["no_negative_annual_spend"])


if __name__ == "__main__":
    unittest.main()
