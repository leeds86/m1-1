import unittest
import pandas as pd
from src.clean_data import validate_monthly, changes


class DataTests(unittest.TestCase):
    def test_duplicate_month_rejected(self):
        with self.assertRaises(ValueError):
            validate_monthly(pd.Series([1, 2], index=['201601', '201601']), '201601', '201602')

    def test_missing_month_rejected(self):
        with self.assertRaises(ValueError):
            validate_monthly(pd.Series([1, 3], index=['201601', '201603']), '201601', '201603')

    def test_missing_value_rejected(self):
        with self.assertRaises(ValueError):
            validate_monthly(pd.Series([float('nan')], index=['201601']), '201601', '201601')

    def test_price_percent_and_rate_points(self):
        frame = pd.DataFrame({'kospi': [100., 110.], 'base_rate': [2.5, 3.]})
        result = changes(frame)
        self.assertAlmostEqual(result.kospi.iloc[1], 10.)
        self.assertAlmostEqual(result.base_rate.iloc[1], .5)


if __name__ == '__main__':
    unittest.main()
