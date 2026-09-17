import unittest
import pandas as pd
from src.analyze import pair, lag_pairs
from src.forecast import predict, backtest


class AnalysisTests(unittest.TestCase):
    def test_lag_uses_future_target(self):
        x = pd.Series([2., 9., 1., 5., 3., 8.])
        y = pd.Series([99., 2., 9., 1., 5., 3.])
        r = lag_pairs(x, y, 1)
        self.assertAlmostEqual(r['r'], 1.)
        self.assertEqual(r['n'], 5)

    def test_constant_is_undefined(self):
        self.assertIsNone(pair(pd.Series([1., 1., 1.]), pd.Series([2., 3., 4.]))['r'])

    def test_pairs_drop_only_unavailable_observations(self):
        r = pair(pd.Series([1., 2., 3., None]), pd.Series([3., 2., 1., 7.]))
        self.assertAlmostEqual(r['r'], -1.)
        self.assertEqual(r['n'], 3)

    def test_naive_constant(self):
        self.assertEqual(list(predict([100.] * 60, 3, 'naive')), [100., 100., 100.])

    def test_future_does_not_change_earlier_prediction(self):
        values = pd.Series([float(i % 7 + 100) for i in range(70)])
        a = backtest(values, initial=60, horizons=(1,), models=('naive',))
        values.iloc[65:] = 999.
        b = backtest(values, initial=60, horizons=(1,), models=('naive',))
        self.assertEqual(a[0]['prediction'], b[0]['prediction'])
        self.assertEqual(a[0]['origin'], 59)


if __name__ == '__main__':
    unittest.main()
