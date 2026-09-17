import json
import unittest
import numpy as np
from src.export import safe


class ExportTests(unittest.TestCase):
    def test_nonfinite_values_become_json_null(self):
        result = json.dumps(safe({'r': np.nan, 'upper': np.inf, 'zero': 0.0}), allow_nan=False)
        self.assertEqual(json.loads(result), {'r': None, 'upper': None, 'zero': 0.0})

    def test_numpy_integer_is_json_serializable(self):
        self.assertEqual(json.loads(json.dumps(safe({'n':np.int64(120)}))), {'n':120})
