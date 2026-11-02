import unittest
import pandas as pd
from gamefeed.report_checks import gauc_coverage, random_recall_expectation


class ReportChecksTest(unittest.TestCase):
    def test_gauc_excludes_single_class_users_but_keeps_them_in_coverage_denominator(self):
        events = pd.DataFrame({'user_id': [1, 1, 2, 2, 3], 'label': [0, 1, 0, 0, 1]})
        result = gauc_coverage(events, 'label')
        self.assertEqual(result['valid_users'], 1)
        self.assertEqual(result['all_negative_users'], 1)
        self.assertEqual(result['all_positive_users'], 1)
        self.assertEqual(result['row_coverage'], 2/5)
        self.assertEqual(result['user_coverage'], 1/3)

    def test_random_expectation_keeps_unavailable_positive_and_caps_k(self):
        self.assertAlmostEqual(random_recall_expectation([1, 9], [1, 2, 3, 4], 2), .25)
        self.assertAlmostEqual(random_recall_expectation([1, 9], [1, 2], 100), .5)
        self.assertEqual(random_recall_expectation([1], [], 100), 0)
