import unittest
from gamefeed.fusion_experiment import quota_list


class QuotaTests(unittest.TestCase):
    def test_zero_semantic_slots_keep_popular_and_cf_membership(self):
        routes = {'popularity': list(range(10)), 'itemcf': list(range(5,105)),
                  'semantic': list(range(200,300))}
        result = quota_list(routes, 0)
        self.assertEqual(len(result), 100)
        self.assertEqual(len(set(result)), 100)
        self.assertTrue(set(range(10)).issubset(result))
        self.assertFalse(set(result) & set(routes['semantic']))

    def test_semantic_quota_deduplicates_overlapping_routes(self):
        routes = {'popularity': list(range(10)), 'itemcf': list(range(100)),
                  'semantic': list(range(5,15)) + list(range(200,290))}
        result = quota_list(routes, 20)
        self.assertEqual(len(result), 100)
        self.assertEqual(len(set(result)), 100)
        self.assertTrue(set(routes['semantic'][:20]).issubset(result))
        self.assertTrue(set(range(10)).issubset(result))

