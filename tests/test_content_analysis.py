import unittest
import pandas as pd
from gamefeed.content_analysis import positive_metrics


class ContentMetricsTests(unittest.TestCase):
    def test_seen_hit_cannot_count_as_unseen_hit(self):
        rows = pd.DataFrame([dict(positive_ids=['seen', 'new'], recommended_ids=['seen'])])
        result = positive_metrics(rows, {'new'})
        self.assertEqual(result['positives'], 1)
        self.assertEqual(result['hits'], 0)
