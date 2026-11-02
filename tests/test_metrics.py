import unittest

from gamefeed.metrics import binary_metrics, hit_rate_at_k, ndcg_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_recall_and_hit_rate(self) -> None:
        recommended = ["a", "b", "c"]
        positives = {"b", "d"}
        self.assertEqual(recall_at_k(recommended, positives, 3), 0.5)
        self.assertEqual(hit_rate_at_k(recommended, positives, 1), 0.0)
        self.assertEqual(hit_rate_at_k(recommended, positives, 2), 1.0)

    def test_auc_hand_example(self) -> None:
        result = binary_metrics([1, 1, 0, 0], [0.8, 0.4, 0.6, 0.2])
        self.assertAlmostEqual(result["auc"], 0.75)

    def test_ndcg_hand_example(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([1, 0, 1], 3), 0.919720789, places=6)


if __name__ == "__main__":
    unittest.main()

