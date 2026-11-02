import unittest

import pandas as pd

from gamefeed.baselines import fit_itemcf, itemcf_recommendations, reciprocal_rank_fusion


class ItemCFTests(unittest.TestCase):
    def test_items_shared_by_users_become_neighbors(self):
        events = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "video_id": ["a", "b", "a", "b"],
                "long_view": [1, 1, 1, 1],
                "is_like": [0, 0, 0, 0],
            }
        )
        neighbors = fit_itemcf(events)
        self.assertEqual(neighbors["a"][0][0], "b")
        self.assertEqual(neighbors["b"][0][0], "a")

    def test_recommendation_allows_history_but_excludes_future_items(self):
        events = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "video_id": ["a", "b", "a", "b"],
                "long_view": [1, 1, 1, 1],
                "is_like": [0, 0, 0, 0],
            }
        )
        queries = pd.DataFrame(
            {
                "user_id": ["u3"],
                "date": [pd.Timestamp("2022-04-22")],
                "split": ["dev"],
                "history_ids": [["a"]],
                "positive_ids": [["b"]],
            }
        )
        items = pd.DataFrame(
            {
                "video_id": ["a", "b", "future"],
                "available_from": pd.to_datetime(["2022-04-01", "2022-04-01", "2022-04-23"]),
            }
        )
        result = itemcf_recommendations(events, queries, items, k=3)
        recommended = result.iloc[0]["recommended_ids"]
        self.assertEqual(recommended[0], "b")
        self.assertIn("a", recommended)
        self.assertNotIn("future", recommended)

    def test_rank_fusion_deduplicates_and_tracks_routes(self):
        common = {"user_id": ["u"], "date": [pd.Timestamp("2022-04-22")], "positive_ids": [["b"]]}
        first = pd.DataFrame({**common, "recommended_ids": [["a", "b"]]})
        second = pd.DataFrame({**common, "recommended_ids": [["b", "c"]]})
        result = reciprocal_rank_fusion({"first": first, "second": second}, {"first": 1.0, "second": 1.0}, k=3)
        self.assertEqual(result.iloc[0]["recommended_ids"][0], "b")
        self.assertEqual(set(result.iloc[0]["recall_channels"][0]), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
