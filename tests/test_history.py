import unittest

import pandas as pd

from gamefeed.history import attach_state, build_daily_state


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            [
                {"event_id": "e1", "user_id": "u1", "video_id": "v1", "date": pd.Timestamp("2022-04-08"), "time_ms": 1, "long_view": 1, "split": "train"},
                {"event_id": "e2", "user_id": "u1", "video_id": "v2", "date": pd.Timestamp("2022-04-08"), "time_ms": 2, "long_view": 0, "split": "train"},
                {"event_id": "e3", "user_id": "u1", "video_id": "v3", "date": pd.Timestamp("2022-04-09"), "time_ms": 3, "long_view": 1, "split": "train"},
                {"event_id": "e4", "user_id": "u1", "video_id": "v4", "date": pd.Timestamp("2022-04-10"), "time_ms": 4, "long_view": 0, "split": "train"},
            ]
        )

    def test_history_uses_previous_days_only(self) -> None:
        states, queries, _ = build_daily_state(self.events, history_length=20)
        first = states.loc[states["date"].eq(pd.Timestamp("2022-04-08")), "history_ids"].iloc[0]
        second = states.loc[states["date"].eq(pd.Timestamp("2022-04-09")), "history_ids"].iloc[0]
        third = states.loc[states["date"].eq(pd.Timestamp("2022-04-10")), "history_ids"].iloc[0]
        self.assertEqual(first, [])
        self.assertEqual(second, ["v1"])
        self.assertEqual(third, ["v1", "v3"])
        self.assertNotIn("v3", second)
        self.assertEqual(len(queries), 2)

    def test_state_join_does_not_change_row_count(self) -> None:
        states, _, _ = build_daily_state(self.events)
        merged = attach_state(self.events, states)
        self.assertEqual(len(merged), len(self.events))
        self.assertFalse(merged["history_ids"].isna().any())


if __name__ == "__main__":
    unittest.main()

