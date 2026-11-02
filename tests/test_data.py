import unittest

import pandas as pd

from gamefeed.data import assign_split
from gamefeed.schema import canonicalize_events


class SplitTests(unittest.TestCase):
    def test_time_split_is_disjoint(self) -> None:
        events = pd.DataFrame(
            {
                "event_id": ["a", "b", "c", "d"],
                "date": pd.to_datetime(["2022-04-08", "2022-04-21", "2022-04-22", "2022-04-29"]),
            }
        )
        config = {
            "train_start": "2022-04-08", "train_end": "2022-04-21",
            "dev_start": "2022-04-22", "dev_end": "2022-04-28",
            "test_start": "2022-04-29", "test_end": "2022-05-08",
        }
        result = assign_split(events, config)
        self.assertEqual(result["split"].tolist(), ["train", "train", "dev", "test"])
        split_sets = [set(group["event_id"]) for _, group in result.groupby("split")]
        self.assertEqual(sum(len(value) for value in split_sets), len(set.union(*split_sets)))

    def test_many_to_one_item_join_keeps_rows(self) -> None:
        events = pd.DataFrame({"video_id": [1, 1, 2], "label": [0, 1, 1]})
        items = pd.DataFrame({"video_id": [1, 2], "tag": [10, 20]})
        merged = events.merge(items, on="video_id", how="left", validate="many_to_one")
        self.assertEqual(len(merged), len(events))

    def test_compact_integer_date_is_parsed_as_calendar_date(self) -> None:
        raw = pd.DataFrame(
            {"user_id": [1], "video_id": [2], "date": [20220421], "time_ms": [1650485801301], "long_view": [1], "is_like": [0]}
        )
        result = canonicalize_events(raw, "sample.csv")
        self.assertEqual(result.loc[0, "date"], pd.Timestamp("2022-04-21"))


if __name__ == "__main__":
    unittest.main()

