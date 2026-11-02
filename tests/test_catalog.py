import unittest

import pandas as pd

from gamefeed.catalog import add_available_from, available_items


class CatalogTests(unittest.TestCase):
    def test_items_enter_catalog_on_following_day(self) -> None:
        items = pd.DataFrame(
            {"video_id": [1, 2], "upload_date": pd.to_datetime(["2022-04-10", None])}
        )
        events = pd.DataFrame(
            {"video_id": [1, 2], "date": pd.to_datetime(["2022-04-12", "2022-04-15"])}
        )
        result = add_available_from(items, events)
        self.assertEqual(set(available_items(result, pd.Timestamp("2022-04-11"))["video_id"]), {1})
        self.assertEqual(set(available_items(result, pd.Timestamp("2022-04-16"))["video_id"]), {1, 2})


if __name__ == "__main__":
    unittest.main()

