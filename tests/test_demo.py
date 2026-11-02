import unittest

from gamefeed.demo import author_capped_rerank


class DemoTests(unittest.TestCase):
    def test_author_cap_is_applied_after_scoring(self):
        rows = [
            {"video_id": "a", "author_id": "same", "score": 0.9},
            {"video_id": "b", "author_id": "same", "score": 0.8},
            {"video_id": "c", "author_id": "same", "score": 0.7},
            {"video_id": "d", "author_id": "other", "score": 0.6},
        ]
        result = author_capped_rerank(rows, top_k=3, max_per_author=2)
        self.assertEqual([row["video_id"] for row in result], ["a", "b", "d"])

    def test_missing_authors_and_captions_are_independent(self):
        rows = [dict(video_id=str(i), author_id=author, caption='', score=1.)
                for i, author in enumerate([None, '', float('nan'), 'MISSING', -1])]
        self.assertEqual(len(author_capped_rerank(rows, 10, 2)), 5)

    def test_duplicate_caption_keeps_best_and_fills_from_remaining(self):
        rows = [dict(video_id='a', author_id='1', caption='同一 文案', score=.9),
                dict(video_id='b', author_id='2', caption='同一  文案', score=.8),
                dict(video_id='c', author_id='3', caption='另一文案', score=.7)]
        self.assertEqual([r['video_id'] for r in author_capped_rerank(rows, 2)], ['a', 'c'])
        self.assertEqual([r['video_id'] for r in author_capped_rerank(rows, 2, deduplicate_caption=False)], ['a', 'b'])


if __name__ == "__main__":
    unittest.main()
