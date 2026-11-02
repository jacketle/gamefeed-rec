import unittest

import numpy as np

from gamefeed.retrieval import ExactIndex, l2_normalize


class RetrievalTests(unittest.TestCase):
    def test_vector_rows_map_back_to_item_ids(self) -> None:
        ids = np.array(["v10", "v20", "v30"])
        vectors = l2_normalize(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
        index = ExactIndex.build(ids, vectors)
        scores, result = index.search(l2_normalize(np.array([[0, 1]], dtype=np.float32)), 2)
        self.assertEqual(result[0, 0], "v20")
        self.assertEqual(result.shape, scores.shape)


if __name__ == "__main__":
    unittest.main()

