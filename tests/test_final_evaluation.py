import unittest
import pandas as pd
from gamefeed.final_evaluation import observed_ndcg, paired_user_interval


class FinalEvaluationTests(unittest.TestCase):
    def test_ndcg_counts_positives_missing_from_candidates(self):
        self.assertAlmostEqual(observed_ndcg(['a'], ['a','b'], 2), 1/(1+1/1.584962500721156))

    def test_cluster_sampling_retains_query_weighting(self):
        frame=pd.DataFrame({'user_id':[1,1,1,2], 'delta':[1.,1.,1.,0.]})
        result=paired_user_interval(frame,repeats=100,seed=3)
        self.assertEqual(result['estimate'], .75)
        self.assertEqual(result['users'], 2)
        self.assertEqual(result['queries'], 4)
        self.assertEqual(result,paired_user_interval(frame,repeats=100,seed=3))


if __name__=='__main__':
    unittest.main()
