import unittest
import pandas as pd
from gamefeed.final_evaluation import observed_ndcg, paired_user_interval, blend_rank_order


class FinalEvaluationTests(unittest.TestCase):
    def test_blend_preserves_candidates_and_agreeing_order(self):
        self.assertEqual(blend_rank_order(['z','a','b'],[.9,.8,.7]),['z','a','b'])
        ids = list(range(100))
        result = blend_rank_order(ids,list(range(100)))
        self.assertEqual(set(result),set(ids))
        self.assertEqual(len(result),100)
        self.assertEqual(result,blend_rank_order(ids,list(range(100))))
        self.assertEqual(blend_rank_order([],[]),[])
        with self.assertRaises(ValueError): blend_rank_order([1,1],[.1,.2])
        with self.assertRaises(ValueError): blend_rank_order([1],[float('nan')])

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
