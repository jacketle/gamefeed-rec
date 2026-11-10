import unittest
import pandas as pd
import numpy as np
import torch
from gamefeed.sequence_evaluation import paired_interval, s0_scores
from gamefeed.sequence_training import Inputs


class BootstrapTests(unittest.TestCase):
    def test_identical_predictions_have_zero_interval(self):
        f=pd.DataFrame(dict(user_id=[1,1,2,2,3],long_view=[0,1,0,1,1]))
        p=np.array([.1,.9,.2,.8,.6])
        r=paired_interval(f,[p]*3,[p]*3,repeats=30)
        self.assertEqual(r['mean_difference'],0)
        self.assertEqual(r['ci95'],[0.,0.])

    def test_perfect_reversal(self):
        f=pd.DataFrame(dict(user_id=[1,1,2,2],long_view=[0,1,0,1]))
        a=np.array([.9,.1,.9,.1]);b=1-a
        r=paired_interval(f,[a]*3,[b]*3,repeats=30)
        self.assertEqual(r['ci95'],[1.,1.])

    def test_propensity_is_constant_within_user_and_scene(self):
        history=pd.DataFrame(dict(user_id=[1,1,2,2],tab=[0,0,0,0],long_view=[1,0,1,1],
                                  split=['train']*4,is_rand=[0]*4))
        target=pd.DataFrame(dict(user_id=[1,1,2],tab=[0,0,0]))
        p=s0_scores(history,target)
        self.assertEqual(p[0],p[1])
        self.assertGreater(p[2],p[0])

    def test_delayed_history_masks_equal_or_later_completion(self):
        inputs=Inputs.__new__(Inputs)
        inputs.use_history_matches=False
        inputs.length=2;inputs.lengths=np.array([2]);inputs.ends=np.array([2])
        inputs.pool=np.array([2,3]);inputs.item_map=np.arange(4);inputs.row_map=np.arange(4)
        inputs.history_available_at=np.array([9,10]);inputs.events=pd.DataFrame({'time_ms':[10]})
        inputs.tag_array=np.ones((4,1),dtype=int);inputs.ids=np.array([2]);inputs.rows=np.array([2])
        inputs.users=np.array([1]);inputs.tabs=np.array([1]);inputs.numeric=np.zeros((1,4));inputs.y=np.array([1])
        batch=inputs.batch(np.array([0]),torch.device('cpu'))
        self.assertEqual(batch['mask'].tolist(),[[True,False]])
        self.assertEqual(batch['history'].tolist(),[[2,0]])
        inputs.history_available_at=None
        self.assertEqual(inputs.batch(np.array([0]),torch.device('cpu'))['mask'].tolist(),[[True,True]])
