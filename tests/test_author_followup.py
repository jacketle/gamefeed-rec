"""Regression checks for paired author trials and request-time history features."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gamefeed.config import Paths
from gamefeed.sequence_model import SequenceRanker
from gamefeed.sequence_data import LOGS
from gamefeed import sequence_training as training


class AuthorFollowupTests(unittest.TestCase):
    def fixture(self, root):
        processed = root / 'processed'; processed.mkdir()
        rows = []
        def add(video, split, date, count=1, start=0, length=0):
            for _ in range(count):
                rows.append(dict(user_id=1, video_id=video, date=date,
                    time_ms=int(pd.Timestamp(str(date), tz='Asia/Shanghai').timestamp()*1000)+len(rows),
                    tab=0, long_view=len(rows)%2, is_rand=0, play_time_ms=1000,
                    source=LOGS[0][0], source_row=len(rows), split=split,
                    history_start=start, history_end=start+length, history_len=length))
        add(13, 'warmup', 20220410, 105)
        add(14, 'warmup', 20220410)
        add(15, 'warmup', 20220410)
        add(11, 'train', 20220423, 100)
        add(12, 'train', 20220423, 101)
        add(12, 'dev', 20220502, start=0, length=3)
        add(12, 'dev', 20220502, start=3, length=0)
        add(14, 'dev', 20220502, start=4, length=2)
        add(15, 'dev', 20220502, start=6, length=1)
        add(16, 'dev', 20220502, start=7, length=1)
        for date, frame in pd.DataFrame(rows).groupby('date'):
            frame.to_parquet(processed/f'events_{date}.parquet', index=False)
        pd.DataFrame(dict(video_id=[11,12,13,14,15,16],
            author_id=[101,102,103,-1,0,-1], video_duration=[2000]*6,
            tag=['1','2','3','2','2','999'])).to_parquet(processed/'items.parquet',index=False)
        # The extra 12 at position 3 must never become part of the first history.
        np.save(processed/'history_pool.npy',np.array([12,14,11,12,14,14,15,16],dtype=np.int32))
        return dict(seed=2026, resolved_paths=Paths(root,root,processed,root/'models',root/'reports'),
            data=dict(history_length=4,train_end=20220501,dev_end=20220505),
            model=dict(device='cpu',threads=1,batch_size=26,learning_rate=.001,
                       epochs=5,patience=2,paired_initialization=True,eval_every_fraction=.25))

    @staticmethod
    def batch():
        return dict(item=torch.tensor([2,3]), tags=torch.tensor([[2,0],[3,0]]),
            history=torch.zeros(2,4,dtype=torch.long),
            history_tags=torch.zeros(2,4,2,dtype=torch.long),
            mask=torch.zeros(2,4,dtype=torch.bool), user=torch.tensor([1,1]),
            tab=torch.tensor([1,1]),numeric=torch.zeros(2,4),
            author=torch.tensor([1,1]),matches=torch.tensor([[.5,1,.5],[0,0,0]]))

    def test_paired_extensions_preserve_baseline_layers_and_initial_predictions(self):
        torch.manual_seed(2026)
        baseline=SequenceRanker(2,8,8,2,'B0').eval()
        original=dict(baseline.named_parameters())
        for extension in ({},{'num_authors':3},{'num_authors':2},{'match_dim':3}):
            torch.manual_seed(2026)
            model=SequenceRanker(2,8,8,2,'B0',paired_initialization=True,**extension).eval()
            for name,p in original.items():
                q=dict(model.named_parameters())[name]
                if name=='output.0.weight':
                    self.assertTrue(torch.equal(p,q[:,:p.shape[1]]),extension)
                    self.assertEqual(q[:,p.shape[1]:].abs().sum().item(),0)
                else:
                    self.assertTrue(torch.equal(p,q),(extension,name))
            self.assertTrue(torch.equal(model(self.batch()),baseline(self.batch())),extension)

    def test_author_frequency_uses_formal_training_and_preserves_raw_id_initialization(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            cfg=self.fixture(Path(tmp)); cfg['model']['use_author']=True
            full=training.Inputs(cfg)
            try:
                self.assertEqual(full.feature_maps['authors'].tolist(),[101,102])
                cfg['model']['author_min_count']=101
                high=training.Inputs(cfg)
                try:
                    self.assertEqual(high.feature_maps['authors'].tolist(),[102])
                    self.assertEqual(int((high.authors[high.train]>0).sum()),101)
                    full_model=SequenceRanker(**full.spec,mode='B0',paired_initialization=True)
                    high_model=SequenceRanker(**high.spec,mode='B0',paired_initialization=True)
                    training.initialize_author_weights(full_model,full,2026,1e-4)
                    training.initialize_author_weights(high_model,high,2026,1e-4)
                    self.assertTrue(torch.equal(full_model.author.weight[2],high_model.author.weight[1]))
                    self.assertEqual(full_model.author.weight[0].abs().sum().item(),0)
                    self.assertGreater(full_model.author.weight[2].abs().sum().item(),0)
                finally:
                    high.pool._mmap.close()
            finally:
                full.pool._mmap.close()

    def test_history_matches_respect_mask_missing_values_and_known_tags(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            cfg=self.fixture(Path(tmp)); cfg['model']['use_history_matches']=True
            inputs=training.Inputs(cfg)
            try:
                batch=inputs.batch(inputs.dev,torch.device('cpu'))
                expected=torch.tensor([[1/3,1,2/3],[0,0,0],[0,0,1],[1,1,1],[0,0,0]])
                self.assertEqual(batch['matches'].dtype,torch.float32)
                torch.testing.assert_close(batch['matches'],expected)
                # Delayed feedback can remove a formerly valid same-author item.
                inputs.history_available_at=np.zeros(len(inputs.pool),dtype=np.int64)
                inputs.history_available_at[0]=inputs.events.iloc[inputs.dev[0]].time_ms
                delayed=inputs.batch(inputs.dev[:1],torch.device('cpu'))
                torch.testing.assert_close(delayed['matches'],torch.tensor([[0.,0.,.5]]))
            finally:
                inputs.pool._mmap.close()

    def test_quarter_checkpoint_is_reloadable_and_patience_counts_complete_epochs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            cfg=self.fixture(Path(tmp)); cfg['model']['use_history_matches']=True
            values=iter([.7,.9,.8,.7]+[.6]*20)
            original_metrics=training.metrics
            original_inputs=training.Inputs
            created=[]
            def inputs(config):
                value=original_inputs(config); created.append(value)
                return value
            def scores(model,inputs,ids,batch_size,device):
                model.eval()
                return np.full(len(ids),next(values),dtype=np.float32)
            def metrics(frame,prediction):
                result=original_metrics(frame,prediction)
                result['gauc']=float(prediction[0])
                return result
            with patch.object(training,'Inputs',side_effect=inputs), \
                 patch.object(training,'evaluate',side_effect=scores), \
                 patch.object(training,'metrics',side_effect=metrics), redirect_stdout(io.StringIO()):
                try:
                    training.run(cfg,['B0'])
                finally:
                    for value in created: value.pool._mmap.close()
            paths=cfg['resolved_paths']
            result=json.loads((paths.report_dir/'experiment_results.json').read_text())['training']['B0_seed2026']
            self.assertEqual(len(result['epochs']),3)
            self.assertAlmostEqual(result['best_dev']['gauc'],.9,places=6)
            checkpoint=torch.load(paths.artifact_dir/'B0_seed2026/best.pt',weights_only=False)
            self.assertAlmostEqual(checkpoint['fractional_epoch'],104/201)
            self.assertEqual(checkpoint['eval_step'],4)
            restored=SequenceRanker(**checkpoint['spec'],mode='B0',**checkpoint.get('model_kwargs',{})).eval()
            restored.load_state_dict(checkpoint['state_dict'])
            self.assertTrue(all(torch.isfinite(p).all() for p in restored.parameters()))
            predictions=pd.read_parquet(paths.artifact_dir/'B0_seed2026/dev_predictions.parquet')
            np.testing.assert_allclose(predictions.score,.9)
            np.testing.assert_allclose([e['fractional_epoch'] for e in result['evaluations'][:4]],
                                       [52/201,104/201,156/201,1.])


if __name__=='__main__':
    unittest.main()
