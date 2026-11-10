import unittest
import pandas as pd
import tempfile
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from gamefeed.config import Paths

from gamefeed.sequence_data import event_history, prepare, LOGS, COLS


class SequenceHistoryTests(unittest.TestCase):
    def test_all_users_includes_user_first_seen_after_warmup(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            self.assertEqual(Path(tmp).resolve().parent,Path(__file__).resolve().parent)
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            def row(user, date, random):
                ts=pd.Timestamp(str(date),tz='Asia/Shanghai')
                return [user,10+user,date,int(ts.timestamp()*1000),0,1,0,random,1000,2000]
            for (name,random),rows in zip(LOGS,[[row(1,20220410,0)],[row(2,20220423,0)],[row(2,20220506,1)]]):
                pd.DataFrame(rows,columns=COLS).to_csv(raw/name,index=False)
            pd.DataFrame(dict(video_id=[11,12],author_id=[1,2],video_duration=[2000,2000],tag=['1','2'],upload_dt=['20220401']*2)).to_csv(raw/'video_features_basic_1k.csv',index=False)
            cfg=dict(seed=2026,resolved_paths=Paths(root,raw,root/'processed',root/'models',root/'reports'),
                data=dict(chunksize=2,sample_users=0,history_length=20,train_start=20220422,train_end=20220501,dev_end=20220505,test_end=20220508))
            with redirect_stdout(io.StringIO()):result=prepare(cfg)
            self.assertEqual(result['sample_users'],2)
            self.assertEqual(json.loads((root/'processed/selected_users.json').read_text()),[1,2])
            train=pd.read_parquet(root/'processed/events_20220423.parquet')
            self.assertEqual(train.history_len.tolist(),[0])

    def test_warmup_only_user_has_zero_embedding_input(self):
        from gamefeed.sequence_training import Inputs
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp);self.assertEqual(root.resolve().parent,Path(__file__).resolve().parent)
            rows=pd.DataFrame([
                [1,11,20220410,'warmup'],[2,12,20220423,'train'],[1,11,20220502,'dev']],
                columns=['user_id','video_id','date','split'])
            rows['time_ms']=pd.to_datetime(rows.date.astype(str),utc=True).astype('int64')//10**6
            rows['tab']=0;rows['long_view']=1;rows['is_rand']=0;rows['play_time_ms']=1000
            rows['source']=LOGS[0][0];rows['source_row']=range(3)
            rows['history_start']=0;rows['history_end']=0;rows['history_len']=0
            for date,g in rows.groupby('date'):g.to_parquet(root/f'events_{date}.parquet',index=False)
            pd.DataFrame(dict(video_id=[11,12],video_duration=[2000,2000],tag=['1','2'],author_id=[101,102])).to_parquet(root/'items.parquet',index=False)
            import numpy as np
            np.save(root/'history_pool.npy',np.array([],dtype='int32'))
            cfg=dict(resolved_paths=Paths(root,root,root,root,root),data=dict(history_length=20,dev_end=20220505,train_end=20220501))
            x=Inputs(cfg)
            self.assertEqual(x.users[x.dev].tolist(),[0])
            self.assertGreater(x.users[x.train][0],0)
            x.pool._mmap.close()
            self.assertFalse((root/'sequence_feature_maps.npz').exists())
            cfg['model'] = {'use_author':True}
            x = Inputs(cfg)
            self.assertEqual(x.feature_maps['authors'].tolist(),[102])
            self.assertEqual(x.authors[x.dev].tolist(),[0])
            self.assertEqual(x.authors[x.train].tolist(),[1])
            x.pool._mmap.close()
    def test_strict_time_random_exclusion_and_user_boundaries(self):
        e = pd.DataFrame([
            [0, 10, 1, 1, 0], [0, 11, 1, 1, 0],
            [0, 12, 2, 1, 1], [0, 13, 3, 1, 0],
            [0, 14, 4, 0, 0], [1, 20, 5, 0, 0],
        ], columns=['user_id','video_id','time_ms','long_view','is_rand'])
        e['source'] = 'test'; e['source_row'] = range(len(e))
        rows, pool, times = event_history(e, 2)
        histories = [pool[r.history_start:r.history_end].tolist() for r in rows.itertuples()]
        self.assertEqual(histories, [[], [], [10,11], [10,11], [11,13], []])
        for r in rows.itertuples():
            self.assertTrue((times[r.history_start:r.history_end] < r.time_ms).all())

    def test_no_positive_history(self):
        e = pd.DataFrame([dict(user_id=0,video_id=1,time_ms=1,long_view=0,is_rand=0,source='x',source_row=0)])
        rows, pool, _ = event_history(e, 20)
        self.assertEqual(len(pool),0)
        self.assertEqual(rows.history_len.tolist(),[0])

    def test_input_order_does_not_change_history(self):
        e = pd.DataFrame([
            dict(user_id=0,video_id=3,time_ms=3,long_view=0,is_rand=0,source='x',source_row=2),
            dict(user_id=0,video_id=1,time_ms=1,long_view=1,is_rand=0,source='x',source_row=0),
            dict(user_id=0,video_id=2,time_ms=2,long_view=1,is_rand=0,source='x',source_row=1),
        ])
        rows, pool, _ = event_history(e, 1)
        self.assertEqual([pool[r.history_start:r.history_end].tolist() for r in rows.itertuples()], [[],[1],[2]])
