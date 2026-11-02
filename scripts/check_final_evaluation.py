"""Cross-check attribution arithmetic and saved lists against live scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gamefeed.config import load_config
from gamefeed.serving import ReplayService

torch.set_num_threads(4)
root=Path('reports/release/final_evaluation')
result=json.loads((root/'results.json').read_text(encoding='utf-8'))
for split in ('dev','test'):
    f=pd.read_parquet(root/f'fusion_{split}_per_query.parquet')
    c=pd.read_csv(root/f'fusion_{split}_changed_hits.csv')
    assert np.isclose(f.delta.mean(),c.macro_contribution.sum())
    assert np.isclose(f.selected.mean()-f.itemcf.mean(),result['fusion'][split]['delta'])
    lists=pd.read_parquet(root/f'{split}_final_lists.parquet')
    final=lists[lists.stage.eq('author_and_caption')]
    assert final.repeat_caption.eq(0).all()
    assert final.excess_author.eq(0).all()
    assert lists.recommended_ids.map(lambda ids: len(ids)==len(set(ids))).all()
service=ReplayService(load_config('configs/release.yaml'))
lists=pd.read_parquet(root/'test_final_lists.parquet')
cases=lists[lists.stage.eq('author_and_caption')].sample(10,random_state=2026)
for q in cases.itertuples():
    live=service.recommend(q.user_id,str(q.date.date()),rank=True,tab=0)
    assert [int(r['video_id']) for r in live['recommendations']]==list(q.recommended_ids)
    original=service.recommend(q.user_id,str(q.date.date()),tab=0)
    expected=service.routes['selected'].loc[(q.user_id,q.date)].recommended_ids[:10]
    assert [int(r['video_id']) for r in original['recommendations']]==list(expected)
print('PASS: attribution sums, unique IDs, caption/author constraints, 10 live scoring cases, default recall order')
