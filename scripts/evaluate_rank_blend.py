"""Evaluate the fixed rank blend using verified candidate scores on dev and test."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gamefeed.config import load_config
from gamefeed.final_evaluation import (blend_rank_order, recall, observed_ndcg, paired_user_interval,
                                       history_group, load_routes, score_candidates)
from gamefeed.demo import author_key
from gamefeed.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    config = load_config(ROOT/'configs/release.yaml'); paths = config['resolved_paths']
    output = paths.report_dir/'rank_blend_v1'; output.mkdir(exist_ok=True)
    settings = dict(rrf_weight=.75,watch_rank_weight=.25,rank_constant=60,scene_tab=0,
                    candidates=100,output_k=10,bootstrap_repeats=2000,seed=2026)
    write_json(output/'settings.json',settings)
    items = pd.read_parquet(paths.processed_dir/'items_content.parquet')
    queries = pd.read_parquet(paths.processed_dir/'recall_queries.parquet')
    histories = queries.set_index(['user_id','date']).history_len.to_dict()
    games = set(items.loc[items.is_game,'video_id'])
    metadata = items.set_index('video_id').to_dict('index')
    authors = {v:author_key(dict(video_id=v,**m)) for v,m in metadata.items()}
    results = dict(settings=settings)
    for split in ('dev','test'):
        routes = load_routes(paths,split)['selected']
        old = pd.read_parquet(paths.report_dir/f'final_evaluation/{split}_candidate_scores.parquet')
        assert old[['user_id','date']].equals(routes[['user_id','date']])
        # Re-score all candidates once to verify the cached item-to-score alignment.
        scores = score_candidates(config,queries,items,routes,split,output)
        difference = 0.
        for column in ('long_probability','like_probability'):
            a = np.concatenate(scores[column].to_numpy()); b = np.concatenate(old[column].to_numpy())
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
            difference = max(difference,float(np.abs(a-b).max()))
        scores['candidate_ids'] = routes.recommended_ids
        scores.to_parquet(output/f'{split}_candidate_scores.parquet',index=False)
        rows = []
        for q,p in zip(routes.itertuples(),scores.itertuples()):
            ids = list(q.recommended_ids); assert len(ids)==100
            blended = blend_rank_order(ids,p.long_probability)
            assert set(ids)==set(blended)
            game_pos = set(q.positive_ids)&games
            for method,ordered in [('rrf',ids),('blend',blended)]:
                top = ordered[:10]; author_counts = Counter(authors[v] for v in top)
                rows.append(dict(user_id=q.user_id,date=q.date,method=method,recommended_ids=top,
                    recall10=recall(top,q.positive_ids,10),ndcg10=observed_ndcg(top,q.positive_ids),
                    recall100=recall(ordered,q.positive_ids,100),
                    game_recall10=recall(top,game_pos,10) if game_pos else np.nan,
                    history_group=history_group(histories[(q.user_id,q.date)]),
                    repeat_author=10-len(author_counts),max_author_count=max(author_counts.values())))
        frame = pd.DataFrame(rows)
        frame.to_parquet(output/f'{split}_lists.parquet',index=False)
        def summarize(f):
            return dict(queries=len(f),recall10=float(f.recall10.mean()),ndcg10=float(f.ndcg10.mean()),
                recall100=float(f.recall100.mean()),game_recall10=float(f.game_recall10.mean()),
                coverage=len({v for row in f.recommended_ids for v in row})/len(items),
                repeat_author_ratio=float(f.repeat_author.sum()/(10*len(f))),
                mean_largest_author_share=float(f.max_author_count.mean()/10))
        summary = {method:summarize(f) for method,f in frame.groupby('method')}
        base = frame[frame.method.eq('rrf')].reset_index(drop=True)
        new = frame[frame.method.eq('blend')].reset_index(drop=True)
        paired = base[['user_id','date','history_group']].copy()
        for metric in ('recall10','ndcg10','game_recall10'):
            paired[metric] = new[metric]-base[metric]
        users = paired.groupby('user_id').recall10.mean()
        comparison = dict(queries_improved=int(paired.recall10.gt(0).sum()),
            queries_worsened=int(paired.recall10.lt(0).sum()),queries_unchanged=int(paired.recall10.eq(0).sum()),
            users_improved=int(users.gt(0).sum()),users_worsened=int(users.lt(0).sum()),
            intervals={m:paired_user_interval(paired.dropna(subset=[m]),m)
                       for m in ('recall10','ndcg10','game_recall10')})
        results[split] = dict(summary=summary,comparison=comparison,cached_score_max_error=difference,
            by_date={str(date):{m:summarize(g) for m,g in f.groupby('method')} for date,f in frame.groupby('date')},
            by_history={str(h):{m:summarize(g) for m,g in f.groupby('method')} for h,f in frame.groupby('history_group')})
        paths_used = [paths.report_dir/f'selected_{split}_recommendations.parquet',
                      paths.report_dir/f'final_evaluation/{split}_candidate_scores.parquet']
        results[split]['input_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths_used}
        write_json(output/'results.json',results)
        print(json.dumps({split:summary},ensure_ascii=False),flush=True)
    test = results['test']['summary']
    both_improve = all(test['blend'][m]>test['rrf'][m] for m in ('recall10','ndcg10'))
    results['decision'] = 'optional_blend_keep_rrf_default' if both_improve else 'keep_rrf'
    results['interpretation'] = 'Offline comparison of previously inspected test data; fixed 100-candidate membership.'
    write_json(output/'results.json',results)


if __name__ == '__main__':
    main()
