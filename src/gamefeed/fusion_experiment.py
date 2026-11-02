"""A bounded dev-only search: six fusion settings and an ItemCF baseline."""
import json
from time import perf_counter
import numpy as np
import pandas as pd
from .baselines import evaluate_retrieval, reciprocal_rank_fusion
from .content_analysis import positive_metrics
from .pipeline import run_popularity, run_itemcf
from .text_retrieval import run_text
from .utils import write_json
from .config import ensure_output_dirs


def quota_list(routes, semantic_quota, k=100):
    """Reserve semantic slots, then popularity slots, then fill by ItemCF."""
    result, seen = [], set()
    def take(values, limit):
        added = 0
        for item in values:
            if item not in seen and len(result) < k:
                result.append(item)
                seen.add(item)
                added += 1
                if added == limit:
                    break
    if semantic_quota:
        take(routes['semantic'], semantic_quota)
    take(routes['popularity'], 10)
    take(routes['itemcf'], k)
    for route in routes.values():
        if len(result) < k:
            take(route, k)
    # Quotas control membership; RRF defines the within-set order.
    score = {}
    for name, values in routes.items():
        weight = {'itemcf': .8, 'semantic': .2, 'popularity': .2}[name]
        for rank, item in enumerate(values, 1):
            score[item] = score.get(item, 0.) + weight / (60 + rank)
    return sorted(result, key=lambda item: (-score[item], str(item)))


def combine(frames, setting):
    if setting['kind'] == 'itemcf':
        result = frames['itemcf'].copy()
        result['recall_channels'] = [[['itemcf'] for _ in ids] for ids in result.recommended_ids]
        return result
    if setting['kind'] == 'rrf':
        return reciprocal_rank_fusion(frames, setting['weights'])
    base = frames['itemcf'][['user_id', 'date', 'positive_ids']].copy()
    lists, channels = [], []
    for index in range(len(base)):
        routes = {name: list(frame.iloc[index].recommended_ids) for name, frame in frames.items()}
        recommended = quota_list(routes, setting['semantic_quota'])
        lists.append(recommended)
        channels.append([[name for name, values in routes.items() if item in values] for item in recommended])
    base['recommended_ids'], base['recall_channels'] = lists, channels
    return base


def route_frames(config, split):
    paths = config['resolved_paths']
    timings = {}
    for name, fn in [('popularity', run_popularity), ('itemcf', run_itemcf)]:
        start = perf_counter()
        fn(config, split)
        timings[name + '_batch_seconds'] = perf_counter() - start
    start = perf_counter()
    run_text(config, split, semantic=True)
    timings['semantic_batch_seconds_including_encoding'] = perf_counter() - start
    frames = {name: pd.read_parquet(paths.report_dir / f'{name}_{split}_recommendations.parquet')
              .sort_values(['user_id', 'date']).reset_index(drop=True)
              for name in ('itemcf', 'semantic', 'popularity')}
    for frame in frames.values():
        if frame[['user_id', 'date']].duplicated().any():
            raise ValueError('Duplicate query keys')
        if not frame[['user_id', 'date']].equals(frames['itemcf'][['user_id', 'date']]):
            raise ValueError('Recall routes do not cover identical queries')
    write_json(paths.report_dir / f'route_timing_{split}.json', timings)
    return frames


def run_fusion_experiment(config):
    ensure_output_dirs(config)
    paths = config['resolved_paths']
    items = pd.read_parquet(paths.processed_dir / 'items_content.parquet')
    events = pd.read_parquet(paths.processed_dir / 'events.parquet')
    catalog = items.video_id.tolist()
    game = set(items.loc[items.is_game, 'video_id'])
    unseen = set(catalog) - set(events.loc[events.split.eq('train'), 'video_id'])
    settings = [{'name': 'itemcf', 'kind': 'itemcf'}]
    settings += [dict(name=f'rrf_{w}', kind='rrf', weights={'itemcf': .8, 'popularity': .2, 'semantic': w}) for w in (0., .2, .4)]
    settings += [dict(name=f'quota_{n}', kind='quota', semantic_quota=n) for n in (0, 10, 20)]
    frames = route_frames(config, 'dev')
    trials = []
    for setting in settings:
        start = perf_counter()
        result = combine(frames, setting)
        seconds = perf_counter() - start
        lengths = result.recommended_ids.map(lambda values: len(set(values)))
        if not lengths.eq(100).all():
            raise ValueError('All compared lists must contain 100 unique candidates')
        metrics = evaluate_retrieval(result, catalog)
        trials.append(dict(setting=setting, metrics=metrics, game=positive_metrics(result, game),
                           unseen=positive_metrics(result, unseen), fusion_batch_seconds=seconds))
    # Choose by macro Recall@100; tie-break with Recall@50. Never inspect test here.
    best = max(trials, key=lambda r: (r['metrics']['recall@100'], r['metrics']['recall@50']))
    write_json(paths.report_dir / 'fusion_selection.json', {'criterion': 'dev macro Recall@100, then Recall@50', 'selected': best['setting'], 'trials': trials})
    selected = combine(frames, best['setting'])
    selected.to_parquet(paths.report_dir / 'selected_dev_recommendations.parquet', index=False)
    write_json(paths.report_dir / 'selected_dev.json', best['metrics'])
    # Selection saved before running the final comparison.
    test_frames = route_frames(config, 'test')
    test = combine(test_frames, best['setting'])
    test.to_parquet(paths.report_dir / 'selected_test_recommendations.parquet', index=False)
    metrics = evaluate_retrieval(test, catalog)
    write_json(paths.report_dir / 'selected_test.json', dict(metrics=metrics, game=positive_metrics(test, game), unseen=positive_metrics(test, unseen)))
    return dict(selected=best['setting'], dev=best['metrics'], test=metrics)
