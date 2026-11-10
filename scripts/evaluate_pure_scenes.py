"""Development-only scene consistency diagnostics with the existing Pure model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score, roc_auc_score

from gamefeed.config import load_config
from gamefeed.datasets import ExposureCollator
from gamefeed.demo import _load_ranker
from gamefeed.features import FeatureSpace
from gamefeed.final_evaluation import blend_rank_order, observed_ndcg, paired_user_interval, recall
from gamefeed.training import _ranker_forward
from gamefeed.utils import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/optimization_v2/pure_scene'
KEYS = ['user_id', 'date', 'tab']


def predict(model, collator, records):
    values = []
    with torch.inference_mode():
        for start in range(0, len(records), 512):
            batch = collator(records[start:start + 512])
            values.extend(torch.sigmoid(_ranker_forward(model, batch)['long_view']).tolist())
    scores = np.asarray(values)
    assert len(scores) == len(records) and np.isfinite(scores).all()
    return scores


def exposure_summary(groups):
    return {name: dict(
        gauc=float(np.average(groups[name + '_auc'], weights=groups.rows)),
        ndcg10=float(groups[name + '_ndcg10'].mean()),
        valid_groups=len(groups), covered_rows=int(groups.rows.sum()),
        users=int(groups.user_id.nunique())) for name in ('actual_scene', 'fixed_tab0')}


def exposure_diagnostic(events, model, collator):
    records = events.to_dict('records')
    actual = predict(model, collator, records)
    fixed = predict(model, collator, [dict(r, tab=0) for r in records])
    np.testing.assert_array_equal(actual[events.tab.eq(0)], fixed[events.tab.eq(0)])
    rows = []
    for key, ix in events.groupby(KEYS, sort=False).indices.items():
        labels = events.long_view.to_numpy()[ix]
        if np.unique(labels).size < 2:
            continue
        row = dict(zip(KEYS, key), rows=len(ix), positives=int(labels.sum()))
        for name, scores in [('actual_scene', actual), ('fixed_tab0', fixed)]:
            row[name + '_auc'] = float(roc_auc_score(labels, scores[ix]))
            row[name + '_ndcg10'] = float(ndcg_score(labels[None, :], scores[ix][None, :], k=10))
        rows.append(row)
    groups = pd.DataFrame(rows)
    assert len(groups) and groups.rows.ge(2).all()
    groups.to_parquet(OUT / 'exposure_groups.parquet', index=False)
    predictions = events[['user_id', 'date', 'tab', 'video_id', 'long_view']].copy()
    predictions['actual_scene_score'] = actual
    predictions['fixed_tab0_score'] = fixed
    predictions.to_parquet(OUT / 'exposure_predictions.parquet', index=False)
    paired = groups[['user_id', 'rows']].copy()
    paired['auc_delta'] = groups.actual_scene_auc - groups.fixed_tab0_auc
    paired['ndcg_delta'] = groups.actual_scene_ndcg10 - groups.fixed_tab0_ndcg10
    # Repeating a group by its exposure count reuses the same weighted-mean bootstrap.
    weighted = paired.loc[paired.index.repeat(paired.rows)].reset_index(drop=True)
    intervals = dict(gauc=paired_user_interval(weighted, 'auc_delta'),
                     ndcg10=paired_user_interval(paired, 'ndcg_delta'))
    result = dict(summary=exposure_summary(groups), actual_minus_tab0=intervals,
                  total_rows=len(events), total_groups=events.groupby(KEYS).ngroups,
                  group_size=dict(median=float(groups.rows.median()), maximum=int(groups.rows.max()),
                                  groups_at_least_10=int(groups.rows.ge(10).sum())),
                  by_tab={str(t): exposure_summary(f) for t, f in groups.groupby('tab')})
    result['interpretation'] = ('GAUC is weighted by exposures in user/date/tab groups with both labels. '
                                'NDCG@10 averages these groups and averages tied score orderings. '
                                'Repeated logged exposures remain separate observations.')
    return result


def candidate_summary(frame, catalog_size):
    return dict(queries=len(frame), users=int(frame.user_id.nunique()),
                recall10=float(frame.recall10.mean()), ndcg10=float(frame.ndcg10.mean()),
                recall100=float(frame.recall100.mean()),
                coverage=len({v for ids in frame.recommended_ids for v in ids}) / catalog_size)


def candidate_diagnostic(events, queries, routes, items, model, collator):
    assert not routes[['user_id', 'date']].duplicated().any()
    assert not queries[['user_id', 'date']].duplicated().any()
    grouped = events.groupby(KEYS, sort=True)
    scenes = grouped.size().rename('exposures').reset_index()
    positive = events[events.long_view.eq(1)].groupby(KEYS).video_id.agg(lambda s: sorted(set(s)))
    scenes = scenes.merge(positive.rename('scene_positives').reset_index(), on=KEYS, how='left', validate='one_to_one')
    scenes.scene_positives = scenes.scene_positives.map(lambda v: v if isinstance(v, list) else [])
    # A parent query is one of the existing 100-candidate user/date lists.
    scenes = scenes.merge(routes[['user_id', 'date', 'recommended_ids']], on=['user_id', 'date'], validate='many_to_one')
    scenes = scenes.merge(queries[['user_id', 'date', 'history_ids', 'history_len']], on=['user_id', 'date'], validate='many_to_one')
    assert not scenes[KEYS].duplicated().any()
    assert len(scenes[['user_id', 'date']].drop_duplicates()) == len(routes)
    assert scenes.recommended_ids.map(lambda ids: len(ids) == len(set(ids)) == 100).all()
    result_rows, score_rows = [], []
    for start in range(0, len(scenes), 32):
        chunk = list(scenes.iloc[start:start + 32].itertuples())
        records = [dict(user_id=q.user_id, video_id=v, tab=q.tab, history_ids=q.history_ids,
                        history_len=q.history_len, long_view=0, is_like=0)
                   for q in chunk for v in q.recommended_ids]
        scores = predict(model, collator, records).reshape(len(chunk), 100)
        for q, probabilities in zip(chunk, scores):
            ids = list(q.recommended_ids)
            score_rows.append(dict(user_id=q.user_id, date=q.date, tab=q.tab,
                                   candidate_ids=ids, long_probability=probabilities.tolist(),
                                   scene_positives=q.scene_positives, exposures=q.exposures))
            if not q.scene_positives:
                continue
            watch = [ids[i] for i in sorted(range(100), key=lambda i: (-probabilities[i], str(ids[i])))]
            blend = blend_rank_order(ids, probabilities)
            assert set(ids) == set(watch) == set(blend)
            for method, ordered in [('rrf', ids), ('watch', watch), ('blend', blend)]:
                result_rows.append(dict(user_id=q.user_id, date=q.date, tab=q.tab, method=method,
                    positive_count=len(q.scene_positives), recommended_ids=ordered[:10],
                    recall10=recall(ordered, q.scene_positives, 10),
                    ndcg10=observed_ndcg(ordered, q.scene_positives),
                    recall100=recall(ordered, q.scene_positives, 100)))
        if start % 1024 == 0:
            print(f'Development scenes scored: {min(start+32, len(scenes))}/{len(scenes)}', flush=True)
    pd.DataFrame(score_rows).to_parquet(OUT / 'candidate_scores.parquet', index=False)
    frame = pd.DataFrame(result_rows)
    frame.to_parquet(OUT / 'candidate_lists.parquet', index=False)
    baseline = frame[frame.method.eq('rrf')].reset_index(drop=True)
    intervals = {}
    for method in ('watch', 'blend'):
        other = frame[frame.method.eq(method)].reset_index(drop=True)
        assert baseline[KEYS].equals(other[KEYS])
        np.testing.assert_array_equal(baseline.recall100, other.recall100)
        paired = baseline[KEYS].copy()
        for metric in ('recall10', 'ndcg10'):
            paired[metric] = other[metric] - baseline[metric]
        intervals[method + '_minus_rrf'] = {m: paired_user_interval(paired, m) for m in ('recall10', 'ndcg10')}
    return dict(parent_queries=len(routes), observed_scenes=len(scenes),
                positive_scenes=len(baseline), zero_positive_scenes=int(scenes.scene_positives.map(len).eq(0).sum()),
                summary={m: candidate_summary(f, len(items)) for m, f in frame.groupby('method')},
                comparisons=intervals,
                by_tab={str(t): {m: candidate_summary(g, len(items)) for m, g in f.groupby('method')}
                        for t, f in frame.groupby('tab')},
                interpretation='One query per observed user/date/tab with positives in that scene. All 100 candidates share the same scene. Zero-positive scenes are counted but excluded from Recall/NDCG and coverage. Unknown candidate feedback stays unknown.')


def write_report(result):
    lines = ['# Pure开发集场景一致性诊断', '',
             '本次复用现有观看模型，未训练新模型。开发曝光按用户、日期、场景分组；100条候选沿用原召回列表。', '',
             '## 曝光组内排序', '',
             '| 输入场景 | 有效组 | 曝光数 | GAUC | NDCG@10 |', '| --- | ---: | ---: | ---: | ---: |']
    for method, row in result['exposure']['summary'].items():
        lines.append(f"| {method} | {row['valid_groups']} | {row['covered_rows']} | {row['gauc']:.5f} | {row['ndcg10']:.5f} |")
    lines += ['', 'GAUC按有效组曝光数加权；NDCG对正负样本均有的组等权平均，并列分数取平均排序效果。重复曝光保留为独立记录。',
              f"曝光有效组的样本量中位数为{result['exposure']['group_size']['median']:.0f}条，至少10条的组有{result['exposure']['group_size']['groups_at_least_10']}个；它的NDCG与下方100候选口径分别解释。", '']
    for metric, ci in result['exposure']['actual_minus_tab0'].items():
        lines.append(f"真实场景相对tab=0的{metric}变化：{ci['estimate']:+.6f}，用户配对95%区间[{ci['lower']:.6f}, {ci['upper']:.6f}]。")
    c = result['candidates']
    lines += ['', '## 场景一致的固定候选评价', '',
              f"原{c['parent_queries']}个用户日期查询拆分为{c['observed_scenes']}个实际出现的场景，其中{c['positive_scenes']}个有正例、{c['zero_positive_scenes']}个无正例。下表只统计有正例的场景。", '',
              '| 排序 | Recall@10 | NDCG@10 | Recall@100 | 内容覆盖率 |', '| --- | ---: | ---: | ---: | ---: |']
    for method, row in c['summary'].items():
        lines.append(f"| {method} | {row['recall10']:.3%} | {row['ndcg10']:.5f} | {row['recall100']:.3%} | {row['coverage']:.2%} |")
    for method, metrics in c['comparisons'].items():
        for metric, ci in metrics.items():
            lines.append(f"\n{method}的{metric}变化：{ci['estimate']:+.6f}，用户配对95%区间[{ci['lower']:.6f}, {ci['upper']:.6f}]。")
    lines += ['', '## 结论与复现', '', result['conclusion'], '',
              '本补充口径将正例限定到各自场景，查询数量和分母均与旧全场景评价不同，数值不直接对比。候选未曝光代表反馈未知。未读取测试集结果，服务仍使用原RRF排序。', '',
              '```powershell', '$env:PYTHONPATH = "$PWD\\src"', 'python scripts/evaluate_pure_scenes.py', '```', '',
              '原始分组、预测和候选列表保存在本目录Parquet文件，汇总、分场景结果和输入校验值见results.json。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / 'configs/release.yaml'); paths = config['resolved_paths']
    events = pd.read_parquet(paths.processed_dir / 'events.parquet', filters=[('split', '==', 'dev')])
    events = events.reset_index(drop=True)
    assert events.split.eq('dev').all()
    queries = pd.read_parquet(paths.processed_dir / 'recall_queries.parquet', filters=[('split', '==', 'dev')])
    routes = pd.read_parquet(paths.report_dir / 'selected_dev_recommendations.parquet')
    items = pd.read_parquet(paths.processed_dir / 'items_content.parquet')
    directory = paths.artifact_dir / 'single_long_view_ranker'
    features = FeatureSpace.load(directory / 'feature_space.json')
    model = _load_ranker(config, features, directory / 'model.pt')
    collator = ExposureCollator(features, items, int(config['data']['history_length']))
    result = dict(settings=dict(split='dev', cpu_threads=2, candidates=100, k=10,
                                rrf_weight=.75, watch_rank_weight=.25, rank_constant=60,
                                bootstrap_repeats=2000, bootstrap_seed=2026))
    result['exposure'] = exposure_diagnostic(events, model, collator)
    result['candidates'] = candidate_diagnostic(events, queries, routes, items, model, collator)
    blend = result['candidates']['comparisons']['blend_minus_rrf']
    gains = all(blend[m]['lower'] > 0 for m in ('recall10', 'ndcg10'))
    result['conclusion'] = ('场景一致后，融合的两个Top10指标配对区间下界均大于0，可进入后续独立验证；当前仅完成开发诊断。'
                            if gains else '场景一致后，融合尚未同时获得两个Top10指标区间下界均大于0的结果。当前证据支持保留RRF，停止在此诊断阶段。')
    inputs = [ROOT / 'configs/release.yaml', directory / 'feature_space.json', directory / 'model.pt',
              paths.report_dir / 'selected_dev_recommendations.parquet']
    result['input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    write_json(OUT / 'results.json', result)
    write_report(result)
    print(json.dumps({k: result[k] for k in ('conclusion',)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
