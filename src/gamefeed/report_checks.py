"""Small, reproducible denominator checks for the saved release report."""
import json

import pandas as pd

from .catalog import available_items
from .final_evaluation import load_routes, observed_ndcg, recall
from .utils import write_json


def gauc_coverage(events, task):
    """Use original user IDs, counting only users with both label values."""
    grouped = events.groupby('user_id')[task].agg(['size', 'sum'])
    valid = grouped['sum'].gt(0) & grouped['sum'].lt(grouped['size'])
    rows, users = len(events), len(grouped)
    return dict(rows=rows, users=users, positives=int(events[task].sum()),
                positive_rate=float(events[task].mean()),
                valid_users=int(valid.sum()), excluded_users=int((~valid).sum()),
                all_negative_users=int(grouped['sum'].eq(0).sum()),
                all_positive_users=int(grouped['sum'].eq(grouped['size']).sum()),
                covered_rows=int(grouped.loc[valid, 'size'].sum()),
                user_coverage=float(valid.mean()),
                row_coverage=float(grouped.loc[valid, 'size'].sum()/rows))


def random_recall_expectation(positives, available, k=100):
    """Uniform without-replacement selection; keep ALL positives in denominator."""
    positives, available = set(positives), set(available)
    if not positives:
        raise ValueError('Recall requires at least one positive')
    if k < 0:
        raise ValueError('k must be nonnegative')
    return (min(k, len(available))/len(available) * len(positives & available)/len(positives)
            if available else 0.0)


def dev_top10_comparison(paths):
    """Compare saved route orders on identical development queries and positives."""
    routes = load_routes(paths, 'dev')
    old = json.loads((paths.report_dir/'final_evaluation/results.json').read_text(encoding='utf-8'))
    results = []
    for method, label in [('popularity', '热门原顺序'), ('itemcf', 'ItemCF原顺序'),
                          ('selected', '融合候选的RRF顺序')]:
        frame = routes[method]
        assert all(len(ids) == len(set(ids)) for ids in frame.recommended_ids)
        values = pd.DataFrame([
            dict(recall10=recall(q.recommended_ids, q.positive_ids, 10),
                 ndcg10=observed_ndcg(q.recommended_ids, q.positive_ids, 10))
            for q in frame.itertuples()])
        results.append(dict(method=method, label=label, queries=len(frame),
                            recall10=float(values.recall10.mean()), ndcg10=float(values.ndcg10.mean())))
    saved = old['lists']['dev']['stages']
    for metric in ('recall10', 'ndcg10'):
        assert abs(results[-1][metric]-saved['recall'][metric]) < 1e-12
    assert saved['ranked']['queries'] == results[-1]['queries']
    results.append(dict(method='ranked', label='融合＋当前模型排序（tab=0，权重0.7/0.3）',
                        **{key: saved['ranked'][key] for key in ('queries', 'recall10', 'ndcg10')}))
    return results


def run(config):
    paths = config['resolved_paths']
    output = paths.report_dir / 'interpretation_checks'
    output.mkdir(exist_ok=True)
    events = pd.read_parquet(paths.processed_dir/'events.parquet')
    items = pd.read_parquet(paths.processed_dir/'items_content.parquet')
    queries = pd.read_parquet(paths.processed_dir/'recall_queries.parquet')
    singles = json.loads((paths.report_dir/'single_ranker_results.json').read_text(encoding='utf-8'))
    shared = json.loads((paths.report_dir/'shared_ranker_results.json').read_text(encoding='utf-8'))
    result = dict(gauc={}, random={}, catalog_items=len(items),
                  gauc_weight='有效用户的曝光记录数',
                  random_definition='每日可用内容中均匀无放回选100条；保留全部正例作分母，逐组等权平均')
    train_users = set(events.loc[events.split.eq('train'), 'user_id'])
    random_rows = []
    for split in ('dev', 'test'):
        subset = events[events.split.eq(split)]
        # Historical scores grouped encoded IDs; verify no UNK users were merged.
        assert set(subset.user_id) <= train_users, 'Recompute model GAUC by original user IDs'
        result['gauc'][split] = {}
        for task in ('long_view', 'is_like'):
            coverage = gauc_coverage(subset, task)
            for old in (singles[task][split], shared[split][task]):
                assert coverage['valid_users'] == old['valid_groups']
                assert coverage['covered_rows'] == old['covered_rows']
            coverage.update(single_auc=singles[task][split]['auc'],
                            single_gauc=singles[task][split]['gauc'],
                            shared_auc=shared[split][task]['auc'],
                            shared_gauc=shared[split][task]['gauc'], unknown_users=0)
            result['gauc'][split][task] = coverage
        rows = []
        for date, batch in queries[queries.split.eq(split)].groupby('date'):
            available = set(available_items(items, date).video_id)
            for query in batch.itertuples():
                positives = set(query.positive_ids)
                rows.append(dict(split=split, user_id=query.user_id, date=date,
                                 available_items=len(available), positives=len(positives),
                                 unavailable_positives=len(positives-available),
                                 expected_recall=random_recall_expectation(positives, available)))
        frame = pd.DataFrame(rows)
        result['random'][split] = dict(queries=len(frame),
            min_available_items=int(frame.available_items.min()),
            max_available_items=int(frame.available_items.max()),
            positive_occurrences=int(frame.positives.sum()),
            unavailable_positive_occurrences=int(frame.unavailable_positives.sum()),
            queries_with_unavailable_positives=int(frame.unavailable_positives.gt(0).sum()),
            expected_recall_at_100=float(frame.expected_recall.mean()),
            fixed_full_catalog_reference=min(100, len(items))/len(items))
        random_rows.extend(rows)
    pd.DataFrame(random_rows).to_parquet(output/'random_per_query.parquet', index=False)
    audit = json.loads((paths.report_dir/'semantic_audit/audit.json').read_text(encoding='utf-8'))
    previous = audit['comparisons']['original_128_mean']['uniform_random_expected_recall@100']
    assert abs(previous-result['random']['dev']['expected_recall_at_100']) < 1e-12
    result['previous_semantic_random_check_matches'] = True
    result['dev_top10'] = dev_top10_comparison(paths)
    write_json(output/'results.json', result)
    lines = ['# GAUC覆盖率与随机基线检查', '',
        '读取已有曝光、候选范围和保存指标进行核对；模型与召回结果保持原样。', '',
        '## GAUC分母', '',
        '同一用户同时有正例和负例才能计算用户内AUC；GAUC按这些用户的曝光记录数加权。',
        '用户覆盖率＝有效用户数÷该集合全部用户数；曝光覆盖率＝有效用户的曝光记录数÷该集合全部曝光记录数。', '',
        '| 集合／任务 | 正例／曝光 | 正例率 | 有效／全部用户 | 用户覆盖率 | 有效／全部曝光 | 曝光覆盖率 | 独立AUC | 独立GAUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split, tasks in result['gauc'].items():
        for task, r in tasks.items():
            label = '观看达标' if task == 'long_view' else '点赞'
            lines.append(f"| {split}／{label} | {r['positives']}/{r['rows']} | {r['positive_rate']:.2%} | {r['valid_users']}/{r['users']} | {r['user_coverage']:.2%} | {r['covered_rows']}/{r['rows']} | {r['row_coverage']:.2%} | {r['single_auc']:.4f} | {r['single_gauc']:.4f} |")
    like = result['gauc']['test']['is_like']
    lines += ['', f"测试点赞被排除的用户中，全负例{like['all_negative_users']}名、全正例{like['all_positive_users']}名。独立与共享模型使用同一有效人群。",
        '已用原始用户ID复核分组；开发和测试用户都在训练词表中，没有多个未知用户被合并成一组的问题。',
        '总体AUC会比较不同用户的曝光记录；GAUC只在同一用户内比较。点赞GAUC的人群覆盖较小，两者回答的问题不同。用户点赞倾向是可能的解释，当前统计没有单独验证这个原因。', '',
        '## 每日可用内容下的随机基线', '',
        '对一组评价样本：N为当天可用内容数，P为当天全部正例数，A为其中可用正例数，K为推荐条数。', '',
        '$$', r'E[\mathrm{Recall@K}]=\frac{\min(K,N)}{N}\times\frac{A}{P}', '$$', '',
        'N为0时期望记为0。每个可用正例被随机选中的概率是min(K,N)/N；不可用正例仍留在P中。最后对全部评价样本取平均。', '',
        '| 集合 | 评价样本数 | 每日可用内容数范围 | 正例出现次数 | 不可用正例出现次数 | 随机Recall@100期望 |',
        '|---|---:|---:|---:|---:|---:|']
    for split, r in result['random'].items():
        lines.append(f"| {split} | {r['queries']} | {r['min_available_items']}—{r['max_available_items']} | {r['positive_occurrences']} | {r['unavailable_positive_occurrences']} | {r['expected_recall_at_100']:.4%} |")
    lines += ['', '正例出现次数按用户和日期分别统计，同一视频出现在不同组中会分别计数。',
        '本数据的全部内容在开发期开始前已达到项目的可用日期，因此逐日计算与100÷7583的结果相同。原BGE检查代码已按每日范围计算，本次补齐分母与不可用正例统计。', '',
        '## 开发集Top10基线对照', '',
        '读取已保存的推荐列表，对相同5410组用户与日期、相同观看达标正例计算指标。每组等权平均，NDCG理想排序使用全部正例，包括未进入候选的正例。', '',
        '| 方案 | 开发Recall@10 | 开发NDCG@10 |', '|---|---:|---:|']
    for r in result['dev_top10']:
        lines.append(f"| {r['label']} | {r['recall10']:.3%} | {r['ndcg10']:.5f} |")
    lines += ['', '融合先按热门10条、ItemCF补齐确定100条候选，再用RRF排列集合内内容。这里的融合顺序指RRF结果，实际并非热门10条直接排在最前。',
        '模型排序行沿用已有tab=0、权重0.7/0.3的结果；其正例为当天各场景汇总。本轮只补已有方案的开发集对照，不重新训练或调权重。', '',
        '复现：`python scripts/check_report_metrics.py`；随机逐组结果：`random_per_query.parquet`；汇总及Top10表：`results.json`。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
