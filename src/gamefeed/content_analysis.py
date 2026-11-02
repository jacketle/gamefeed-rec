"""Evaluate only target positives, leaving each method's candidate list intact."""
import pandas as pd
from .utils import write_json


def positive_metrics(results, target_ids, k=100):
    total, hits, recalls = 0, 0, []
    for row in results.itertuples():
        positives = set(row.positive_ids) & target_ids
        if not positives:
            continue
        matched = len(positives & set(row.recommended_ids[:k]))
        total += len(positives)
        hits += matched
        recalls.append(matched / len(positives))
    return dict(queries=len(recalls), positives=total, hits=hits,
                micro_recall=hits / total if total else None,
                macro_recall=sum(recalls) / len(recalls) if recalls else None)


def run_content_analysis(config, split='dev'):
    paths = config['resolved_paths']
    items = pd.read_parquet(paths.processed_dir / 'items_content.parquet')
    events = pd.read_parquet(paths.processed_dir / 'events.parquet')
    seen = set(events.loc[events.split.eq('train'), 'video_id'])
    targets = {'game': set(items.loc[items.is_game, 'video_id']),
               'unseen': set(items.video_id) - seen}
    output = {}
    for method in ('popularity', 'itemcf', 'fusion', 'text', 'semantic', 'semantic_fusion'):
        source = paths.report_dir / f'{method}_{split}_recommendations.parquet'
        if source.exists():
            results = pd.read_parquet(source)
            output[method] = {name: positive_metrics(results, ids) for name, ids in targets.items()}
    write_json(paths.report_dir / f'content_groups_{split}.json', output)
    return output
