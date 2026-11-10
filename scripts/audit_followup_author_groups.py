"""Compare completed seed-2026 development predictions by train-author support."""
import json

import numpy as np
import pandas as pd

from gamefeed.sequence_training import metrics
from gamefeed.utils import write_json
from run_author_followup import ROOT, OUT, ART, KEYS, VARIANTS


def main():
    for name in VARIANTS:
        report = json.loads((OUT/name/'experiment_results.json').read_text(encoding='utf-8'))
        if report.get('training', {}).get('B0_seed2026', {}).get('status') != 'development_comparison':
            raise ValueError(f'{name} seed 2026 has not completed')
        if not (ART/name/'B0_seed2026/dev_predictions.parquet').is_file():
            raise FileNotFoundError(f'{name} development predictions are missing')

    with np.load(ART/'full_author/sequence_feature_maps.npz') as maps:
        authors = maps['authors']
        counts = maps['author_train_counts']
        if len(authors) != len(counts) or len(np.unique(authors)) != len(authors) or (counts < 1).any():
            raise ValueError('Full-author frequency mapping is invalid')
        frequencies = pd.Series(counts, index=authors)
    baseline = pd.read_parquet(ART/'baseline/B0_seed2026/dev_predictions.parquet')
    items = pd.read_parquet(ROOT/'data/processed/kuairand_1k/full1000/items.parquet',
                            columns=['video_id', 'author_id']).set_index('video_id')
    if not items.index.is_unique or not baseline.video_id.isin(items.index).all():
        raise ValueError('Development item metadata is incomplete or duplicated')
    raw_authors = baseline.video_id.map(items.author_id)
    support = raw_authors.map(frequencies).fillna(0)
    masks = {'unknown': support.eq(0), '1': support.eq(1), '2-5': support.between(2, 5),
             '6-20': support.between(6, 20), '21-100': support.between(21, 100), '>100': support.gt(100)}
    groups = {name: dict(rows=int(mask.sum()), exposure_share=float(mask.mean()), variants={})
              for name, mask in masks.items()}
    for name in VARIANTS:
        frame = baseline if name == 'baseline' else pd.read_parquet(ART/name/'B0_seed2026/dev_predictions.parquet')
        if not frame[KEYS].reset_index(drop=True).equals(baseline[KEYS].reset_index(drop=True)):
            raise ValueError(f'{name} development record identity differs')
        for group, mask in masks.items():
            subset = frame.loc[mask].reset_index(drop=True)
            groups[group]['variants'][name] = (metrics(subset, subset.score.to_numpy())
                                                if len(subset) else dict(rows=0, gauc=None))
    result = dict(seed=2026, split='dev', rows=len(baseline),
                  frequency_source='full_author formal-training author counts; unknown includes missing authors',
                  grouping='Exposure-weighted user GAUC within each author-frequency group',
                  groups=groups)
    output = ROOT/'reports/optimization_v2/author_groups.json'
    write_json(output, result)
    print(output)


if __name__ == '__main__':
    main()
