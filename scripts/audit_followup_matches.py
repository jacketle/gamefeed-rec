"""Read-only train/development coverage of the three explicit history matches."""
from pathlib import Path
import time

import numpy as np
import torch

from gamefeed.config import load_config
from gamefeed.sequence_training import Inputs
from gamefeed.utils import write_json


ROOT = Path(__file__).resolve().parents[1]
NAMES = ['same_author_fraction', 'same_author_any', 'shared_tag_history_fraction']


def summarize(inputs, indices, batch_size=8192):
    nonzero = np.zeros(3, dtype=np.int64)
    positive_nonzero = np.zeros(3, dtype=np.float64)
    total = np.zeros(3, dtype=np.float64)
    cross = np.zeros((3, 3), dtype=np.float64)
    positives = zero_history = 0
    for start in range(0, len(indices), batch_size):
        batch = inputs.batch(indices[start:start+batch_size], torch.device('cpu'))
        values = batch['matches'].numpy().astype(np.float64)
        y = batch['label'].numpy()
        assert values.shape == (len(y), 3) and np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        empty = ~batch['mask'].numpy().any(axis=1)
        assert not values[empty].any(), 'Empty histories must yield zero matches'
        active = values > 0
        nonzero += active.sum(0)
        positive_nonzero += (active * y[:, None]).sum(0)
        total += values.sum(0)
        cross += values.T @ values
        positives += int(y.sum())
        zero_history += int(empty.sum())
    rows = len(indices)
    mean = total / rows
    covariance = cross / rows - np.outer(mean, mean)
    scale = np.sqrt(np.maximum(np.diag(covariance), 0))
    correlation = np.divide(covariance, np.outer(scale, scale),
                            out=np.full((3, 3), np.nan), where=np.outer(scale, scale)>0)
    features = {}
    for i, name in enumerate(NAMES):
        n = int(nonzero[i]); p = float(positive_nonzero[i])
        features[name] = dict(nonzero_rows=n, nonzero_coverage=n/rows, mean=float(mean[i]),
            long_view_rate_nonzero=p/n if n else None,
            long_view_rate_zero=(positives-p)/(rows-n) if rows>n else None)
    return dict(rows=rows, long_view_rate=positives/rows, zero_history_rows=zero_history,
        zero_history_matches_all_zero=True, features=features,
        correlation=[[float(v) if np.isfinite(v) else None for v in row] for row in correlation])


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    cfg = load_config(ROOT/'configs/kuairand_1k_followup.yaml')
    cfg['model'].update(use_history_matches=True, device='cpu', threads=2)
    inputs = Inputs(cfg)
    try:
        assert not inputs.events.split.eq('test').any()
        result = dict(feature_order=NAMES, batch_size=8192, device='cpu', threads=2,
            note='Unadjusted feature-label associations; model comparisons determine ranking gains.',
            history_rule=inputs.feature_info['history_matches'],
            train=summarize(inputs, inputs.train), dev=summarize(inputs, inputs.dev))
    finally:
        inputs.pool._mmap.close()
    result['elapsed_seconds'] = time.perf_counter()-started
    output = ROOT/'reports/optimization_v2/match_coverage.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(output)


if __name__ == '__main__':
    main()
