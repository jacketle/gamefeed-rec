"""Development-only audit of saved scores; never trains or reads test predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gamefeed.final_evaluation import recall, observed_ndcg, paired_user_interval


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'reports/release'
    scores = pd.read_parquet(folder / 'final_evaluation/dev_candidate_scores.parquet')
    recall = pd.read_parquet(folder / 'selected_dev_recommendations.parquet')
    assert scores[['user_id', 'date']].equals(recall[['user_id', 'date']])
    rows = []
    for query, prediction in zip(recall.itertuples(), scores.itertuples()):
        ids = list(query.recommended_ids)
        positives = set(query.positive_ids)
        watch = np.asarray(prediction.long_probability)
        like = np.asarray(prediction.like_probability)
        assert len(ids) == len(watch) == len(like)
        assert len(ids) == len(set(ids)) and positives
        order = sorted(range(len(ids)), key=lambda i: (-watch[i], str(ids[i])))
        ranks = np.empty(len(ids))
        ranks[order] = np.arange(1, len(ids) + 1)
        variants = {
            'rrf': -np.arange(len(ids)),
            'watch_only': watch,
            'mix_07': .7 * watch + .3 * like,
            'rank_blend_025': .75 / (60 + np.arange(1, len(ids) + 1)) + .25 / (60 + ranks),
        }
        for name, values in variants.items():
            top = sorted(range(len(ids)), key=lambda i: (-values[i], str(ids[i])))[:10]
            selected = [ids[i] for i in top]
            rows.append(dict(user_id=query.user_id, date=query.date, method=name,
                             recall10=recall(selected,positives,10),
                             ndcg10=observed_ndcg(selected,positives)))
    frame = pd.DataFrame(rows)
    summary = frame.groupby('method')[['recall10', 'ndcg10']].mean().to_dict('index')
    paired = frame.pivot(index=['user_id', 'date'], columns='method', values='recall10')
    delta = paired.rank_blend_025 - paired.rrf
    interval = paired_user_interval(delta.rename('delta').reset_index())
    result = dict(split='dev', queries=len(recall), methods=summary,
                  blend_minus_rrf_recall10=float(delta.mean()),
                  user_bootstrap_ci95=[interval['lower'],interval['upper']],
                  note='Exploratory development comparison; interval excludes model/search uncertainty. Test untouched.')
    output = folder / 'optimization_audit'
    output.mkdir(exist_ok=True)
    (output / 'rank_blend_dev.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
