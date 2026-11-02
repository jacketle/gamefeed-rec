# -*- coding: utf-8 -*-
"""Check cached BGE embeddings and controlled dev-only retrieval alternatives."""
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/release/semantic_audit'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
items = pd.read_parquet(ROOT / 'data/processed/kuairand/items_content.parquet')
queries = pd.read_parquet(ROOT / 'data/processed/kuairand/recall_queries.parquet')
queries = queries[queries.split.eq('dev')].reset_index(drop=True)
saved = pd.read_parquet(ROOT / 'reports/release/semantic_dev_recommendations.parquet')
fallback = pd.read_parquet(ROOT / 'reports/release/popularity_dev_recommendations.parquet')
fallback = {(r.user_id, r.date): list(r.recommended_ids) for r in fallback.itertuples()}
saved = {(r.user_id, r.date): list(r.recommended_ids) for r in saved.itertuples()}
cache = np.load(ROOT / 'artifacts/release/content_vectors.npz')
ids = items.video_id.to_numpy()
assert np.array_equal(ids, cache['video_ids']), 'Cached ID order differs'
vectors = cache['vectors']
mapping = {v: i for i, v in enumerate(ids)}
texts = (items.text + ' ' + items.category_text).fillna('').tolist()
model = SentenceTransformer('BAAI/bge-small-zh-v1.5', local_files_only=True, device='cpu')
official_length = model.max_seq_length
lengths = np.array([len(x) for x in model.tokenizer(texts, truncation=False)['input_ids']])
result = dict(model='BAAI/bge-small-zh-v1.5', official_max_length=official_length,
              project_max_length=128, items=len(ids), embedding_shape=list(vectors.shape),
              truncated_at_128=int((lengths > 128).sum()), truncated_at_512=int((lengths > 512).sum()),
              token_length_percentiles=np.percentile(lengths, [50, 95, 99, 100]).tolist(),
              finite_vectors=bool(np.isfinite(vectors).all()),
              max_norm_error=float(np.abs(np.linalg.norm(vectors, axis=1)-1).max()),
              pooling_config=model[1].get_config_dict())
model.max_seq_length = 128
sample = np.random.default_rng(2026).choice(len(ids), 32, replace=False)
encoded = model.encode([texts[i] for i in sample], normalize_embeddings=True, batch_size=16)
# Match SentenceTransformer's strip/lowercase preprocessing before comparing CLS.
inputs = model.tokenize([texts[i] for i in sample])
with torch.no_grad():
    raw = model[0].auto_model(**inputs).last_hidden_state[:, 0]
    official = torch.nn.functional.normalize(raw, dim=1).numpy()
result['official_cls_max_abs_error'] = float(np.abs(encoded-official).max())
result['cache_reencode_min_cosine'] = float(np.sum(encoded*vectors[sample], axis=1).min())
print(json.dumps(result, ensure_ascii=False), flush=True)
extended = vectors.copy()
long_indices = np.flatnonzero(lengths > 128)
model.max_seq_length = 512
if len(long_indices):
    extended[long_indices] = model.encode([texts[i] for i in long_indices], normalize_embeddings=True,
                                         batch_size=16, show_progress_bar=True)
rows = list(queries.itertuples())
histories = [[mapping[v] for v in q.history_ids if v in mapping] for q in rows]

def evaluate(embeddings, recent=False):
    values, lists, random_values = [], [], []
    user_vectors = np.zeros((len(rows), embeddings.shape[1]), dtype=np.float32)
    for i, hist in enumerate(histories):
        if hist:
            weights = 1/(1+.1*np.arange(len(hist)-1, -1, -1)) if recent else np.ones(len(hist))
            user_vectors[i] = np.average(embeddings[hist], axis=0, weights=weights)
    user_vectors /= np.maximum(np.linalg.norm(user_vectors, axis=1, keepdims=True), 1e-12)
    with threadpool_limits(limits=4):
        for start in range(0, len(rows), 128):
            scores = user_vectors[start:start+128] @ embeddings.T
            for j, q in enumerate(rows[start:start+128]):
                allowed = np.flatnonzero(items.available_from.le(q.date).to_numpy())
                order = allowed[np.argsort(-scores[j, allowed], kind='stable')[:100]]
                recs = ids[order].tolist() if histories[start+j] else fallback[(q.user_id, q.date)]
                positives = set(q.positive_ids)
                values.append(len(set(recs)&positives)/len(positives))
                random_values.append(min(100, len(allowed))/len(allowed)*len(positives & set(ids[allowed]))/len(positives))
                lists.append(recs)
    return {'recall@100': float(np.mean(values)), 'queries': len(rows),
            'uniform_random_expected_recall@100': float(np.mean(random_values))}, lists

result['comparisons'] = {}
for name, emb, recent in [('original_128_mean', vectors, False), ('length_512_mean', extended, False),
                           ('original_128_recent_weight', vectors, True)]:
    metrics, recs = evaluate(emb, recent)
    if name == 'original_128_mean':
        metrics['mean_top100_overlap_with_saved'] = float(np.mean([
            len(set(r)&set(saved[(q.user_id,q.date)]))/100 for r,q in zip(recs,rows)]))
    result['comparisons'][name] = metrics
    print(name, metrics, flush=True)

# Item-to-item examples help distinguish textual similarity from user preference.
chosen = items[items.is_game & items.text.str.contains('王者|原神|英雄', regex=True)].head(3).index
examples = []
for index in chosen:
    scores = vectors @ vectors[index]
    scores[index] = -np.inf
    nearest = np.argsort(-scores)[:5]
    examples.append({'video_id': int(ids[index]), 'text': texts[index], 'neighbors': [
        {'video_id': int(ids[j]), 'text': texts[j], 'cosine': float(scores[j])} for j in nearest]})
result['text_neighbors'] = examples
result['scope'] = 'Developer-set diagnostics on the development split only; original release outputs preserved.'
(OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved', OUT / 'audit.json', flush=True)
