"""Text retrieval using TF-IDF or pretrained BGE plus averaged history vectors."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .baselines import evaluate_retrieval, reciprocal_rank_fusion
from .utils import write_json


def run_text(config, split='dev', semantic=False):
    paths = config['resolved_paths']
    items = pd.read_parquet(paths.processed_dir / 'items_content.parquet')
    events = pd.read_parquet(paths.processed_dir / 'events.parquet')
    queries = pd.read_parquet(paths.processed_dir / 'recall_queries.parquet')
    queries = queries[queries.split.eq(split)]
    train_ids = set(events.loc[events.split.eq('train'), 'video_id'])
    # Fit vocabulary/IDF only on content referenced during training.
    text = (items.text + ' ' + items.category_text).fillna('')
    if semantic:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-small-zh-v1.5', local_files_only=True, device='cpu')
        model.max_seq_length = 128
        vectors = model.encode(text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(paths.artifact_dir / 'content_vectors.npz', video_ids=items.video_id.to_numpy(), vectors=vectors)
    else:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=30000, min_df=2)
        vectorizer.fit(text[items.video_id.isin(train_ids)])
        vectors = vectorizer.transform(text)
    method = 'semantic' if semantic else 'text'
    ids = items.video_id.to_numpy()
    mapping = {v: i for i, v in enumerate(ids)}
    popular = pd.read_parquet(paths.report_dir / f'popularity_{split}_recommendations.parquet')
    fallback = {(r.user_id, r.date): list(r.recommended_ids) for r in popular.itertuples()}
    rows = []
    for q in queries.itertuples():
        history = [mapping[v] for v in q.history_ids if v in mapping]
        available = items.available_from.le(q.date).to_numpy()
        if history:
            user = normalize(np.asarray(vectors[history].mean(axis=0)).reshape(1, -1))
            scores = np.asarray(vectors @ user.T).ravel()
            scores[~available] = -np.inf
            indices = np.flatnonzero(available)
            order = indices[np.argsort(-scores[indices], kind='stable')[:100]]
            recommended = ids[order].tolist()
        else:
            recommended = fallback[(q.user_id, q.date)]
        rows.append(dict(user_id=q.user_id, date=q.date, positive_ids=list(q.positive_ids), recommended_ids=recommended))
    result = pd.DataFrame(rows)
    metrics = evaluate_retrieval(result, ids.tolist())
    result.to_parquet(paths.report_dir / f'{method}_{split}_recommendations.parquet', index=False)
    write_json(paths.report_dir / f'{method}_{split}.json', metrics)
    collaborative = pd.read_parquet(paths.report_dir / f'itemcf_{split}_recommendations.parquet')
    fused = reciprocal_rank_fusion({'itemcf': collaborative, 'text': result, 'popularity': popular},
                                  {'itemcf': .6, 'text': .2, 'popularity': .2})
    fused.to_parquet(paths.report_dir / f'{method}_fusion_{split}_recommendations.parquet', index=False)
    output = {'text': metrics, 'content_fusion': evaluate_retrieval(fused, ids.tolist()),
              'weights': {'itemcf': .6, 'text': .2, 'popularity': .2}}
    write_json(paths.report_dir / f'{method}_experiment_{split}.json', output)
    return output
