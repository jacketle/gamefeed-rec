"""Smoke check a running local demo and record warm scoring timings."""
import json
import platform
import statistics
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode
from gamefeed.demo import author_key, caption_key

BASE = 'http://127.0.0.1:8765'
ROOT = Path(__file__).resolve().parents[1]

def get(path):
    with urlopen(BASE + path, timeout=60) as response:
        return json.load(response)

options = get('/api/options')
assert len(options['queries']) == 7122
query = options['queries'][0]
params = dict(user=query['user_id'], date=query['date'], tab=options['tabs'][0], rank='1')
for method in ('selected','itemcf','popularity','semantic'):
    for rank in ('0','1'):
        result = get('/api/recommend?' + urlencode(dict(params, method=method, rank=rank)))
        rows = result['recommendations']
        assert result['candidate_count'] == 100 and 0 < len(rows) <= 10
        assert len({r['video_id'] for r in rows}) == len(rows)
        if rank == '1':
            assert max(Counter(author_key(r) for r in rows).values()) <= 2
            texts = [caption_key(r) for r in rows if caption_key(r)]
            assert len(texts) == len(set(texts))
        assert all(0 <= r['long_probability'] <= 1 and 0 <= r['like_probability'] <= 1 for r in rows)
        assert 'positive_ids' not in result
game = get('/api/recommend?' + urlencode(dict(params, game='1')))
assert all(row['is_game'] for row in game['recommendations'])
for tab in options['tabs']:
    get('/api/recommend?' + urlencode(dict(params, tab=tab)))
try:
    get('/api/recommend?user=invalid&date=2022-04-29')
    raise AssertionError('Invalid query must be rejected')
except HTTPError as error:
    assert error.code == 400
for _ in range(5):
    get('/api/recommend?' + urlencode(params))
timings = []
for q in options['queries'][:30]:
    result = get('/api/recommend?' + urlencode(dict(user=q['user_id'], date=q['date'], tab=options['tabs'][0], rank='1')))
    timings.append(result['scoring_and_rerank_ms'])
timings.sort()
output = dict(checked_methods=4, checked_queries=30, warmup_requests=5,
              scope='Cached recall lookup + two CPU rankers + metadata + author-cap and caption-dedup rerank; excludes model loading, recall generation and HTTP transfer',
              median_ms=statistics.median(timings), p95_ms=timings[28],
              p95_definition='nearest-rank ceil(0.95 * 30)', processor=platform.processor(),
              game_results=len(game['recommendations']))
(ROOT / 'reports/release/demo_benchmark.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(output, ensure_ascii=False, indent=2))
