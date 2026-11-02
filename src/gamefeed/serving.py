"""Local offline-replay demo. Cached recall lists, live model scoring."""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse
import pandas as pd
import torch
from .datasets import ExposureCollator
from .demo import _load_ranker, author_capped_rerank
from .features import FeatureSpace


class ReplayService:
    def __init__(self, config):
        self.config = config
        paths = config['resolved_paths']
        self.items = pd.read_parquet(paths.processed_dir / 'items_content.parquet').set_index('video_id', drop=False)
        queries = pd.read_parquet(paths.processed_dir / 'recall_queries.parquet')
        self.queries = queries[queries.split.eq('test')].set_index(['user_id', 'date'])
        self.routes = {name: pd.read_parquet(paths.report_dir / f'{name}_test_recommendations.parquet').set_index(['user_id','date'])
                       for name in ('popularity','itemcf','semantic','selected')}
        self.models, self.collators = {}, {}
        for task in ('long_view', 'is_like'):
            directory = paths.artifact_dir / f'single_{task}_ranker'
            features = FeatureSpace.load(directory / 'feature_space.json')
            self.models[task] = _load_ranker(config, features, directory / 'model.pt')
            self.collators[task] = ExposureCollator(features, self.items.reset_index(drop=True), int(config['data']['history_length']))
        events = pd.read_parquet(paths.processed_dir / 'events.parquet')
        self.tabs = sorted(int(v) for v in events.tab.dropna().unique())
        self.report = json.loads((paths.report_dir / 'release_summary.json').read_text(encoding='utf-8'))

    def metadata(self, item_id):
        row = self.items.loc[item_id]
        def clean(value):
            return '' if pd.isna(value) else str(value)
        return dict(video_id=str(item_id), author_id=clean(row.author_id),
                    caption=clean(row.caption) or clean(row.show_cover_text) or '该内容暂无文案',
                    dedup_text=clean(row.caption) or clean(row.show_cover_text),
                    category=clean(row.first_level_category_name), is_game=bool(row.is_game))

    def choices(self):
        # Pick examples by prior history only, without using test outcomes.
        games = set(self.items.loc[self.items.is_game, 'video_id'])
        rows = []
        for (user, date), q in self.queries.iterrows():
            count = sum(v in games for v in q.history_ids)
            rows.append(dict(user_id=str(user), date=str(date.date()), history_len=len(q.history_ids), game_history=count))
        rows.sort(key=lambda r: (-r['game_history'], r['date'], r['user_id']))
        return dict(queries=rows, tabs=self.tabs, summary=self.report)

    def recommend(self, user_id, date, method='selected', tab=0, rank=False, game_only=False):
        if method not in self.routes or tab not in self.tabs:
            raise ValueError('未知推荐方案或场景编号')
        key = (int(user_id), pd.Timestamp(date))
        if key not in self.queries.index:
            raise ValueError('该用户与日期没有可回放的查询')
        start = perf_counter()
        query = self.queries.loc[key]
        recall = self.routes[method].loc[key]
        ids = list(recall.recommended_ids)
        records = [dict(user_id=key[0], video_id=v, tab=tab, history_ids=query.history_ids,
                        history_len=len(query.history_ids), long_view=0, is_like=0) for v in ids]
        probabilities = {}
        with torch.no_grad():
            for task, model in self.models.items():
                batch = self.collators[task](records)
                logits = model(*(batch[k] for k in ('user_ids','item_ids','tag_ids','durations','tabs',
                    'history_item_ids','history_mask','history_len_scaled')))[task]
                probabilities[task] = torch.sigmoid(logits).tolist()
        scored = []
        channels = recall.get('recall_channels', [[method] for _ in ids])
        for position, (v, p_long, p_like) in enumerate(zip(ids, probabilities['long_view'], probabilities['is_like'])):
            row = self.metadata(v)
            row.update(long_probability=p_long, like_probability=p_like, score=.7*p_long+.3*p_like,
                       recall_rank=position+1, channels=list(channels[position]))
            scored.append(row)
        visible = [r for r in scored if r['is_game']] if game_only else scored
        result = author_capped_rerank(visible, 10, 2) if rank else visible[:10]
        return dict(user_id=str(user_id), date=date, method=method, tab=tab,
                    history=[self.metadata(v) for v in reversed(query.history_ids)], recommendations=result,
                    candidate_count=len(ids), displayed_count=len(result),
                    scoring_and_rerank_ms=1000*(perf_counter()-start),
                    note='离线回放：召回列表预计算，排序实时计算。0.7/0.3为演示权重；游戏筛选后的列表不代表总体评价。')


def serve(config, port=8765):
    service = ReplayService(config)
    assets = Path(__file__).parent / 'web'
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path.startswith('/api/'):
                try:
                    if url.path == '/api/options':
                        payload = service.choices()
                    elif url.path == '/api/recommend':
                        params = parse_qs(url.query)
                        get = lambda name, default='': params.get(name, [default])[0]
                        payload = service.recommend(get('user'), get('date'), get('method','selected'),
                            int(get('tab','0')), get('rank','0')=='1', get('game','0')=='1')
                    else:
                        raise ValueError('未知接口')
                    code = 200
                except (ValueError, KeyError) as exc:
                    payload, code = {'error': str(exc)}, 400
                content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif url.path in ('/', '/index.html', '/app.js', '/style.css'):
                super().do_GET()
            else:
                self.send_error(404)
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(assets)))
    print(f'GameFeed http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
