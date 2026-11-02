"""Stream official supplementary CSVs and retain only the Pure catalog."""
import pandas as pd
from .utils import write_json


def prepare_supplement(config):
    paths = config['resolved_paths']
    items = pd.read_parquet(paths.processed_dir / 'items.parquet')
    wanted = set(items.video_id.astype(str))
    report = {}
    for name in ('captions', 'categories'):
        source = paths.raw_dir / 'supplement' / f'kuairand_video_{name}.csv'
        kept, count = [], 0
        for chunk in pd.read_csv(source, dtype={'final_video_id': str}, chunksize=100000, encoding='utf-8'):
            count += len(chunk)
            selected = chunk[chunk.final_video_id.isin(wanted)]
            if len(selected):
                kept.append(selected.copy())
        data = pd.concat(kept, ignore_index=True).rename(columns={'final_video_id': 'video_id'})
        data['video_id'] = data.video_id.astype(items.video_id.dtype)
        if data.video_id.duplicated().any():
            raise ValueError(f'Duplicate Pure IDs in {source}')
        report[name] = {'source_rows': count, 'matched_items': len(data)}
        items = items.merge(data, on='video_id', how='left', validate='one_to_one')
    items['text'] = items[['caption', 'show_cover_text']].fillna('').agg(' '.join, axis=1).str.strip()
    category_columns = [c for c in items if c.endswith('_category_name')]
    items['category_text'] = items[category_columns].fillna('').replace('UNKNOWN', '').agg(' '.join, axis=1)
    items['is_game'] = items['category_text'].str.contains('游戏', regex=False)
    report['game_items'] = int(items.is_game.sum())
    report['nonempty_text_items'] = int(items.text.ne('').sum())
    items.to_parquet(paths.processed_dir / 'items_content.parquet', index=False)
    write_json(paths.report_dir / 'supplement_quality.json', report)
    return report
