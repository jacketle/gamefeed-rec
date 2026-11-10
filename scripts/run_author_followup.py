"""Run the bounded author-frequency, initialization and explicit-match comparisons."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from gamefeed.config import load_config
from gamefeed.sequence_model import SequenceRanker
from gamefeed.sequence_training import Inputs, evaluate, metrics
from gamefeed.sequence_evaluation import paired_interval
from gamefeed.utils import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/optimization_v2/sequence'
ART = ROOT / 'artifacts/kuairand_1k/author_v2'
KEYS = ['user_id', 'video_id', 'tab', 'time_ms', 'long_view']
VARIANTS = {
    'baseline': {},
    'full_author': {'use_author': True, 'author_min_count': 1, 'author_init_std': 1.0},
    'high_author': {'use_author': True, 'author_min_count': 101, 'author_init_std': 1.0},
    'high_author_small': {'use_author': True, 'author_min_count': 101, 'author_init_std': 0.0001},
    'history_matches': {'use_history_matches': True},
}


def user_scene_metrics(frame, scores):
    conditional = frame.copy()
    conditional['user_id'] = pd.factorize(pd.MultiIndex.from_frame(frame[['user_id', 'tab']]))[0]
    return metrics(conditional, scores)


def predictions(name, seed, split='dev'):
    return pd.read_parquet(ART/name/f'B0_seed{seed}'/f'{split}_predictions.parquet')


def comparison(a, b):
    assert a[KEYS].equals(b[KEYS]), 'Prediction identity differs'
    return dict(baseline=metrics(a, a.score.to_numpy()), candidate=metrics(b, b.score.to_numpy()),
                difference=paired_interval(a, [a.score.to_numpy()], [b.score.to_numpy()]),
                user_scene_baseline=user_scene_metrics(a, a.score.to_numpy()),
                user_scene_candidate=user_scene_metrics(b, b.score.to_numpy()))


def main():
    os.chdir(ROOT)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    OUT.mkdir(parents=True, exist_ok=True)
    (ART/'configs').mkdir(parents=True, exist_ok=True)
    common = yaml.safe_load((ROOT/'configs/kuairand_1k_followup.yaml').read_text(encoding='utf-8'))
    configs = {}
    for name, flags in VARIANTS.items():
        cfg = json.loads(json.dumps(common))
        cfg['model'].update(flags)
        cfg['paths'] = {k: str(ROOT/v) for k, v in cfg['paths'].items()}
        cfg['paths']['artifact_dir'] = str(ART/name)
        cfg['paths']['report_dir'] = str(OUT/name)
        path = ART/'configs'/f'{name}.yaml'
        if path.exists():
            assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'Changed settings for {name}'
        else:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        configs[name] = path
    result = dict(settings=dict(variants=VARIANTS, screening_seed=2026, confirm_seeds=[2027, 2028],
                               min_gain=0.0005, bootstrap_repeats=2000,
                               test_policy='Only evaluate the development-selected variant after paired seed confirmation'),
                  dev_variants={})

    def save():
        write_json(OUT/'comparison.json', result)

    def status(phase, **extra):
        value = dict(phase=phase, pid=os.getpid(), **extra)
        write_json(ART/'run_status.json', value)
        print(json.dumps(value), flush=True)

    def train(name, seed):
        report_path = OUT/name/'experiment_results.json'
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            completed = report.get('training', {}).get(f'B0_seed{seed}', {})
            if (completed.get('status') == 'development_comparison'
                    and (ART/name/f'B0_seed{seed}/best.pt').is_file()
                    and (ART/name/f'B0_seed{seed}/dev_predictions.parquet').is_file()):
                status('reuse_completed', variant=name, seed=seed)
                return
        (ART/name).mkdir(parents=True, exist_ok=True)
        status('training', variant=name, seed=seed)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT/'src')
        env['PYTHONUNBUFFERED'] = '1'
        with (ART/name/'run.log').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-m', 'gamefeed.sequence_training', '--config', str(configs[name]),
                            '--modes', 'B0', '--seed', str(seed)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def group_results(name, seed, split):
        a, b = predictions('baseline', seed, split), predictions(name, seed, split)
        assert a[KEYS].equals(b[KEYS])
        groups = {}
        for tab, frame in a.groupby('tab'):
            ix = frame.index.to_numpy()
            target = frame.reset_index(drop=True)
            groups[f'tab_{tab}'] = dict(rows=len(ix), baseline=metrics(target, a.score.to_numpy()[ix]),
                                       candidate=metrics(target, b.score.to_numpy()[ix]))
        checkpoint = torch.load(ART/'baseline'/f'B0_seed{seed}/best.pt', map_location='cpu', weights_only=False)
        with np.load(ART/'baseline/sequence_feature_maps.npz') as maps:
            known = maps['known_item_ids']
        positions = np.searchsorted(known, a.video_id.to_numpy())
        valid = positions < len(known)
        valid &= known[np.minimum(positions, len(known)-1)] == a.video_id.to_numpy()
        mapped = np.where(valid, positions+2, 1)
        trained = checkpoint['state_dict']['trained_items'].numpy()[mapped] & valid
        for label, mask in [('trained_video', trained), ('untrained_video', ~trained)]:
            if mask.any():
                target = a.loc[mask].reset_index(drop=True)
                groups[label] = dict(rows=int(mask.sum()), baseline=metrics(target, a.score.to_numpy()[mask]),
                                     candidate=metrics(target, b.score.to_numpy()[mask]))
        return groups

    try:
        save()
        for name in VARIANTS:
            train(name, 2026)
            if name != 'baseline':
                result['dev_variants'][name] = comparison(predictions('baseline', 2026), predictions(name, 2026))
                result['dev_variants'][name]['groups'] = group_results(name, 2026, 'dev')
                save()
                status('screened', variant=name, difference=result['dev_variants'][name]['difference'])
        # These comparisons isolate frequency filtering and initialization, in addition to comparison against B0.
        result['stage_comparisons'] = {
            'frequency_high_minus_full': comparison(predictions('full_author', 2026), predictions('high_author', 2026)),
            'small_init_minus_default': comparison(predictions('high_author', 2026), predictions('high_author_small', 2026)),
        }
        eligible = [(name, row['difference']['mean_difference']) for name, row in result['dev_variants'].items()
                    if row['difference']['mean_difference'] >= .0005 and row['difference']['ci95'][0] > 0]
        result['selected_variant'] = max(eligible, key=lambda pair: pair[1])[0] if eligible else None
        result['evaluate_test'] = False
        save()
        winner = result['selected_variant']
        if winner:
            for seed in (2027, 2028):
                train('baseline', seed)
                train(winner, seed)
            pairs = [comparison(predictions('baseline', s), predictions(winner, s)) for s in (2026, 2027, 2028)]
            base = predictions('baseline', 2026)
            aa, bb = [], []
            for seed in (2026, 2027, 2028):
                a, b = predictions('baseline', seed), predictions(winner, seed)
                assert a[KEYS].equals(base[KEYS]) and b[KEYS].equals(base[KEYS])
                aa.append(a.score.to_numpy()); bb.append(b.score.to_numpy())
            interval = paired_interval(base, aa, bb)
            result['confirmation'] = dict(seeds=[2026, 2027, 2028], per_seed=pairs, difference=interval)
            result['evaluate_test'] = bool(all(r['difference']['mean_difference']>0 for r in pairs)
                                           and interval['mean_difference']>=.0005 and interval['ci95'][0]>0)
            save()
        if result['evaluate_test']:
            status('test_evaluation', variant=winner)
            torch.set_num_threads(common['model']['threads'])
            for name in ('baseline', winner):
                cfg = load_config(configs[name])
                inputs = Inputs(cfg, include_test=True)
                ix = np.flatnonzero(inputs.events.split.eq('test') & inputs.events.is_rand.eq(0))
                for seed in (2026, 2027, 2028):
                    ck = torch.load(ART/name/f'B0_seed{seed}/best.pt', map_location='cpu', weights_only=False)
                    assert ck['spec'] == inputs.spec and ck['feature_info'] == inputs.feature_info
                    model = SequenceRanker(**ck['spec'], mode='B0', **ck.get('model_kwargs', {})).to('cuda')
                    model.load_state_dict(ck['state_dict'])
                    frame = inputs.events.iloc[ix][KEYS].copy()
                    frame['score'] = evaluate(model, inputs, ix, common['model']['eval_batch_size'], torch.device('cuda'))
                    frame.to_parquet(ART/name/f'B0_seed{seed}/test_predictions.parquet', index=False)
                    del model, ck
                inputs.pool._mmap.close()
                del inputs
                gc.collect(); torch.cuda.empty_cache()
            pairs = [comparison(predictions('baseline', s, 'test'), predictions(winner, s, 'test')) for s in (2026, 2027, 2028)]
            frame = predictions('baseline', 2026, 'test')
            aa, bb = [], []
            for seed in (2026, 2027, 2028):
                a, b = predictions('baseline', seed, 'test'), predictions(winner, seed, 'test')
                assert a[KEYS].equals(frame[KEYS]) and b[KEYS].equals(frame[KEYS])
                aa.append(a.score.to_numpy()); bb.append(b.score.to_numpy())
            result['test'] = dict(per_seed=pairs, difference=paired_interval(frame, aa, bb),
                                  groups=group_results(winner, 2026, 'test'))
        test = result.get('test', {}).get('difference', {})
        result['decision'] = ('prefer_'+winner if test.get('mean_difference', 0)>0 and test.get('ci95', [0])[0]>0
                              else 'retain_B0')
        save(); status('complete', decision=result['decision'])
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
