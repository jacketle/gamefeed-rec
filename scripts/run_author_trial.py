"""B0 + candidate author: one development screen, then two paired seeds if useful."""
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

from gamefeed.config import load_config, ensure_output_dirs
from gamefeed.sequence_training import Inputs, evaluate, metrics
from gamefeed.sequence_model import SequenceRanker
from gamefeed.sequence_evaluation import paired_interval
from gamefeed.utils import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/kuairand_1k_author.yaml'
BASE = ROOT / 'artifacts/kuairand_1k/full1000'
KEYS = ['user_id','video_id','tab','time_ms','long_view']


def paired_predictions(root, seed, reference, filename='dev_predictions.parquet'):
    frame = pd.read_parquet(root / f'B0_seed{seed}' / filename)
    if not frame[KEYS].reset_index(drop=True).equals(reference[KEYS].reset_index(drop=True)):
        raise ValueError('Prediction records differ')
    return frame.score.to_numpy()


def main():
    os.chdir(ROOT)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    config = load_config(CONFIG); ensure_output_dirs(config)
    paths = config['resolved_paths']
    torch.set_num_threads(config['model']['threads'])
    device = torch.device(config['model']['device'])
    result = dict(settings=dict(minimum_gain=.0005, bootstrap_repeats=2000,
                 seeds=[2026,2027,2028], author_dim=16, max_epochs=5, learning_rate=.001,
                 batch_size=256, patience=2, training_label='long_view'))

    def save(): write_json(paths.report_dir/'comparison.json', result)
    def status(phase, **extra):
        value = dict(phase=phase,pid=os.getpid(),**extra)
        write_json(paths.artifact_dir/'run_status.json',value)
        print(json.dumps(value),flush=True)

    def train(seed):
        report_path = paths.report_dir/'experiment_results.json'
        report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        if (report.get('training',{}).get(f'B0_seed{seed}',{}).get('status') == 'development_comparison'
                and (paths.artifact_dir/f'B0_seed{seed}/best.pt').is_file()
                and (paths.artifact_dir/f'B0_seed{seed}/dev_predictions.parquet').is_file()):
            return
        status('training',seed=seed)
        env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT/'src'); env['PYTHONUNBUFFERED'] = '1'
        with (paths.artifact_dir/'run.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','gamefeed.sequence_training','--config',str(CONFIG),
                            '--modes','B0','--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    try:
        status('baseline_check')
        inputs = Inputs(config)
        with np.load(paths.processed_dir/'sequence_feature_maps.npz') as old:
            for name in old.files:
                np.testing.assert_array_equal(old[name],inputs.feature_maps[name])
        ck = torch.load(BASE/'B0_seed2026/best.pt',map_location='cpu',weights_only=False)
        assert ck['spec'] == {k:v for k,v in inputs.spec.items() if k!='num_authors'}
        assert ck['feature_info'] == {k:v for k,v in inputs.feature_info.items() if k!='author_count'}
        frame = inputs.events.iloc[inputs.dev][KEYS].reset_index(drop=True)
        original = paired_predictions(BASE,2026,frame)
        sample = np.linspace(0,len(inputs.dev)-1,2048,dtype=int)
        model = SequenceRanker(**ck['spec'],mode='B0').to(device)
        model.load_state_dict(ck['state_dict'])
        actual = evaluate(model,inputs,inputs.dev[sample],256,device)
        np.testing.assert_allclose(actual,original[sample],atol=1e-6,rtol=1e-6)
        result['baseline_check'] = dict(rows=len(frame), checked_predictions=len(sample),
            max_absolute_difference=float(np.abs(actual-original[sample]).max()),
            author_coverage=float((inputs.authors[inputs.dev]>0).mean()))
        save()
        inputs.pool._mmap.close()
        del inputs,model,ck,original; gc.collect(); torch.cuda.empty_cache()
        train(2026)
        a = paired_predictions(BASE,2026,frame)
        b = paired_predictions(paths.artifact_dir,2026,frame)
        interval = paired_interval(frame,[a],[b])
        result['screening'] = dict(baseline=metrics(frame,a),author=metrics(frame,b),difference=interval)
        result['repeat_seeds'] = bool(interval['mean_difference']>=.0005 and interval['ci95'][0]>0)
        save(); status('screened',repeat_seeds=result['repeat_seeds'],gain=interval['mean_difference'])
        seeds = [2026]
        if result['repeat_seeds']:
            for seed in (2027,2028): train(seed)
            seeds = [2026,2027,2028]
        original = [paired_predictions(BASE,s,frame) for s in seeds]
        enhanced = [paired_predictions(paths.artifact_dir,s,frame) for s in seeds]
        result['development'] = dict(seeds=seeds,
            per_seed=[dict(seed=s,baseline=metrics(frame,a),author=metrics(frame,b))
                      for s,a,b in zip(seeds,original,enhanced)],
            difference=paired_interval(frame,original,enhanced))
        differences = [r['author']['gauc']-r['baseline']['gauc'] for r in result['development']['per_seed']]
        result['evaluate_test'] = bool(len(seeds)==3 and all(d>0 for d in differences)
                                      and result['development']['difference']['ci95'][0]>0)
        save()
        # Write the development choice before loading the test observations.
        status('evaluation',include_test=result['evaluate_test'])
        inputs = Inputs(config,include_test=result['evaluate_test'])
        checkpoint = torch.load(BASE/'B0_seed2026/best.pt',map_location='cpu',weights_only=False)
        trained = checkpoint['state_dict']['trained_items'].numpy()
        for split in (['dev','test'] if result['evaluate_test'] else ['dev']):
            ids = np.flatnonzero(inputs.events.split.eq(split)&inputs.events.is_rand.eq(0))
            target = inputs.events.iloc[ids][KEYS].reset_index(drop=True)
            baseline_name = 'dev_predictions.parquet' if split=='dev' else 'test_standard_predictions.parquet'
            old_scores = [paired_predictions(BASE,s,target,baseline_name) for s in seeds]
            new_scores = []
            for seed in seeds:
                if split=='dev':
                    pred = paired_predictions(paths.artifact_dir,seed,target)
                else:
                    ck = torch.load(paths.artifact_dir/f'B0_seed{seed}/best.pt',map_location='cpu',weights_only=False)
                    assert ck['spec']==inputs.spec and ck['feature_info']==inputs.feature_info
                    model = SequenceRanker(**ck['spec'],mode='B0').to(device)
                    model.load_state_dict(ck['state_dict'])
                    pred = evaluate(model,inputs,ids,256,device)
                    out = target.copy(); out['score'] = pred
                    out.to_parquet(paths.artifact_dir/f'B0_seed{seed}/test_standard_predictions.parquet',index=False)
                    del model,ck
                new_scores.append(pred)
            known_item = trained[inputs.ids[ids]] & (inputs.ids[ids]>1)
            known_author = inputs.authors[ids]>0
            masks = dict(all=np.ones(len(ids),bool),trained_video=known_item,untrained_video=~known_item,
                         known_author=known_author,unknown_author=~known_author)
            masks.update({f'tab_{t}':target.tab.to_numpy()==t for t in sorted(target.tab.unique())})
            groups = {}
            for name,mask in masks.items():
                if not mask.any(): continue
                subset = target.loc[mask].reset_index(drop=True)
                pairs = [dict(seed=s,baseline=metrics(subset,a[mask]),author=metrics(subset,b[mask]))
                         for s,a,b in zip(seeds,old_scores,new_scores)]
                groups[name] = dict(rows=int(mask.sum()),per_seed=pairs)
            result[split+'_groups'] = groups
            if split=='test':
                result['test'] = dict(per_seed=groups['all']['per_seed'],
                                     difference=paired_interval(target,old_scores,new_scores))
            save()
        result['decision'] = ('adopt_author' if result.get('test',{}).get('difference',{}).get('ci95',[0])[0]>0
                              and result['test']['difference']['mean_difference']>0 else 'keep_B0')
        save(); status('complete',decision=result['decision'])
    except Exception:
        status('failed',error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
