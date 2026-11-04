"""Two development-only L2 trials; repeat the winner only after the preset gate.

Run with PYTHONPATH=src. Completed training runs are reused on restart.
Progress is written to artifacts/kuairand_1k/item_l2/run.log and run_status.json.
"""
from datetime import datetime, timezone
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

from gamefeed.config import load_config, ensure_output_dirs
from gamefeed.sequence_training import metrics
from gamefeed.sequence_evaluation import paired_interval

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/kuairand_1k/full1000'
CONFIG=ROOT/'configs/kuairand_1k_item_l2.yaml'


def history_coverage(config, reference):
    """Fixed bins from the original B1 seed 2026 checkpoint."""
    root=config['resolved_paths'].processed_dir
    cols=['user_id','video_id','tab','time_ms','long_view','split','is_rand','history_end','history_len']
    frames=[]
    for path in sorted(root.glob('events_*.parquet')):
        if config['data']['train_end'] < int(path.stem.split('_')[1]) <= config['data']['dev_end']:
            f=pd.read_parquet(path,columns=cols)
            frames.append(f.loc[f.split.eq('dev') & f.is_rand.eq(0)])
    ev=pd.concat(frames,ignore_index=True)
    for c in reference.columns:
        np.testing.assert_array_equal(ev[c],reference[c])
    ck=torch.load(BASE/'B1_seed2026/best.pt',map_location='cpu',weights_only=False)
    seen=ck['state_dict']['trained_items'].numpy().copy();del ck
    with np.load(root/'sequence_feature_maps.npz') as maps:
        known=maps['known_item_ids']
    pool=np.load(root/'history_pool.npy',mmap_mode='r')
    loc=np.searchsorted(known,pool)
    exists=(loc<len(known)) & (known[np.minimum(loc,len(known)-1)]==pool)
    pool_seen=exists & seen[np.where(exists,loc+2,1)]
    result=np.zeros(len(ev),np.float32)
    offsets=np.arange(config['data']['history_length'])[None,:]
    for start in range(0,len(ev),50000):
        g=ev.iloc[start:start+50000];lengths=g.history_len.to_numpy()[:,None]
        pos=g.history_end.to_numpy()[:,None]-lengths+offsets
        valid=(offsets<lengths) & pool_seen[np.minimum(pos,len(pool)-1)]
        result[start:start+len(g)]=valid.sum(1)/lengths[:,0].clip(min=1)
    return result


def main():
    os.chdir(ROOT)
    config=load_config(CONFIG);ensure_output_dirs(config)
    paths=config['resolved_paths']; experiment=config['experiment']
    report_path=paths.report_dir/'comparison.json'
    result=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {'screening':{}}
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['PYTHONUNBUFFERED']='1'
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    log_path=paths.artifact_dir/'run.log'
    def status(state,phase,**extra):
        payload=dict(state=state,phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        (paths.artifact_dir/'run_status.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(payload,flush=True)
    def save():
        report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def variant(strength):
        raw=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
        key=f'l2_{strength:g}'
        raw['paths']['artifact_dir']+=f'/{key}'
        raw['paths']['report_dir']+=f'/{key}'
        # Generated configs live under artifacts, so resolve paths explicitly.
        raw['paths']={name:str(ROOT/value) for name,value in raw['paths'].items()}
        raw['model']['item_l2']=strength
        path=paths.artifact_dir/f'{key}.yaml'
        path.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf-8')
        cfg=load_config(path);ensure_output_dirs(cfg)
        return path,cfg
    def train(strength,mode,seed):
        path,cfg=variant(strength)
        completed=cfg['resolved_paths'].report_dir/'experiment_results.json'
        previous=json.loads(completed.read_text(encoding='utf-8')) if completed.exists() else {}
        record=previous.get('training',{}).get(f'{mode}_seed{seed}',{})
        model_dir=cfg['resolved_paths'].artifact_dir/f'{mode}_seed{seed}'
        if (record.get('status')=='development_comparison' and record.get('item_l2')==strength
                and all((model_dir/name).is_file() for name in ('best.pt','dev_predictions.parquet'))):
            return cfg
        status('running',f'train_{mode}_seed{seed}_l2_{strength:g}')
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            print('\n'+datetime.now(timezone.utc).isoformat()+f' {mode} seed={seed} item_l2={strength:g}',file=log)
            subprocess.run([sys.executable,'-u','-m','gamefeed.sequence_training','--config',str(path),
                            '--modes',mode,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,check=True,cwd=ROOT)
        return cfg
    def prediction(root,mode,seed,reference):
        f=pd.read_parquet(root/f'{mode}_seed{seed}/dev_predictions.parquet').reset_index(drop=True)
        if not f.drop(columns='score').equals(reference):
            raise ValueError('Development record alignment failed')
        return f.score.to_numpy()
    try:
        status('running','prepare_comparison')
        reference=pd.read_parquet(BASE/'B1_seed2026/dev_predictions.parquet').drop(columns='score').reset_index(drop=True)
        original={m:prediction(BASE,m,2026,reference) for m in ('B0','B1')}
        coverage=history_coverage(config,reference)
        groups={'全部':np.ones(len(reference),bool),'历史ID覆盖0–25%':coverage<=.25,
                '历史ID覆盖25–75%':(coverage>.25)&(coverage<=.75),'历史ID覆盖75–100%':coverage>.75}
        result['baseline']={m:metrics(reference,p) for m,p in original.items()}
        for strength in experiment['item_l2_strengths']:
            cfg=train(strength,'B1',2026)
            status('running',f'compare_l2_{strength:g}')
            score=prediction(cfg['resolved_paths'].artifact_dir,'B1',2026,reference)
            entry={'strength':strength,'metrics':metrics(reference,score),
                   'difference':paired_interval(reference,[original['B1']],[score]),'groups':{}}
            for name,mask in groups.items():
                f=reference.loc[mask].reset_index(drop=True)
                entry['groups'][name]={'rows':len(f),'original_B1':metrics(f,original['B1'][mask]),
                                      'regularized_B1':metrics(f,score[mask])}
            result['screening'][str(strength)]=entry;save()
        best=max(result['screening'].values(),key=lambda r:r['metrics']['gauc'])
        delta=best['difference']
        result['selected_strength']=best['strength']
        result['worth_repeating']=bool(delta['mean_difference']>=experiment['minimum_gauc_gain'] and delta['ci95'][0]>0)
        save()
        if result['worth_repeating']:
            strength=best['strength']
            for seed in [2026,*experiment['followup_seeds']]:
                train(strength,'B0',seed)
                if seed!=2026:train(strength,'B1',seed)
            status('running','compare_three_seeds')
            _,cfg=variant(strength);root=cfg['resolved_paths'].artifact_dir
            seeds=[2026,*experiment['followup_seeds']]
            scores={m:[prediction(root,m,s,reference) for s in seeds] for m in ('B0','B1')}
            old={m:[prediction(BASE,m,s,reference) for s in seeds] for m in ('B0','B1')}
            result['confirmation']={'seeds':seeds,'per_seed':{m:[metrics(reference,p) for p in v] for m,v in scores.items()},
                'B1_regularized_minus_original':paired_interval(reference,old['B1'],scores['B1']),
                'B1_regularized_minus_B0_regularized':paired_interval(reference,scores['B0'],scores['B1']),
                'B1_regularized_minus_B0_original':paired_interval(reference,old['B0'],scores['B1'])}
            save()
        lines=['# 视频ID向量正则化实验','',
            '开发集对照；种子2026初筛。候选与有效历史中的ID合并去重，排除补位0和未知1，对其向量各坐标的平方取平均，乘以正则系数加到BCE上。B0只计算候选ID。',
            '这不是DIN频率修正MBA的复现。保留Adam、学习率、历史长度与提前停止设置；未使用测试集。',
            '', '| 设置 | 开发GAUC | 开发预测损失 | 相对原B1变化（百分点） | 用户抽样95%区间（百分点） |',
            '| --- | ---: | ---: | ---: | --- |']
        for m,v in result['baseline'].items():
            lines.append(f'| 原{m} | {v["gauc"]:.5f} | {v["log_loss"]:.5f} | — | — |')
        for entry in result['screening'].values():
            v=entry['metrics'];d=entry['difference'];lo,hi=d['ci95']
            lines.append(f'| B1，λ={entry["strength"]} | {v["gauc"]:.5f} | {v["log_loss"]:.5f} | {d["mean_difference"]*100:+.3f} | [{lo*100:.3f}, {hi*100:.3f}] |')
        lines+=['','预先约定：选择开发GAUC最高的强度；至少改善0.05个百分点且用户配对抽样区间下界大于0，才追加种子2027、2028和同正则的B0对照。区间用于探索性筛选，不包含挑选强度及训练随机性。',
                '', '初筛结论：'+('达到追加验证条件，已完成三种子和B0对照，见下方。' if result['worth_repeating'] else '未达到追加验证条件，结束本轮实验，保留原B0方案。')]
        for entry in result['screening'].values():
            lines+=['',f'## 历史ID覆盖分组：λ={entry["strength"]}','',
                    '| 分组 | 曝光数 | 原B1 GAUC | 正则B1 GAUC |','| --- | ---: | ---: | ---: |']
            for name,g in entry['groups'].items():
                lines.append(f'| {name} | {g["rows"]} | {g["original_B1"]["gauc"]:.5f} | {g["regularized_B1"]["gauc"]:.5f} |')
        if 'confirmation' in result:
            lines+=['','## 三种子追加对照','','```json',json.dumps(result['confirmation'],ensure_ascii=False,indent=2),'```']
        (paths.report_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status('complete','complete',worth_repeating=result['worth_repeating'],selected_strength=best['strength'])
    except BaseException as exc:
        status('failed','failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':
    main()
