"""Final evaluation, only after the three paired training seeds complete."""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import load_config
from .sequence_model import SequenceRanker
from .sequence_training import Inputs, evaluate, metrics
from .utils import write_json

SEEDS=(2026,2027,2028)
MODES=('B0','B1','B2')


def paired_interval(frame, scores_a, scores_b, repeats=2000, seed=2026):
    """User-cluster resampling of the mean paired-seed GAUC difference."""
    groups=frame.groupby('user_id',sort=True).indices
    # Include invalid users with zero weight: resample the whole evaluation population.
    weights=[]; diffs=[]
    y=frame.long_view.to_numpy()
    for _,ix in groups.items():
        if np.unique(y[ix]).size<2:
            weights.append(0);diffs.append([0.]*len(scores_a));continue
        weights.append(len(ix))
        diffs.append([roc_auc_score(y[ix],b[ix])-roc_auc_score(y[ix],a[ix]) for a,b in zip(scores_a,scores_b)])
    w=np.asarray(weights,dtype=float); d=np.asarray(diffs).mean(1)
    if not w.sum():return {'mean_difference':None,'ci95':None,'valid_bootstrap':0}
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(w),np.full(len(w),1/len(w)),size=repeats)
    den=counts@w; valid=den>0
    values=(counts[valid]@(w*d))/den[valid]
    return dict(mean_difference=float(w@d/w.sum()),ci95=np.quantile(values,[.025,.975]).tolist(),
                valid_bootstrap=int(valid.sum()),users=len(w),repeats=repeats,
                seed_pairs=len(scores_a),
                interpretation='User sampling uncertainty conditional on the evaluated paired trained models')


def select_on_dev(report, artifact_dir=None):
    train=report['training']
    for mode in MODES:
        for seed in SEEDS:
            r=train[f'{mode}_seed{seed}']
            if r['status']!='development_comparison':raise ValueError('Pilot results cannot select the final model')
    means={m:float(np.mean([train[f'{m}_seed{s}']['best_dev']['gauc'] for s in SEEDS])) for m in MODES}
    chosen=max(MODES,key=lambda m:means[m])
    intervals={}
    if artifact_dir is not None:
        reference=pd.read_parquet(artifact_dir/'B0_seed2026/dev_predictions.parquet').drop(columns='score').reset_index(drop=True)
        scores={m:[] for m in MODES}
        for m in MODES:
            for s in SEEDS:
                f=pd.read_parquet(artifact_dir/f'{m}_seed{s}/dev_predictions.parquet')
                if not f.drop(columns='score').reset_index(drop=True).equals(reference):
                    raise ValueError('Mismatched paired development records')
                scores[m].append(f.score.to_numpy(copy=True))
        del f
        for a,b in [('B0','B1'),('B0','B2'),('B1','B2')]:
            intervals[f'{b}_minus_{a}']=paired_interval(reference,scores[a],scores[b])
        # Favor simpler alternatives if the higher mean has no clear paired support.
        for simpler in MODES[:MODES.index(chosen)]:
            interval=intervals[f'{chosen}_minus_{simpler}']['ci95']
            if interval is not None and interval[0]<=0:
                chosen=simpler;break
    return dict(chosen=chosen,criterion='Mean development GAUC; prefer simpler model when paired user interval includes zero',
                mean_dev_gauc=means,development_intervals=intervals,chosen_before_test=True,seeds=list(SEEDS))


def s0_scores(events, target):
    hist=events.loc[events.split.isin(['warmup','train']) & events.is_rand.eq(0),['user_id','tab','long_view']]
    prior=float(hist.long_view.mean())
    t=hist.groupby('tab').long_view.agg(['sum','count'])
    tr=(t['sum']+20*prior)/(t['count']+20)
    p=hist.groupby(['user_id','tab']).long_view.agg(['sum','count'])
    pr=(p['sum']+20*p.index.get_level_values('tab').map(tr).to_numpy())/(p['count']+20)
    scores=pr.reindex(pd.MultiIndex.from_frame(target[['user_id','tab']])).to_numpy()
    return np.where(np.isnan(scores),target.tab.map(tr).fillna(prior).to_numpy(),scores)


def run(config):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    paths=config['resolved_paths']; report_path=paths.report_dir/'experiment_results.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    selection=select_on_dev(report,paths.artifact_dir)
    report['selection']=selection
    # Save the development decision BEFORE constructing/reading test samples.
    write_json(report_path,report)
    torch.set_num_threads(config['model']['threads'])
    device=torch.device(config['model']['device']); bs=config['model']['batch_size']
    inputs=Inputs(config,include_test=True)
    results={}; examples=[]
    for split in ('dev','test'):
        for random in (0,1):
            ids=np.flatnonzero(inputs.events.split.eq(split)&inputs.events.is_rand.eq(random))
            if not len(ids):continue
            frame=inputs.events.iloc[ids].reset_index(drop=True)
            label=f'{split}_{"random" if random else "standard"}'
            scores={m:[] for m in MODES}; details={}
            for mode in MODES:
                details[mode]=[]
                for seed in SEEDS:
                    checkpoint=torch.load(paths.artifact_dir/f'{mode}_seed{seed}/best.pt',map_location='cpu',weights_only=False)
                    assert checkpoint['spec']==inputs.spec
                    assert checkpoint['feature_info']==inputs.feature_info
                    model=SequenceRanker(**inputs.spec,mode=mode).to(device)
                    model.load_state_dict(checkpoint['state_dict'])
                    started=time.perf_counter(); pred=evaluate(model,inputs,ids,bs,device)
                    prediction_seconds=time.perf_counter()-started
                    scores[mode].append(pred)
                    detail=dict(seed=seed,metrics=metrics(frame,pred),prediction_seconds=prediction_seconds)
                    detail['tabs']={str(t):metrics(frame.iloc[ix].reset_index(drop=True),pred[ix]) for t,ix in frame.groupby('tab').indices.items()}
                    # Compare known and unseen candidate IDs without changing the evaluation population.
                    known=inputs.ids[ids]>1
                    detail['item_groups']={}
                    for name,mask in [('known',known),('unseen',~known)]:
                        if mask.any():detail['item_groups'][name]=metrics(frame.loc[mask].reset_index(drop=True),pred[mask])
                    dur=inputs.items.video_duration.fillna(0).to_numpy()[inputs.rows[ids]-1]/1000
                    detail['duration_groups']={}
                    for name,mask in [('0-15s',dur<=15),('15-60s',(dur>15)&(dur<=60)),('60s+',dur>60)]:
                        if mask.any():detail['duration_groups'][name]=metrics(frame.loc[mask].reset_index(drop=True),pred[mask])
                    lengths=frame.history_len.to_numpy()
                    detail['history_groups']={}
                    for name,mask in [('empty',lengths==0),('1-5',(lengths>0)&(lengths<=5)),('6-19',(lengths>5)&(lengths<20)),('20',lengths==20)]:
                        if mask.any():detail['history_groups'][name]=metrics(frame.loc[mask].reset_index(drop=True),pred[mask])
                    details[mode].append(detail)
                    out=frame[['user_id','video_id','tab','time_ms','long_view']].copy();out['score']=pred
                    out.to_parquet(paths.artifact_dir/f'{mode}_seed{seed}/{label}_predictions.parquet',index=False)
                    del model,checkpoint
                    torch.cuda.empty_cache()
            summary={m:{metric:dict(mean=float(np.mean([r['metrics'][metric] for r in details[m]])),
                                   std=float(np.std([r['metrics'][metric] for r in details[m]],ddof=1)))
                        for metric in ['auc','gauc','log_loss']} for m in MODES}
            result=dict(rows=len(frame),users=frame.user_id.nunique(),positive_rate=float(frame.long_view.mean()),
                        unknown_user_rows=int((inputs.users[ids]==0).sum()),unknown_scene_rows=int((inputs.tabs[ids]==0).sum()),
                        S0=metrics(frame,s0_scores(inputs.events,frame)),models=details,summary=summary,
                        B2_minus_B1=paired_interval(frame,scores['B1'],scores['B2']))
            baseline=s0_scores(inputs.events,frame)
            result['S0_tabs']={str(t):metrics(frame.iloc[ix].reset_index(drop=True),baseline[ix]) for t,ix in frame.groupby('tab').indices.items()}
            results[label]=result
            print(f'{label}: rows={len(frame):,}; GAUC '+str({m:round(summary[m]['gauc']['mean'],5) for m in MODES}),flush=True)
            if split=='test' and not random:
                # Conservative check: wait until the recorded play duration has elapsed.
                # Keep original checkpoints, last-20 window and length feature unchanged.
                positive=inputs.events.loc[inputs.events.long_view.eq(1)&inputs.events.is_rand.eq(0),
                    ['user_id','video_id','time_ms','source','source_row','play_time_ms']].sort_values(
                    ['user_id','time_ms','source','source_row'],kind='stable')
                assert np.array_equal(positive.video_id.to_numpy(),inputs.pool)
                available=(positive.time_ms+positive.play_time_ms.clip(lower=0)).to_numpy()
                affected=[]
                for start in range(0,len(ids),bs):
                    ix=ids[start:start+bs];pos=inputs.ends[ix,None]-inputs.lengths[ix,None]+np.arange(inputs.length)[None,:]
                    valid=np.arange(inputs.length)[None,:]<inputs.lengths[ix,None]
                    pending=available[np.minimum(pos,len(available)-1)]>=inputs.events.time_ms.to_numpy()[ix,None]
                    affected.extend(ix[(valid&pending).any(1)].tolist())
                affected=np.array(affected,dtype=np.int64)
                result['feedback_sensitivity']=dict(affected_rows=len(affected),models={},
                    rule='Mask last-20 history entries until event time + recorded play duration; no refill, no retraining, unchanged length feature')
                inputs.history_available_at=available
                for mode in ('B1','B2'):
                    results_delayed=[]
                    for si,seed in enumerate(SEEDS):
                        pred=scores[mode][si].copy()
                        if len(affected):
                            ck=torch.load(paths.artifact_dir/f'{mode}_seed{seed}/best.pt',map_location='cpu',weights_only=False)
                            model=SequenceRanker(**inputs.spec,mode=mode).to(device);model.load_state_dict(ck['state_dict'])
                            pred[np.searchsorted(ids,affected)]=evaluate(model,inputs,affected,bs,device)
                            del model,ck
                        results_delayed.append(dict(seed=seed,metrics=metrics(frame,pred),
                            mean_absolute_score_change=float(np.abs(pred-scores[mode][si]).mean())))
                    result['feedback_sensitivity']['models'][mode]=results_delayed
                inputs.history_available_at=None
                # Fixed seed; choose one improved and one worsened POSITIVE-vs-NEGATIVE ordering within user x tab.
                a,b=scores['B1'][0],scores['B2'][0]
                for direction in ('improved','worsened'):
                    found=None
                    for (user,tab),ix in frame.groupby(['user_id','tab'],sort=True).indices.items():
                        pos=ix[frame.long_view.to_numpy()[ix]==1]; neg=ix[frame.long_view.to_numpy()[ix]==0]
                        for pi in pos[:50]:
                            valid=((a[pi]<=a[neg])&(b[pi]>b[neg])) if direction=='improved' else ((a[pi]>a[neg])&(b[pi]<=b[neg]))
                            if valid.any():
                                ni=int(neg[np.flatnonzero(valid)[0]]);found=(int(pi),ni);break
                        if found:break
                    if found:
                        rows=[]
                        for local in found:
                            r=frame.iloc[local]; idx=ids[local]
                            history=inputs.pool[int(r.history_start):int(r.history_end)].tolist()
                            batch=inputs.batch(np.array([idx]),device)
                            ck=torch.load(paths.artifact_dir/'B2_seed2026/best.pt',map_location='cpu',weights_only=False)
                            model=SequenceRanker(**inputs.spec,mode='B2').to(device);model.load_state_dict(ck['state_dict']);model.eval()
                            with torch.inference_mode():
                                ci=torch.where(model.trained_items[batch['item']],batch['item'],torch.ones_like(batch['item']))
                                hi=torch.where(model.trained_items[batch['history']],batch['history'],torch.ones_like(batch['history']))
                                cv=model.encode(ci,batch['tags']);hv=model.encode(hi,batch['history_tags']);cv=cv[:,None,:].expand_as(hv)
                                att=model.attention(torch.cat([hv,cv,hv-cv,hv*cv],-1)).squeeze(-1)
                                w=(torch.softmax(att.masked_fill(~batch['mask'],-1e9),1)*batch['mask']).cpu().numpy()[0]
                            order=np.argsort(w[:len(history)])[::-1][:5]
                            rows.append(dict(user_id=int(r.user_id),tab=int(r.tab),video_id=int(r.video_id),time_ms=int(r.time_ms),
                                label=int(r.long_view),B1=float(a[local]),B2=float(b[local]),
                                history_top=[dict(video_id=int(history[j]),weight=float(w[j])) for j in order]))
                            del model,ck
                        examples.append(dict(direction=direction,records=rows))
            report['final_evaluation']=results
            write_json(report_path,report)
    write_json(paths.artifact_dir/'examples.json',examples)
    lines=['# 两种历史模型的案例对照','','案例按固定种子2026、用户与场景排序，选择第一组符合条件的正负记录。两条记录可能发生在不同时刻，各自使用当时历史，不称为同一次展示。权重仅解释模型计算。']
    for ex in examples:
        lines+=['', '## '+('排序改善' if ex['direction']=='improved' else '排序变差'),'', '| 记录 | 视频ID | 标签 | 均值模型分数 | 加权模型分数 |','| --- | --- | ---: | ---: | ---: |']
        for i,r in enumerate(ex['records'],1):
            lines.append(f'| {i} | {r["video_id"]} | {r["label"]} | {r["B1"]:.5f} | {r["B2"]:.5f} |')
        lines+=['','关注的历史（视频ID／权重）：']
        for i,r in enumerate(ex['records'],1):lines.append(f'- 记录{i}：'+', '.join(f'{h["video_id"]}／{h["weight"]:.3f}' for h in r['history_top']))
    lines+=['','游戏类别映射尚未完整确认，本轮不将数字标签直接命名为游戏类别；以上为一般内容案例。逐记录标识仅用于本地复查。']
    (paths.report_dir/'examples.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/kuairand_1k.yaml')
    run(load_config(p.parse_args().config))
