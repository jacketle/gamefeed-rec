"""GPU pilot and matched training for the prepared 1K exposure samples.

python -m gamefeed.sequence_training --config configs/kuairand_1k.yaml --epochs 1
No test labels are evaluated by this training command.
"""
from __future__ import annotations
import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import psutil
from sklearn.metrics import roc_auc_score, log_loss

from .config import load_config
from .features import parse_tags
from .sequence_model import SequenceRanker
from .sequence_data import compact_events
from .utils import write_json


def metrics(frame, scores):
    y = frame.long_view.to_numpy()
    rows = []
    for uid, ix in frame.groupby('user_id', sort=False).indices.items():
        if np.unique(y[ix]).size == 2:
            rows.append(dict(user=int(uid), rows=len(ix), auc=float(roc_auc_score(y[ix],scores[ix]))))
    weight = sum(r['rows'] for r in rows)
    return dict(auc=float(roc_auc_score(y,scores)) if np.unique(y).size==2 else None,
        log_loss=float(log_loss(y,np.clip(scores,1e-7,1-1e-7),labels=[0,1])),
        gauc=sum(r['auc']*r['rows'] for r in rows)/weight if weight else None,
        rows=len(frame), positive_rate=float(y.mean()), valid_users=len(rows),
        total_users=int(frame.user_id.nunique()), covered_rows=weight,
        user_coverage=len(rows)/frame.user_id.nunique(), exposure_coverage=weight/len(frame))


class Inputs:
    def __init__(self, config, include_test=False):
        root = config['resolved_paths'].processed_dir
        self.length = int(config['data']['history_length'])
        # Test data is deliberately not loaded by the training command.
        end_date = config['data']['test_end'] if include_test else config['data']['dev_end']
        paths = sorted(p for p in root.glob('events_*.parquet') if int(p.stem.split('_')[1]) <= end_date)
        columns=['user_id','video_id','date','time_ms','tab','long_view','is_rand',
                 'play_time_ms','source','source_row','split','history_start','history_end','history_len']
        self.events = pd.concat([compact_events(pd.read_parquet(p, columns=columns)) for p in paths], ignore_index=True)
        model_config = config.get('model', {})
        use_author = model_config.get('use_author', False)
        self.use_history_matches = model_config.get('use_history_matches', False)
        item_columns = ['video_id','video_duration','tag'] + (['author_id'] if use_author or self.use_history_matches else [])
        self.items = pd.read_parquet(root/'items.parquet',columns=item_columns).sort_values('video_id').reset_index(drop=True)
        raw = self.items.video_id.to_numpy(np.int64)
        self.row_map = np.zeros(int(raw.max())+1, np.int32)
        self.row_map[raw] = np.arange(len(raw), dtype=np.int32)+1
        allowed = self.events.loc[self.events.date.le(config['data']['train_end']) & self.events.is_rand.eq(0),['user_id','video_id','tab']]
        known = np.sort(allowed.video_id.unique())
        self.item_map = np.ones(len(self.row_map), np.int32)
        self.item_map[known] = np.arange(len(known), dtype=np.int32)+2
        self.num_items = len(known)+2
        # Parse repeated tag strings once, rather than millions of Python lists.
        tag_codes, tag_patterns = pd.factorize(self.items.tag.fillna(''), sort=False)
        raw_tags = [parse_tags(v) for v in tag_patterns]
        allowed_rows = self.row_map[known]-1
        tag_set = sorted({t for i in np.unique(tag_codes[allowed_rows]) for t in raw_tags[i]})
        tag_map = {t:i+2 for i,t in enumerate(tag_set)}
        self.num_tags = len(tag_map)+2
        max_tags = max(1,max(map(len,raw_tags)))
        pattern_array = np.zeros((len(raw_tags),max_tags),np.int32)
        for i,tags in enumerate(raw_tags):
            values = [tag_map.get(t,1) for t in tags] or [1]
            pattern_array[i,:len(values)] = values
        self.tag_array = np.zeros((len(self.items)+1,max_tags),np.int32)
        self.tag_array[1:] = pattern_array[tag_codes]
        self.tag_array[0,0]=1
        self.items = self.items.drop(columns='tag')
        self.pool = np.load(root/'history_pool.npy',mmap_mode='r')
        self.history_available_at = None  # Optional evaluation-only delayed-feedback sensitivity.
        # Raw IDs remain only in local processed files. Padding uses a separate mask.
        user_ids=sorted(allowed.user_id.unique()); tab_ids=sorted(allowed.tab.unique())
        self.user_map={u:i+1 for i,u in enumerate(user_ids)}
        self.tab_map={t:i+1 for i,t in enumerate(tab_ids)}
        ev=self.events
        self.rows=self.row_map[ev.video_id.to_numpy()]
        self.ids=self.item_map[ev.video_id.to_numpy()]
        self.users=ev.user_id.map(self.user_map).fillna(0).to_numpy(np.int64)
        self.tabs=ev.tab.map(self.tab_map).fillna(0).to_numpy(np.int64)
        self.train=np.flatnonzero(ev.split.eq('train') & ev.is_rand.eq(0))
        self.dev=np.flatnonzero(ev.split.eq('dev') & ev.is_rand.eq(0))
        if not len(self.train) or not len(self.dev):raise ValueError('Empty train/dev')
        # Warmup vocabulary membership does not train user/scene embeddings.
        self.users=np.where(np.isin(self.users,np.unique(self.users[self.train])),self.users,0)
        self.tabs=np.where(np.isin(self.tabs,np.unique(self.tabs[self.train])),self.tabs,0)
        durations=np.zeros(len(self.items)+1,np.float32)
        durations[1:]=np.log1p(self.items.video_duration.fillna(0).clip(lower=0).to_numpy(np.float32)/1000)
        numeric=np.stack([durations[self.rows],np.log1p(ev.history_len.to_numpy(np.float32))],1)
        mean=numeric[self.train].mean(0); std=numeric[self.train].std(0).clip(min=1e-6)
        numeric=(numeric-mean)/std
        hour=pd.to_datetime(ev.time_ms,unit='ms',utc=True).dt.tz_convert('Asia/Shanghai').dt.hour.to_numpy()
        self.numeric=np.column_stack([numeric,np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24)]).astype(np.float32)
        self.ends=ev.history_end.to_numpy(np.int64)
        self.lengths=ev.history_len.to_numpy(np.int64)
        self.y=ev.long_view.to_numpy(np.float32)
        self.spec=dict(num_users=len(user_ids)+1,num_items=self.num_items,num_tags=self.num_tags,num_tabs=len(tab_ids)+1)
        self.feature_info=dict(vocabulary_items=len(known),tags=len(tag_set),max_tags=max_tags,
            numeric_mean=mean.tolist(),numeric_std=std.tolist(),history_length=self.length)
        self.output_root=root
        self.feature_maps = dict(known_item_ids=known,raw_item_rows=raw,
                                users=np.array(user_ids),tabs=np.array(tab_ids),tags=np.array(tag_set))
        if use_author:
            # Only authors of formal training exposures enter this vocabulary.
            author_ids = self.items.author_id.fillna(-1).to_numpy(np.int64)
            train_authors = author_ids[self.rows[self.train]-1]
            all_authors, counts = np.unique(train_authors[train_authors >= 0], return_counts=True)
            minimum = int(model_config.get('author_min_count', 1))
            if minimum < 1:
                raise ValueError('author_min_count must be positive')
            self.author_kept_indices = np.flatnonzero(counts >= minimum)
            self.author_full_count = len(all_authors)
            known_authors = all_authors[self.author_kept_indices]
            author_map = {int(a):i+1 for i,a in enumerate(known_authors)}
            author_rows = np.zeros(len(self.items)+1, np.int32)
            author_rows[1:] = self.items.author_id.map(author_map).fillna(0).to_numpy(np.int32)
            self.authors = author_rows[self.rows]
            self.spec['num_authors'] = len(known_authors)+1
            self.feature_maps['authors'] = known_authors
            self.feature_info['author_count'] = len(known_authors)
            if 'author_min_count' in model_config or model_config.get('paired_initialization'):
                self.feature_info.update(author_min_count=minimum, full_train_author_count=len(all_authors))
                self.feature_maps['all_train_authors'] = all_authors
                self.feature_maps['author_train_counts'] = counts[self.author_kept_indices]
        if self.use_history_matches:
            self.raw_author_rows = np.full(len(self.items)+1, -1, dtype=np.int64)
            self.raw_author_rows[1:] = self.items.author_id.fillna(-1).to_numpy(np.int64)
            self.spec['match_dim'] = 3
            self.feature_info['history_matches'] = dict(
                names=['same_author_fraction', 'same_author_any', 'shared_tag_history_fraction'],
                denominator='valid prior history positions; empty history yields zero',
                author_rule='raw author IDs >= 0, independent of embedding frequency threshold',
                tag_rule='any shared known tag (>1) marks one matching history position')

    def batch(self, indices, device):
        offsets=np.arange(self.length)[None,:]
        valid=offsets<self.lengths[indices,None]
        pos=self.ends[indices,None]-self.lengths[indices,None]+offsets
        history_raw=self.pool[np.minimum(pos,len(self.pool)-1)] if len(self.pool) else np.zeros_like(pos)
        if self.history_available_at is not None and len(self.pool):
            valid &= self.history_available_at[np.minimum(pos,len(self.pool)-1)] < self.events.time_ms.to_numpy()[indices,None]
        history=np.where(valid,self.item_map[history_raw],0)
        history_tags=self.tag_array[self.row_map[history_raw]].copy()
        history_tags[~valid]=0
        arrays=dict(item=self.ids[indices],tags=self.tag_array[self.rows[indices]],history=history,
                    history_tags=history_tags,mask=valid,user=self.users[indices],tab=self.tabs[indices],
                    numeric=self.numeric[indices],label=self.y[indices])
        if hasattr(self, 'authors'):
            arrays['author'] = self.authors[indices]
        if self.use_history_matches:
            candidate_author = self.raw_author_rows[self.rows[indices], None]
            history_author = self.raw_author_rows[self.row_map[history_raw]]
            same_author = valid & (candidate_author >= 0) & (history_author >= 0) & (candidate_author == history_author)
            candidate_tags = arrays['tags'][:, None, :, None]
            prior_tags = history_tags[:, :, None, :]
            shared_tags = ((candidate_tags > 1) & (prior_tags > 1) & (candidate_tags == prior_tags)).any(axis=(2, 3)) & valid
            denominator = valid.sum(1).clip(min=1)
            arrays['matches'] = np.column_stack([same_author.sum(1)/denominator,
                same_author.any(1), shared_tags.sum(1)/denominator]).astype(np.float32)
        return {k:torch.as_tensor(a,dtype=torch.bool if k=='mask' else torch.float32 if k in ('numeric','label','matches') else torch.long,device=device) for k,a in arrays.items()}


def initialize_author_weights(model, inputs, seed, std):
    """Same raw author gets the same initial vector across frequency thresholds."""
    if not np.isfinite(std) or std <= 0:
        raise ValueError('author_init_std must be finite and positive')
    if model.author is None:
        return
    generator = torch.Generator(device='cpu').manual_seed(int(seed)+17071)
    full = torch.randn((inputs.author_full_count, 16), generator=generator)
    selected = full[inputs.author_kept_indices] * std
    with torch.no_grad():
        model.author.weight.zero_()
        model.author.weight[1:].copy_(selected.to(model.author.weight.device))


def evaluation_steps(num_batches, epoch, fraction=1.0):
    """Quarter evaluations in the first two epochs; always include epoch end."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('eval_every_fraction must be in (0, 1]')
    pieces = round(1/fraction)
    if not np.isclose(pieces*fraction, 1):
        raise ValueError('eval_every_fraction must evenly divide one epoch')
    if epoch > 2:
        return {num_batches}
    return {int(np.ceil(num_batches*i/pieces)) for i in range(1, pieces+1)}


def evaluate(model, inputs, ids, batch_size, device):
    model.eval(); scores=[]
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            batch=inputs.batch(ids[start:start+batch_size],device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores)


def run(config, modes, epochs=None, seed=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cfg=config['model']; seed=int(seed if seed is not None else config['seed'])
    item_l2=float(cfg.get('item_l2',0.0))
    if not np.isfinite(item_l2) or item_l2<0:
        raise ValueError('item_l2 must be finite and nonnegative')
    if item_l2 and cfg.get('history_id_dropout',0):
        raise ValueError('Do not combine item L2 with history masking in this trial')
    device=torch.device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter(); inputs=Inputs(config)
    paths=config['resolved_paths']; bs=int(cfg['batch_size'])
    eval_bs = int(cfg.get('eval_batch_size', bs))
    eval_fraction = float(cfg.get('eval_every_fraction', 1.0))
    evaluation_steps(1, 1, eval_fraction)
    if eval_bs < 1:
        raise ValueError('eval_batch_size must be positive')
    paths.artifact_dir.mkdir(parents=True,exist_ok=True)
    paths.report_dir.mkdir(parents=True,exist_ok=True)
    map_dir = paths.artifact_dir if cfg.get('feature_maps_in_artifact', False) or cfg.get('use_author', False) else paths.processed_dir
    np.savez(map_dir/'sequence_feature_maps.npz', **inputs.feature_maps)
    report_path=paths.report_dir/'experiment_results.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    report['features']=inputs.feature_info
    report['feature_rules']='Untrained user/scene IDs map to zero; untrained item IDs use zero embedding and retain known tags'
    report['training_tabs']={str(t):int(n) for t,n in inputs.events.iloc[inputs.train].tab.value_counts().items()}
    report['hardware']=dict(torch=torch.__version__,device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu',batch_size=bs)
    # Historical user-scenario propensity: constant within user x tab.
    historical=inputs.events.loc[inputs.events.split.isin(['warmup','train']) & inputs.events.is_rand.eq(0),['user_id','tab','long_view']]
    global_rate=float(historical.long_view.mean())
    tab_stats=historical.groupby('tab').long_view.agg(['sum','count'])
    tab_rates=(tab_stats['sum']+20*global_rate)/(tab_stats['count']+20)
    pair_stats=historical.groupby(['user_id','tab']).long_view.agg(['sum','count'])
    tab_prior=pair_stats.index.get_level_values('tab').map(tab_rates).to_numpy()
    pair_rates=(pair_stats['sum']+20*tab_prior)/(pair_stats['count']+20)
    dev_frame=inputs.events.iloc[inputs.dev].reset_index(drop=True)
    pairs=pd.MultiIndex.from_frame(dev_frame[['user_id','tab']])
    propensity=pair_rates.reindex(pairs).to_numpy()
    fallback=dev_frame.tab.map(tab_rates).fillna(global_rate).to_numpy()
    propensity=np.where(np.isnan(propensity),fallback,propensity)
    report['S0']=dict(smoothing=20,dev=metrics(dev_frame,propensity),
        note='Fixed initial smoothing, not tuned. Constant within user and scenario; no item discrimination.')
    write_json(report_path,report)
    print(f'Inputs ready: {len(inputs.train):,} train / {len(inputs.dev):,} dev, {inputs.num_items:,} item embeddings',flush=True)
    for mode in modes:
        torch.manual_seed(seed); np.random.seed(seed)
        history_id_dropout=float(cfg.get('history_id_dropout',0.0))
        model_kwargs = dict(history_id_dropout=history_id_dropout,
                            paired_initialization=bool(cfg.get('paired_initialization', False)))
        model=SequenceRanker(**inputs.spec,mode=mode,**model_kwargs)
        author_std = float(cfg.get('author_init_std', 1.0))
        if model.author is not None and (model_kwargs['paired_initialization'] or 'author_init_std' in cfg):
            initialize_author_weights(model, inputs, seed, author_std)
        initialization = dict(paired=model_kwargs['paired_initialization'],
            extra_input_columns=model.output[0].in_features-92)
        if initialization['extra_input_columns']:
            initialization['extra_input_initial_max_abs'] = float(model.output[0].weight[:,92:].detach().abs().max())
        if model.author is not None and (model_kwargs['paired_initialization'] or 'author_init_std' in cfg):
            initialization.update(author_init_std=author_std, author_generator_seed=seed+17071,
                full_train_author_count=inputs.author_full_count,
                author_alignment='all formal training raw authors sorted; retained rows selected after initialization')
        model = model.to(device)
        optim=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        loss_fn=torch.nn.BCEWithLogitsLoss()
        out=paths.artifact_dir/f'{mode}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
        best=-float('inf'); stale=0; curve=[]; evaluations=[]
        result=dict(mode=mode,seed=seed,history_id_dropout=history_id_dropout,item_l2=item_l2,
                    epochs=curve,evaluations=evaluations,initialization=initialization,
                    model_kwargs=model_kwargs,eval_batch_size=eval_bs,eval_every_fraction=eval_fraction,
                    parameters=sum(p.numel() for p in model.parameters()),status='training')
        report.setdefault('training',{})[f'{mode}_seed{seed}']=result
        rng=np.random.default_rng(seed)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,(epochs or cfg['epochs'])+1):
            t=time.perf_counter(); model.train(); total_loss=0.; total_penalty=0.
            training_seconds=0.; epoch_improved=False
            order=rng.permutation(inputs.train)
            num_batches=(len(order)+bs-1)//bs
            eval_steps=evaluation_steps(num_batches,epoch,eval_fraction)
            for start in range(0,len(order),bs):
                ix=order[start:start+bs]; batch=inputs.batch(ix,device)
                optim.zero_grad(set_to_none=True)
                loss=loss_fn(model(batch),batch['label'])
                penalty=item_l2*model.item_l2(batch) if item_l2 else loss.new_zeros(())
                (loss+penalty).backward();optim.step()
                total_loss+=loss.item()*len(ix)
                total_penalty+=penalty.item()*len(ix)
                if start and start//bs%500==0:print(f'{mode} epoch {epoch}: {start:,}/{len(order):,}',flush=True)
                step=start//bs+1
                if step not in eval_steps:
                    continue
                training_seconds+=time.perf_counter()-t
                evaluation_started=time.perf_counter()
                scores=evaluate(model,inputs,inputs.dev,eval_bs,device)
                m=metrics(dev_frame,scores)
                processed=start+len(ix)
                point=dict(epoch=epoch,fractional_epoch=epoch-1+processed/len(order),
                    eval_step=(epoch-1)*num_batches+step,
                    train_loss=total_loss/processed,train_regularization=total_penalty/processed,
                    train_objective=(total_loss+total_penalty)/processed,train_seconds=training_seconds,
                    evaluation_seconds=time.perf_counter()-evaluation_started,dev=m)
                evaluations.append(point)
                if step == num_batches:
                    curve.append(point.copy())
                print(f'{mode} epoch {point["fractional_epoch"]:.3f}: BCE={point["train_loss"]:.5f} L2={point["train_regularization"]:.6f} dev GAUC={m["gauc"]:.5f} train={training_seconds:.1f}s',flush=True)
                if m['gauc']>best:
                    best=m['gauc']; epoch_improved=True
                    torch.save({'state_dict':model.state_dict(),'spec':inputs.spec,'mode':mode,'seed':seed,'epoch':epoch,
                        'fractional_epoch':point['fractional_epoch'],'eval_step':point['eval_step'],
                        'feature_info':inputs.feature_info,'history_id_dropout':history_id_dropout,'item_l2':item_l2,
                        'model_kwargs':model_kwargs},out/'best.pt')
                    prediction=dev_frame[['user_id','video_id','tab','time_ms','long_view']].copy()
                    prediction['score']=scores
                    prediction.to_parquet(out/'dev_predictions.parquet',index=False)
                    result['best_dev']=m
                    result['best_fractional_epoch']=point['fractional_epoch']
                    result['best_eval_step']=point['eval_step']
                report['elapsed_seconds']=time.perf_counter()-started
                write_json(report_path,report)
                model.train()
                t=time.perf_counter()
            stale=0 if epoch_improved else stale+1
            if stale>=cfg['patience']:break
        result.update(peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
                      status='pilot' if epochs==1 else 'development_comparison')
        checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        seen=checkpoint['state_dict']['trained_items'].numpy()
        candidate_known=inputs.ids[inputs.dev]>1
        history_known=history_seen=history_count=0
        for start in range(0,len(inputs.dev),bs):
            batch=inputs.batch(inputs.dev[start:start+bs],torch.device('cpu'))
            h=batch['history'][batch['mask']].numpy()
            history_count+=len(h); history_known+=int((h>1).sum()); history_seen+=int((seen[h] & (h>1)).sum())
        result['dev_coverage']=dict(candidate_vocabulary=float(candidate_known.mean()),
            candidate_trained=float((seen[inputs.ids[inputs.dev]] & candidate_known).mean()),
            history_vocabulary=history_known/max(history_count,1),history_trained=history_seen/max(history_count,1),
            tag_available=float((inputs.tag_array[inputs.rows[inputs.dev]]>1).any(1).mean()))
        prediction=pd.read_parquet(out/'dev_predictions.parquet')
        result['dev_tabs']={str(tab):metrics(g.reset_index(drop=True),g.score.to_numpy()) for tab,g in prediction.groupby('tab')}
        mem=psutil.Process().memory_info()
        result['peak_process_mib']=getattr(mem,'peak_wset',mem.rss)/2**20
        report.setdefault('training',{})[f'{mode}_seed{seed}']=result
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(report_path,report)
        del model,optim
        if device.type=='cuda':torch.cuda.empty_cache()
    print('Completed. Test split was not evaluated.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/kuairand_1k.yaml')
    parser.add_argument('--modes',nargs='+',choices=['B0','B1','B2'],default=['B0','B1','B2'])
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    run(load_config(args.config),args.modes,args.epochs,args.seed)
