"""One B1 seed with history-ID masking, then development-only comparison."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from gamefeed.config import load_config, ensure_output_dirs
from gamefeed.sequence_training import run, metrics
from gamefeed.sequence_evaluation import paired_interval

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/kuairand_1k/full1000'
CONFIG=ROOT/'configs/kuairand_1k_history_mask.yaml'


def compare(config):
    paths=config['resolved_paths']
    new=pd.read_parquet(paths.artifact_dir/'B1_seed2026/dev_predictions.parquet').reset_index(drop=True)
    original={m:pd.read_parquet(BASE/f'{m}_seed2026/dev_predictions.parquet').reset_index(drop=True) for m in ('B0','B1')}
    for f in original.values():
        assert f.drop(columns='score').equals(new.drop(columns='score'))
    files=sorted(paths.processed_dir.glob('events_*.parquet'))
    columns=['user_id','video_id','tab','time_ms','long_view','split','is_rand','history_end','history_len']
    frames=[]
    for p in files:
        if config['data']['train_end'] < int(p.stem.split('_')[1]) <= config['data']['dev_end']:
            f=pd.read_parquet(p,columns=columns)
            frames.append(f.loc[f.split.eq('dev') & f.is_rand.eq(0)])
    ev=pd.concat(frames,ignore_index=True)
    for c in ['user_id','video_id','tab','time_ms','long_view']:
        np.testing.assert_array_equal(ev[c],new[c])
    # Fixed groups based on original B1's trained IDs, shared by all comparisons.
    ck=torch.load(BASE/'B1_seed2026/best.pt',map_location='cpu',weights_only=False)
    seen=ck['state_dict']['trained_items'].numpy().copy();del ck
    with np.load(paths.processed_dir/'sequence_feature_maps.npz') as maps:
        known=maps['known_item_ids']
    pool=np.load(paths.processed_dir/'history_pool.npy',mmap_mode='r')
    loc=np.searchsorted(known,pool)
    exists=(loc<len(known)) & (known[np.minimum(loc,len(known)-1)]==pool)
    pool_seen=exists & seen[np.where(exists,loc+2,1)]
    coverage=np.zeros(len(ev),np.float32)
    offsets=np.arange(config['data']['history_length'])[None,:]
    for start in range(0,len(ev),50000):
        g=ev.iloc[start:start+50000]; lengths=g.history_len.to_numpy()[:,None]
        pos=g.history_end.to_numpy()[:,None]-lengths+offsets
        good=(offsets<lengths) & pool_seen[np.minimum(pos,len(pool)-1)]
        coverage[start:start+len(g)]=good.sum(1)/lengths[:,0].clip(min=1)
    predictions={'B0':original['B0'].score.to_numpy(),'B1':original['B1'].score.to_numpy(),'B1_mask50':new.score.to_numpy()}
    result={'seed':2026,'history_id_dropout':0.5,'split':'development_only','groups':{}}
    groups={'all':np.ones(len(new),bool),'history_id_coverage_0_25':coverage<=.25,
            'history_id_coverage_25_75':(coverage>.25)&(coverage<=.75),'history_id_coverage_75_100':coverage>.75}
    for name,mask in groups.items():
        frame=new.loc[mask].reset_index(drop=True)
        if not len(frame):continue
        result['groups'][name]={'rows':len(frame),'models':{m:metrics(frame,s[mask]) for m,s in predictions.items()},
            'masked_minus_original':paired_interval(frame,[predictions['B1'][mask]],[predictions['B1_mask50'][mask]])}
        print('Compared development group:',name,len(frame),flush=True)
    d=result['groups']['all']['masked_minus_original'];lo,hi=d['ci95']
    m=result['groups']['all']['models']
    # A screening rule only: repeated training is a separate follow-up.
    result['worth_repeating']=bool(lo>0 and m['B1_mask50']['log_loss']<=m['B1']['log_loss'])
    (paths.report_dir/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 历史视频ID随机屏蔽：首次对照','',
        '种子2026；B1训练时以50%的概率将历史ID向量置零，保留标签和有效位置，不放大保留向量。候选输入不变，评价时关闭随机屏蔽。使用原始全量划分，最多5轮、连续2轮无改善停止。','',
        '| 开发数据范围 | 曝光数 | 原B0 GAUC | 原B1 GAUC | 屏蔽版B1 GAUC | 原B1损失 | 屏蔽版损失 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,g in result['groups'].items():
        v=g['models'];lines.append(f'| {name} | {g["rows"]} | {v["B0"]["gauc"]:.5f} | {v["B1"]["gauc"]:.5f} | {v["B1_mask50"]["gauc"]:.5f} | {v["B1"]["log_loss"]:.5f} | {v["B1_mask50"]["log_loss"]:.5f} |')
    lines+=['',f'屏蔽版相对原B1开发GAUC变化：{d["mean_difference"]*100:+.3f}个百分点；按用户配对抽样2,000次的95%区间：[{lo*100:.3f}, {hi*100:.3f}]个百分点。',
        '', '分组按原B1检查点能使用的历史ID比例划分，各模型评价相同记录。该区间描述本次训练的用户抽样波动，不包含重复训练的变化。',
        '', '后续判断：'+('GAUC差值区间高于零且损失未变差，值得补另外两个种子。' if result['worth_repeating'] else '未同时满足GAUC差值区间高于零、损失不变差，先结束本次尝试并检查具体结果。'),
        '', '本次未读取测试预测或改变最终方案。屏蔽能够改善时，表示此处理有效；是否由ID不可用差异或一般正则化作用导致，结合分组判断。']
    (paths.report_dir/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    os.chdir(ROOT)
    config=load_config(CONFIG);ensure_output_dirs(config)
    status_path=config['resolved_paths'].artifact_dir/'run_status.json'
    def status(state,phase,**kw):
        status_path.write_text(json.dumps(dict(state=state,phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kw),ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        status('running','training_B1_seed2026')
        run(config,['B1'],seed=2026)
        status('running','development_comparison')
        result=compare(config)
        status('complete','complete',worth_repeating=result['worth_repeating'])
        print('Complete: one training run and development comparison.',flush=True)
    except BaseException as exc:
        status('failed','failed',error=str(exc));traceback.print_exc();raise
