"""Explain existing full-1K results using saved predictions; no training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gamefeed.sequence_evaluation import paired_interval
from gamefeed.sequence_training import metrics

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/kuairand_1k/full1000'
ART = ROOT/'artifacts/kuairand_1k/full1000'
SEEDS = (2026, 2027, 2028)
MODES = ('B0','B1','B2')


def load(split):
    frame = None
    scores = {m: [] for m in MODES}
    for m in MODES:
        for s in SEEDS:
            f = pd.read_parquet(ART/f'{m}_seed{s}/{split}_predictions.parquet')
            base = f.drop(columns='score').reset_index(drop=True)
            if frame is None:
                frame = base
            else:
                assert frame.equals(base), 'Prediction rows must match before pairing'
            scores[m].append(f.score.to_numpy(copy=True))
    return frame, scores


def summarize(frame, scores, mask=None):
    if mask is not None:
        frame = frame.loc[mask].reset_index(drop=True)
        scores = {m: [v[mask] for v in vs] for m,vs in scores.items()}
    result = dict(rows=len(frame), users=int(frame.user_id.nunique()),
                  positive_rate=float(frame.long_view.mean()),
                  tabs={str(k): int(v) for k,v in frame.tab.value_counts().items()}, models={})
    for m in MODES:
        individual = [metrics(frame,s) for s in scores[m]]
        result['models'][m] = {k: float(np.mean([r[k] for r in individual]))
                               for k in ('auc','gauc','log_loss','user_coverage','exposure_coverage')}
        result['models'][m]['mean_probability'] = float(np.mean([s.mean() for s in scores[m]]))
        result['models'][m]['seed_gauc'] = [r['gauc'] for r in individual]
    for a,b in [('B0','B1'),('B0','B2')]:
        result[f'{b}_minus_{a}'] = paired_interval(frame,scores[a],scores[b])
    return result


def main():
    report = json.loads((REPORT/'experiment_results.json').read_text(encoding='utf-8'))
    result = {'curves': {}, 'groups': {}}
    for key, value in report['training'].items():
        result['curves'][key] = [dict(epoch=e['epoch'], train_loss=e['train_loss'],
            dev_loss=e['dev']['log_loss'], dev_gauc=e['dev']['gauc']) for e in value['epochs']]
    for split in ('dev','test'):
        rf, rs = load(split+'_random')
        result['groups'][split+'_random'] = summarize(rf,rs)
        sf, ss = load(split+'_standard')
        masks = {
            'standard_all': np.ones(len(sf),bool),
            'standard_random_tabs': sf.tab.isin(rf.tab.unique()).to_numpy(),
            'standard_random_tabs_users': (sf.tab.isin(rf.tab.unique()) & sf.user_id.isin(rf.user_id.unique())).to_numpy(),
            'standard_random_tabs_users_videos': (sf.tab.isin(rf.tab.unique()) & sf.user_id.isin(rf.user_id.unique()) & sf.video_id.isin(rf.video_id.unique())).to_numpy(),
        }
        for name,mask in masks.items():
            if mask.sum() and sf.loc[mask,'long_view'].nunique()==2:
                result['groups'][split+'_'+name] = summarize(sf,ss,mask)
                print(split,name,int(mask.sum()),flush=True)
    (REPORT/'history_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 历史模型诊断：使用已有预测，不重新训练','',
             '全部结果使用三个相同种子的平均指标；差值区间按用户配对抽样2,000次。限定范围的对照不做重新训练，也不将常规展示与随机展示当成随机分组实验。','',
             '## 学习曲线','','| 模型与种子 | 轮数 | 训练损失 | 开发损失 | 开发GAUC |','| --- | ---: | ---: | ---: | ---: |']
    for key,curve in result['curves'].items():
        for c in curve:
            lines.append(f'| {key} | {c["epoch"]} | {c["train_loss"]:.5f} | {c["dev_loss"]:.5f} | {c["dev_gauc"]:.5f} |')
    lines += ['', '## 限定场景、用户和视频范围','','random为随机展示；standard_random_tabs是常规展示中随机展示涉及的场景；继续加users、videos表示进一步限制到随机展示中出现过的用户和视频。限制到同一集合不保证记录数量和内容分布完全相同。','',
              '| 数据范围 | 曝光数 | 用户数 | 正例率 | B0 GAUC | B1 GAUC | B2 GAUC | B1−B0百分点及95%区间 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,g in result['groups'].items():
        m=g['models']; d=g['B1_minus_B0']; lo,hi=d['ci95']
        lines.append(f'| {name} | {g["rows"]} | {g["users"]} | {g["positive_rate"]:.2%} | {m["B0"]["gauc"]:.5f} | {m["B1"]["gauc"]:.5f} | {m["B2"]["gauc"]:.5f} | {d["mean_difference"]*100:+.3f} [{lo*100:.3f}, {hi*100:.3f}] |')
    lines += ['', '## 预测概率','','| 数据 | 模型 | 实际正例率 | 平均预测概率 | LogLoss |','| --- | --- | ---: | ---: | ---: |']
    for name,g in result['groups'].items():
        if name.endswith('_random') or name.endswith('standard_all'):
            for m,v in g['models'].items():
                lines.append(f'| {name} | {m} | {g["positive_rate"]:.2%} | {v["mean_probability"]:.2%} | {v["log_loss"]:.5f} |')
    markdown = '\n'.join(lines)+'\n'
    destination = REPORT/'history_diagnosis.md'
    if destination.exists():
        previous = destination.read_text(encoding='utf-8')
        # Keep the written interpretation when regenerating the numeric tables.
        start = previous.find('## 诊断结论')
        end = previous.find('全部结果使用三个相同种子的平均指标')
        if 0 <= start < end:
            markdown = markdown.replace('\n\n', '\n\n'+previous[start:end], 1)
    destination.write_text(markdown,encoding='utf-8')
    print('Saved history_diagnosis.json and .md',flush=True)


if __name__ == '__main__':
    main()
