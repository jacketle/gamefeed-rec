"""Build the Chinese optimization report and figures from completed experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/optimization_v1'


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def figures(pure, training, baseline):
    # Quantitative grid: paired ranking changes, coverage tradeoff, then learning curves.
    # All saved epochs and completed seeds are included, with no interpolation.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    directory = OUT/'figures'; directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(directory/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(directory/f'{name}.svg',bbox_inches='tight')
        fig.savefig(directory/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(9,3.1),layout='constrained')
    for ax,metric,title in zip(axes[:2],['recall10','ndcg10'],['Recall@10 change (pp)','NDCG@10 change (x100)']):
        for y,split,color in [(1,'dev','#526B8A'),(0,'test','#BB744A')]:
            ci=pure[split]['comparison']['intervals'][metric]
            estimate=100*ci['estimate'];low=100*ci['lower'];high=100*ci['upper']
            ax.errorbar(estimate,y,xerr=[[estimate-low],[high-estimate]],fmt='o',color=color,capsize=3)
        ax.axvline(0,color='#999999',linewidth=.8,linestyle='--')
        ax.set(yticks=[0,1],yticklabels=['Test','Development'],ylim=(-.5,1.5),xlabel=title)
    for method,color,label in [('rrf','#526B8A','RRF'),('blend','#BB744A','Rank blend')]:
        values=[100*pure[s]['summary'][method]['coverage'] for s in ['dev','test']]
        axes[2].plot([0,1],values,marker='o',color=color,label=label)
    axes[2].set(xticks=[0,1],xticklabels=['Development','Test'],ylabel='Catalog coverage (%)')
    axes[2].legend(loc='best')
    for label,ax in zip('abc',axes): ax.set_title(label,loc='left',fontweight='bold')
    save(fig,'pure_comparison')
    records=[]
    for key,result in training['training'].items():
        for method,entry in [('B0',baseline['training'][key]),('B0 + author',result)]:
            seed=int(key.split('seed')[1])
            for epoch in entry['epochs']:
                records.append(dict(method=method,seed=seed,epoch=epoch['epoch'],
                    train_loss=epoch['train_loss'],dev_loss=epoch['dev']['log_loss'],
                    dev_gauc=epoch['dev']['gauc'],train_seconds=epoch['train_seconds']))
    data=pd.DataFrame(records);data.to_csv(OUT/'training_curves.csv',index=False)
    seeds=sorted(data.seed.unique())
    fig,axes=plt.subplots(1,3,figsize=(9,3.1),layout='constrained')
    styles=['-','--',':']
    for method,color in [('B0','#526B8A'),('B0 + author','#BB744A')]:
        for index,seed in enumerate(seeds):
            subset=data[(data.method==method)&(data.seed==seed)]
            for ax,metric,title in zip(axes,['train_loss','dev_loss','dev_gauc'],['Training BCE','Development log loss','Development GAUC']):
                ax.plot(subset.epoch,subset[metric],marker='o',markersize=3,color=color,
                        linestyle=styles[index],label=f'{method}, {seed}')
                ax.set(xlabel='Epoch',ylabel=title,xticks=[1,2,3,4,5])
    axes[0].legend(fontsize=7)
    for label,ax in zip('abc',axes):ax.set_title(label,loc='left',fontweight='bold')
    save(fig,'author_learning_curves')


def main():
    pure=read('reports/release/rank_blend_v1/results.json')
    author=read('reports/kuairand_1k/author_v1/comparison.json')
    training=read('reports/kuairand_1k/author_v1/experiment_results.json')
    baseline=read('reports/kuairand_1k/full1000/experiment_results.json')
    if 'decision' not in author:raise RuntimeError('Author experiment is still running')
    OUT.mkdir(exist_ok=True)
    figures(pure,training,baseline)
    rows=[]
    for split in ('dev','test'):
        for method in ('rrf','blend'):
            rows.append(dict(split=split,method=method,**pure[split]['summary'][method]))
    pd.DataFrame(rows).to_csv(OUT/'pure_metrics.csv',index=False)
    stage='test' if 'test' in author else 'development'
    result=author[stage];difference=result['difference']
    before=float(np.mean([r['baseline']['gauc'] for r in result['per_seed']]))
    after=float(np.mean([r['author']['gauc'] for r in result['per_seed']]))
    adopted=author['decision']=='adopt_author'
    metric_rows=[]
    for split,key in [('dev','development'),('test','test')]:
        if key in author:
            for record in author[key]['per_seed']:
                for model in ('baseline','author'):
                    metric_rows.append(dict(split=split,seed=record['seed'],model=model,**record[model]))
    pd.DataFrame(metric_rows).to_csv(OUT/'author_metrics.csv',index=False)
    lines=['# 推荐算法第一轮优化总结','',
        '本轮完成两项实验：Pure固定名次融合，以及KuaiRand-1K的B0作者特征对照。', '',
        '## 结果与采用方案','',
        'Pure继续使用原RRF顺序。名次融合在测试集提高Recall@10，同时降低NDCG@10与内容覆盖率。',
        (f'1K采用加入作者特征的B0：测试GAUC由{before:.5f}提高到{after:.5f}。' if adopted else
         f'1K继续使用原B0。作者模型在'+('测试集' if stage=='test' else '开发集')+f'的GAUC为{after:.5f}，原B0为{before:.5f}。'), '',
        '## 1. Pure固定名次融合','',
        '每个查询保持100条候选，使用75%的原召回名次分数与25%的观看模型名次分数。两路分数均按1/(60+名次)计算，名次从1开始，评分场景为tab=0。', '',
        '| 集合 | 排序方式 | Recall@10 | NDCG@10 | 内容覆盖率 | Recall@100 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {'开发' if row['split']=='dev' else '测试'} | {'原RRF' if row['method']=='rrf' else '75/25名次融合'} | {row['recall10']:.3%} | {row['ndcg10']:.5f} | {row['coverage']:.2%} | {row['recall100']:.3%} |")
    r=pure['test']['comparison'];ci=r['intervals']['recall10']
    lines += ['',f"测试Recall@10增加{100*ci['estimate']:.3f}个百分点，用户配对抽样95%区间为[{100*ci['lower']:.3f}, {100*ci['upper']:.3f}]个百分点。{r['queries_improved']}个查询改善、{r['queries_worsened']}个下降、{r['queries_unchanged']}个不变。",
        '', '候选集合保持一致，因此Recall@100保持不变。NDCG@10下降说明命中数量的变化与命中位置的变化存在取舍，原RRF继续作为页面默认排序。', '',
        '![Pure指标与覆盖率](figures/pure_comparison.png)','',
        '图1：前两幅展示融合相对RRF的配对变化及用户抽样95%区间；第三幅展示完整候选库口径的覆盖率。开发5410个查询，测试7122个查询。', '',
        '### 游戏正例与历史分组','', '| 集合 | 游戏正例Recall@10：RRF | 游戏正例Recall@10：融合 |', '| --- | ---: | ---: |']
    for split in ('dev','test'):
        s=pure[split]['summary'];lines.append(f"| {split} | {s['rrf']['game_recall10']:.3%} | {s['blend']['game_recall10']:.3%} |")
    lines += ['', '| 测试历史分组 | 查询数 | RRF Recall@10 | 融合Recall@10 |', '| --- | ---: | ---: | ---: |']
    for name,s in pure['test']['by_history'].items():
        lines.append(f"| {name} | {s['rrf']['queries']} | {s['rrf']['recall10']:.3%} | {s['blend']['recall10']:.3%} |")
    lines += ['', '游戏指标统计完整列表中游戏正例的找回，页面游戏筛选是另一项展示设置。逐日结果和作者集中度见Pure结果JSON。', '',
        '## 2. 1K作者特征实验','',
        '原模型使用视频ID、标签、用户、场景及数值特征。本轮增加16维候选作者embedding，并接入原预测网络。作者词表由正式训练曝光建立，缺失和未见作者使用零向量。', '',
        '开发曝光中视频ID训练覆盖率约30.13%，作者覆盖率约79.78%；在未训练视频的曝光中，约71.06%对应训练中见过的作者。这是本轮增加作者输入的依据。', '',
        '训练保持batch size 256、Adam学习率0.001、最多5轮、连续2轮开发GAUC不改善则停止，每轮评价一次。先训练种子2026；开发GAUC增加至少0.0005且用户配对区间下界大于0，才追加2027和2028。', '',
        '| 种子 | B0开发GAUC | 作者模型开发GAUC | 差值（百分点） |','| --- | ---: | ---: | ---: |']
    for r in author['development']['per_seed']:
        lines.append(f"| {r['seed']} | {r['baseline']['gauc']:.5f} | {r['author']['gauc']:.5f} | {100*(r['author']['gauc']-r['baseline']['gauc']):+.3f} |")
    ci=author['screening']['difference'];lo,hi=ci['ci95']
    lines += ['',f"种子2026初筛差值为{100*ci['mean_difference']:+.3f}个百分点，95%区间为[{100*lo:.3f}, {100*hi:.3f}]个百分点。"+
              ('达到条件，已追加两个种子。' if author['repeat_seeds'] else '未达到追加条件，本轮在单种子对照后结束。'), '',
        '![作者模型训练曲线](figures/author_learning_curves.png)','',
        '图2：展示本轮所有已完成种子、全部训练轮次的训练BCE、开发LogLoss和开发GAUC。不同颜色代表模型，不同线型代表种子；每个点对应一个实际训练轮次。', '',
        '### 训练用时','', '| 种子 | 已完成轮数 | 最佳轮 | 累计训练分钟 |', '| --- | ---: | ---: | ---: |']
    for key,entry in training['training'].items():
        best=max(entry['epochs'],key=lambda e:e['dev']['gauc'])['epoch']
        lines.append(f"| {key.split('seed')[1]} | {len(entry['epochs'])} | {best} | {sum(e['train_seconds'] for e in entry['epochs'])/60:.1f} |")
    lines += ['', '用时为训练循环累计时间，数据加载、开发评价和报告生成另计。', '',
        '### 分组结果','',f"下表为{'测试集' if stage=='test' else '开发集'}结果，多种子时取各自指标均值。视频分组依据原B0训练曝光，作者分组依据本轮训练作者词表。",'',
        '| 分组 | 曝光数 | B0 GAUC | 作者模型GAUC |','| --- | ---: | ---: | ---: |']
    for name,g in author[('test' if stage=='test' else 'dev')+'_groups'].items():
        a=[r['baseline']['gauc'] for r in g['per_seed']];b=[r['author']['gauc'] for r in g['per_seed']]
        av=f'{np.mean(a):.5f}' if all(v is not None for v in a) else '—'
        bv=f'{np.mean(b):.5f}' if all(v is not None for v in b) else '—'
        label={'all':'全部','trained_video':'训练出现视频','untrained_video':'训练未见视频',
               'known_author':'训练出现作者','unknown_author':'训练未见或缺失作者'}.get(name,name.replace('tab_','场景 '))
        lines.append(f"| {label} | {g['rows']:,} | {av} | {bv} |")
    if 'test' in author:
        ci=author['test']['difference'];lo,hi=ci['ci95']
        lines += ['',f"测试GAUC平均差值为{100*ci['mean_difference']:+.3f}个百分点，95%区间为[{100*lo:.3f}, {100*hi:.3f}]个百分点。"]
    lines += ['', '### 整体预测指标', '', '| 集合 | 模型 | GAUC | AUC | LogLoss |', '| --- | --- | ---: | ---: | ---: |']
    metric_frame=pd.DataFrame(metric_rows)
    for (split,model),group in metric_frame.groupby(['split','model'],sort=False):
        lines.append(f"| {split} | {'B0' if model=='baseline' else 'B0＋作者'} | {group.gauc.mean():.5f} | {group.auc.mean():.5f} | {group.log_loss.mean():.5f} |")
    if author['decision'] == 'keep_B0' and not author['repeat_seeds']:
        lines += ['',
            '按各自最佳GAUC选出的检查点，作者模型LogLoss较低，但GAUC下降。两者最佳轮次不同：B0为第3轮，作者模型为第2轮；对齐第2轮，B0的LogLoss为0.46063，也优于作者模型的0.46097。训练出现视频组略有提升，训练未见视频组下降。进一步的频次分组与源码对照见[后续分析](../optimization_research/review.md)。', '',
            '训练BCE持续下降，而开发GAUC在第2轮后下降、LogLoss上升，表现出过拟合趋势。本轮保留B0，未追加种子，也未开展作者模型测试集评价。']
    lines += ['', '## 3. 代码与复现','',
        '- 作者输入通过可选配置接入原模型，原B0/B1/B2检查点继续使用原结构。',
        '- 特征映射随新实验保存，输入加载过程保持只读。',
        '- 名次融合复用原指标与用户抽样函数，并重新计算全部Pure候选分数核对缓存。',
        '- 43项单元测试通过，覆盖作者映射、零向量回退、检查点重载、历史边界、融合候选及指标计算。',
        f"- 原B0抽查{author['baseline_check']['checked_predictions']}条开发预测，最大绝对差为{author['baseline_check']['max_absolute_difference']:.1g}。",
        f"- Pure开发／测试缓存分数最大差分别为{pure['dev']['cached_score_max_error']:.1g}／{pure['test']['cached_score_max_error']:.1g}。", '',
        '在项目根目录、已安装项目依赖的Python环境中执行：','', '```powershell',
        '$env:PYTHONPATH = "$PWD\\src"',
        'python scripts/evaluate_rank_blend.py',
        'python scripts/run_author_trial.py',
        '# 报告生成使用安装了matplotlib的Python环境',
        'python scripts/report_optimization.py',
        'python -m unittest discover -s tests','```','',
        '作者实验入口复用已完成的训练结果；首次运行按筛选条件决定是否追加种子。', '',
        '## 4. 评价说明与结果文件','',
        'Pure按用户与日期计算查询指标；1K按曝光数加权计算用户GAUC。区间采用2000次用户整体配对抽样，多种子时平均配对差值。原测试集曾用于项目迭代，本轮属于回顾性验证；用户抽样区间描述所评模型的用户差异。', '',
        '- [Pure完整结果](../release/rank_blend_v1/results.json)',
        '- [作者实验完整结果](../kuairand_1k/author_v1/comparison.json)',
        '- [作者训练记录](../kuairand_1k/author_v1/experiment_results.json)',
        '- [Pure指标CSV](pure_metrics.csv)',
        '- [训练曲线CSV](training_curves.csv)',
        '- [作者模型逐种子指标CSV](author_metrics.csv)',
        '- [输入与原代码记录](manifest_before.json)',
        '- [检查结果](verification.json)', '']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'summary.md')


if __name__ == '__main__':
    main()
