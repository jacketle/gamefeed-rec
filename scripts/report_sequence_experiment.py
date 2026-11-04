"""Render the completed 1K experiment from aggregate JSON; no training.

Figure contract: quantitative comparison of history representations. Panel 1
shows all three seeds, panel 2 shows seed-2026 optimization, panel 3 shows
matched scenario differences. No interpolated observations or selected seeds.
Python/matplotlib; 183mm width, minimum 8pt, PNG + editable SVG/PDF.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(folder):
    report=json.loads((folder/'experiment_results.json').read_text(encoding='utf-8'))
    final=report['final_evaluation']; training=report['training']; seeds=(2026,2027,2028)
    modes=('B0','B1','B2'); names={'B0':'基础模型','B1':'历史均值','B2':'候选相关加权'}
    colors={'B0':'#7b8790','B1':'#447a9c','B2':'#a2663c'}
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    out=folder/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/f'{name}.svg',bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,['dev_standard','test_standard'],['a  开发集','b  测试集']):
        for i,m in enumerate(modes):
            values=[r['metrics']['gauc'] for r in final[key]['models'][m]]
            ax.scatter(np.full(3,i)+[-.09,0,.09],values,c=colors[m],s=27)
            ax.plot([i-.2,i+.2],[np.mean(values)]*2,c=colors[m],lw=2)
        ax.set_xticks(range(3),[names[m] for m in modes])
        ax.set_ylabel('按用户加权AUC（GAUC）');ax.set_title(title,loc='left');ax.grid(axis='y',alpha=.2)
    save(fig,'01_three_seed_comparison')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for mode in modes:
        curve=training[f'{mode}_seed2026']['epochs'];x=[r['epoch'] for r in curve]
        axes[0].plot(x,[r['train_loss'] for r in curve],'-o',label=names[mode],c=colors[mode],ms=4)
        axes[1].plot(x,[r['dev']['gauc'] for r in curve],'-o',label=names[mode],c=colors[mode],ms=4)
    for ax in axes:ax.set_xlabel('训练轮数');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('训练二元交叉熵');axes[0].set_title('a  训练误差',loc='left')
    axes[1].set_ylabel('开发集GAUC');axes[1].set_title('b  开发效果',loc='left')
    save(fig,'02_learning_curves')
    test=final['test_standard'];tabs=sorted(test['models']['B1'][0]['tabs'],key=int)
    counts=report['training_tabs']; total=sum(counts.values())
    major={t for t,n in counts.items() if n/total>=.05}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for ax,group,title in zip(axes,[[t for t in tabs if t in major],[t for t in tabs if t not in major]],
                             ['a  训练曝光占比≥5%的场景','b  其余场景（独立纵轴）']):
        kept=[]
        for tab in group:
            pairs=[(a['tabs'][tab]['gauc'],b['tabs'][tab]['gauc']) for a,b in zip(test['models']['B1'],test['models']['B2'])]
            if any(a is None or b is None for a,b in pairs):continue
            delta=np.array([b-a for a,b in pairs])*100;i=len(kept);kept.append(tab)
            ax.scatter(np.full(3,i)+[-.09,0,.09],delta,color=colors['B2'],s=25)
            ax.plot([i-.2,i+.2],[delta.mean()]*2,c=colors['B2'],lw=2)
        ax.axhline(0,c='#777777',ls='--',lw=.8)
        ax.set_xticks(range(len(kept)),[t+'\nn='+str(test['models']['B1'][0]['tabs'][t]['rows']) for t in kept],fontsize=8)
        ax.set_xlabel('场景编号／测试曝光数');ax.set_title(title,loc='left',fontsize=9);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('B2 − B1 的GAUC差（百分点）')
    save(fig,'03_scenario_differences')
    n=report['preparation']['sample_users']; chosen=report['selection']['chosen']
    lines=['# KuaiRand-1K：历史建模实验结果','',f'固定{n}名用户，种子2026、2027、2028；每个模型最多5轮，依据开发GAUC选择检查点。',
        '',f'**开发集选择：{chosen}（{names[chosen]}）。** 先记录选择，再评价测试集；不按测试结果重新调参。',
        '', '## 主要结果','', '| 数据 | 模型 | AUC均值 | GAUC均值 ± 种子标准差 | LogLoss均值 |', '| --- | --- | ---: | ---: | ---: |']
    for key,title in [('dev_standard','开发常规'),('test_standard','测试常规'),('dev_random','开发随机'),('test_random','测试随机')]:
        r=final[key];s0=r['S0'];lines.append(f'| {title} | S0用户场景倾向 | {s0["auc"]:.5f} | {s0["gauc"]:.5f} | {s0["log_loss"]:.5f} |')
        for m in modes:
            s=r['summary'][m];lines.append(f'| {title} | {m} {names[m]} | {s["auc"]["mean"]:.5f} | {s["gauc"]["mean"]:.5f} ± {s["gauc"]["std"]:.5f} | {s["log_loss"]["mean"]:.5f} |')
    lines+=['','![三种子结果](figures/01_three_seed_comparison.png)','', '每个点是一次训练，横线是三次均值；非集成预测。图中全部种子均展示。','', '## 加权相对均值的差异','', '| 数据 | 曝光数 | 用户数 | 正例率 | GAUC差（百分点） | 用户抽样95%区间 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for key,r in final.items():
        b=r['B2_minus_B1'];lo,hi=b['ci95'];lines.append(f'| {key} | {r["rows"]} | {r["users"]} | {r["positive_rate"]:.2%} | {b["mean_difference"]*100:.3f} | [{lo*100:.3f}, {hi*100:.3f}] |')
    lines+=['','GAUC有效评价范围（所有模型使用同一批标签）：']
    for key,r in final.items():
        m=r['S0'];lines.append(f'- {key}：有效用户{m["valid_users"]}/{m["total_users"]}，有效曝光{m["covered_rows"]}/{m["rows"]}（{m["exposure_coverage"]:.2%}）。')
    lines+=['','按用户配对抽样2,000次，保留用户全部记录；每次先计算三个种子的差，再取均值。区间描述评价用户变化对这三次训练平均差值的影响。',
        '', '## 每个种子与计算成本','', '| 模型／种子 | 最佳轮 | 开发GAUC | 测试GAUC | 总训练秒数 | 峰值显存MiB |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in modes:
        for i,seed in enumerate(seeds):
            r=training[f'{m}_seed{seed}'];best=max(r['epochs'],key=lambda x:x['dev']['gauc']);t=final['test_standard']['models'][m][i]['metrics']
            lines.append(f'| {m}／{seed} | {best["epoch"]} | {best["dev"]["gauc"]:.5f} | {t["gauc"]:.5f} | {sum(e["train_seconds"] for e in r["epochs"]):.1f} | {r["peak_gpu_mib"]:.1f} |')
    lines+=['','| 模型 | 参数量 | 训练进程峰值MiB（种子2026） | 测试批量预测秒数均值 |', '| --- | ---: | ---: | ---: |']
    for m in modes:
        r=training[f'{m}_seed2026'];secs=np.mean([x['prediction_seconds'] for x in final['test_standard']['models'][m]])
        lines.append(f'| {m} | {r["parameters"]:,} | {r["peak_process_mib"]:.1f} | {secs:.2f} |')
    lines+=['','预测耗时覆盖测试常规曝光的分批特征组装、传入GPU、模型前向和取回分数；不包含模型加载与指标计算。批次256，三个种子的耗时取均值；它是批量离线耗时。']
    lines+=['','![学习曲线](figures/02_learning_curves.png)','', '学习曲线显示种子2026的全部已运行轮次；其他两种子逐轮结果在JSON中。','', '## 场景与有效评价覆盖','', '![场景差异](figures/03_scenario_differences.png)','', '按训练曝光占比5%划分两个面板，纵轴分别显示；n为该场景测试曝光数。每个点是一对相同种子的差值，横线为均值。小样本场景的GAUC可能大幅跳动；无有效正负组的场景只在下表报告覆盖。',
        '', '| 场景 | 曝光数 | 正例率 | 有效用户比例 | 有效曝光比例 | B0 GAUC | B1 GAUC | B2 GAUC |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for tab,r in final['test_standard']['models']['B0'][0]['tabs'].items():
        values=[]
        for m in modes:
            v=[x['tabs'][tab]['gauc'] for x in final['test_standard']['models'][m]]
            values.append(f'{np.mean(v):.5f}' if all(x is not None for x in v) else '无有效组')
        lines.append(f'| {tab} | {r["rows"]} | {r["positive_rate"]:.2%} | {r["user_coverage"]:.2%} | {r["exposure_coverage"]:.2%} | '+ ' | '.join(values)+' |')
    lines+=['','## 内容与历史分组','', '| 分组 | 曝光数 | 正例率 | B0 GAUC | B1 GAUC | B2 GAUC |','| --- | ---: | ---: | ---: | ---: | ---: |']
    group_names={'known':'词表内视频','unseen':'词表外视频','0-15s':'视频时长≤15秒','15-60s':'视频时长15—60秒','60s+':'视频时长>60秒','20':'历史20条','empty':'空历史','1-5':'历史1—5条','6-19':'历史6—19条'}
    for field in ['item_groups','duration_groups','history_groups']:
        for name,r in final['test_standard']['models']['B0'][0][field].items():
            values=[]
            for m in modes:
                v=[x[field][name]['gauc'] for x in final['test_standard']['models'][m]]
                values.append(f'{np.mean(v):.5f}' if all(x is not None for x in v) else '无有效组')
            lines.append(f'| {group_names[name]} | {r["rows"]} | {r["positive_rate"]:.2%} | '+ ' | '.join(values)+' |')
    history_groups=final['test_standard']['models']['B0'][0]['history_groups']
    lines+=['','分组词表只使用截至5月1日的常规日志；实际向量训练覆盖另见下表。'+
        ('本批测试记录的历史都已达到20条上限。' if list(history_groups)==['20'] else '历史分组按每条曝光当时可用的历史计算，上限20条。')]
    lines+=['','## 输入覆盖与解释','', '| 模型 | 开发候选词表覆盖 | 开发候选实际训练覆盖 | 开发历史实际训练覆盖 | 开发候选标签覆盖 |', '| --- | ---: | ---: | ---: | ---: |']
    for m in modes:
        c=training[f'{m}_seed2026']['dev_coverage'];lines.append(f'| {m} | {c["candidate_vocabulary"]:.2%} | {c["candidate_trained"]:.2%} | {c["history_trained"]:.2%} | {c["tag_available"]:.2%} |')
    sensitivity=final['test_standard']['feedback_sensitivity']
    lines+=['','## 历史反馈延迟检查','',
        f'测试集有{sensitivity["affected_rows"]}条记录的最近20条历史包含可能尚未结束的观看。用“曝光时间＋记录的播放时长”作为保守完成时间，遮蔽这些历史条目，再用原模型预测。历史长度数值特征保持原值，也不补入更早的历史。',
        '', '| 模型 | 原GAUC均值 | 遮蔽后GAUC均值 | 差值（百分点） |', '| --- | ---: | ---: | ---: |']
    for m in ('B1','B2'):
        before=final['test_standard']['summary'][m]['gauc']['mean']
        after=np.mean([r['metrics']['gauc'] for r in sensitivity['models'][m]])
        lines.append(f'| {m} | {before:.5f} | {after:.5f} | {(after-before)*100:.3f} |')
    lines+=['','这是固定模型的输入敏感性检查；真实反馈上报时间未在日志中给出。',
        '', '## 模型和指标怎么理解','',
        '- S0：根据用户和场景过去的观看达标比例打分。同一个用户、同一个场景下，所有视频分数相同，所以同时有正负例时，该组AUC为0.5。总体GAUC还会受到不同场景比例影响。',
        '- B0：输入用户、候选视频ID和标签、场景、时刻、视频时长及历史长度，使用多层全连接网络（MLP）预测观看达标概率。',
        '- B1：在B0输入基础上，增加最近20条观看达标历史的平均内容向量。',
        '- B2：输入相同，改为让当前候选参与计算每条历史的权重，再加权求和。借鉴DIN（Deep Interest Network，深度兴趣网络），本实现使用softmax归一化权重。',
        '', '$$', r'h_{\mathrm{mean}}=\frac{1}{L}\sum_{j=1}^{L}h_j,\qquad h_{\mathrm{weighted}}=\sum_{j=1}^{L}\alpha_jh_j', '$$',
        '', '$$', r'\alpha_j=\frac{\exp(g(h_j,v,h_j-v,h_j\odot v))}{\sum_{k=1}^{L}\exp(g(h_k,v,h_k-v,h_k\odot v))}', '$$',
        '', 'L是有效历史条数，h是历史视频向量，v是当前候选向量；g是小型神经网络，⊙表示对应位置相乘。补位条目的权重为0，空历史输出全零向量。',
        '', '- AUC：随机取一条观看达标和一条未达标记录，模型把达标记录排在前面的比例，平分计0.5。',
        '- GAUC：先分别计算每个用户的AUC，再按有效用户的曝光记录数加权。只有单一标签的用户不参与计算，覆盖率单独报告。',
        '- LogLoss：二元交叉熵；概率判断越准确，数值通常越小。',
        '- 种子标准差：三次训练结果的波动；用户抽样区间：固定这三次训练，改变评价用户组成时的差值波动。',
        '', '$$', r'\mathrm{GAUC}=\frac{\sum_{u\in U_{\mathrm{valid}}}n_u\mathrm{AUC}_u}{\sum_{u\in U_{\mathrm{valid}}}n_u}', '$$',
        '', 'n是该用户的曝光记录数，U_valid是同时有正负例的用户集合。']
    delta=final['test_standard']['B2_minus_B1'];lo,hi=delta['ci95']
    lines+=['','## 结论与范围','',
        f'1. 预先按开发集选择{names[chosen]}。测试阶段保持该决定；B2相对B1的测试GAUC差为{delta["mean_difference"]*100:.3f}个百分点。',
        '2. '+('配对用户抽样区间全部高于零，这三次训练支持加权优于均值。' if lo>0 else '配对区间未全部高于零，尚未形成加权历史稳定优于均值的证据。'),
        '3. ID覆盖较低时，标签表示承担更多内容区分；词表存在不等于向量得到训练。未参与训练的ID向量统一使用零回退，保留可用标签。',
        f'本轮标签词表为{report["features"]["tags"]}种。结合训练曲线、输入覆盖和场景分组解释差异；这些对照分别描述优化过程、可用信息和适用人群。',
        (f'4. 本轮保留全部{n}名日志用户，包括历史积累期之后首次出现的用户。数据按紧凑类型保存，相同标签组合只解析一次；实际资源占用见上表。' if n==1000 else
         f'4. 本轮范围为固定{n}用户，作为小规模对照；全量结果另行报告。'),
        '5. 日期统一采用time_ms换算的上海日期，原date保留；历史严格早于当前时间。先前反馈视为已可用，潜在时间重叠数量见数据检查。',
        '6. 当前评价的是实际曝光上的观看达标预测，不等同于新召回列表或游戏业务的线上收益。游戏映射未完整验证，未给数字标签随意命名。',
        '7. 可选50／100条历史、全库召回与多任务不属于本轮必做；先保留可解释的主对照结果。',
        '', '## 交付与复现','', '汇总JSON包含三个种子的完整曲线、各场景、已知／未见视频及视频时长分组。',
        '', '本图为定量比较图：三种子表现、训练过程和同场景差值分别回答三个问题。数据源为同目录experiment_results.json；没有删除任何种子。PNG供阅读，SVG/PDF保留可编辑文本。',
        '', '参考：[KuaiRand数据说明](https://kuairand.com/)、[DIN论文](https://arxiv.org/abs/1706.06978)。结果均来自本地实验。',
        '', '```powershell', '$env:PYTHONPATH = Join-Path $PWD src',
        f'python -m gamefeed.sequence_training --config configs/kuairand_1k{"_full" if n==1000 else ""}.yaml --seed 2026',
        '# 同样运行2027、2028后：', f'python -m gamefeed.sequence_evaluation --config configs/kuairand_1k{"_full" if n==1000 else ""}.yaml',
        f'python scripts/report_sequence_experiment.py --report-dir {folder.relative_to(Path(__file__).resolve().parents[1]).as_posix() if folder.is_absolute() else folder.as_posix()}', '```']
    if n==1000 and (folder.parent/'item_l2/history_followup_case.md').exists():
        lines+=['','## 历史模型后续排查（2026-11-04）','',
                '保持原三种子主实验结果，继续采用B0。后续种子2026对照中，屏蔽50%历史ID使开发GAUC下降0.126个百分点；视频ID正则0.01提高0.026个百分点，用户抽样95%区间为[−0.025，0.080]个百分点，未达到追加验证条件。',
                '', '[查看完整排查案例、对照表和训练曲线](../item_l2/history_followup_case.md)。']
    (folder/'experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote report and three figures:',folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=Path('reports/kuairand_1k/pilot100'))
    run(p.parse_args().report_dir)
