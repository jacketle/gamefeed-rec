"""Summarize completed history diagnostics; reads aggregate results, never trains.

Figure contract: quantitative grid, seed 2026, all observed epochs in five
matched runs. Training BCE, development BCE and development GAUC answer
different questions. No smoothing, invented repeats or curve uncertainty.
Python/matplotlib, 224 x 178 mm, >=9pt, editable SVG/PDF and PNG preview.
Only typography conventions are reused from the existing report figures.
"""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/kuairand_1k'
OUT=REPORT/'item_l2'
CAPTION='种子2026，展示五组训练的全部已运行轮次。上方分别为训练与开发的预测损失BCE，均不含正则项；下方为开发GAUC。圆环标出各组按开发GAUC选中的轮次。线段连接实际观测值，纵轴范围按指标分别设置。'
CONCLUSION='两次改进尝试完成后，继续采用B0基础模型。屏蔽历史ID降低了排序效果；较强正则带来小幅提高，但未达到追加验证条件。'


def build():
    sources={
        'original':REPORT/'full1000/experiment_results.json',
        'mask':REPORT/'history_mask50/experiment_results.json',
        'mask_comparison':REPORT/'history_mask50/comparison.json',
        'l2_weak':OUT/'l2_0.001/experiment_results.json',
        'l2_strong':OUT/'l2_0.01/experiment_results.json',
        'l2_comparison':OUT/'comparison.json',
    }
    if not all(p.exists() for p in sources.values()):
        return ''
    data={k:json.loads(p.read_text(encoding='utf-8')) for k,p in sources.items()}
    comparison=data['l2_comparison']
    if comparison['worth_repeating']:
        raise ValueError('This case describes a completed screening stop; review changed results')
    specs=[('B0 基础模型','original','B0','#6e7781','s','-'),
           ('B1 历史均值','original','B1','#386f99','o','-'),
           ('B1 屏蔽50%历史ID','mask','B1','#a27d48','^','--'),
           ('B1 正则0.001','l2_weak','B1','#8b6b9a','D',':'),
           ('B1 正则0.01','l2_strong','B1','#287c70','v','-.')]
    runs=[data[key]['training'][f'{mode}_seed2026'] for _,key,mode,*_ in specs]
    base=runs[1]['best_dev']['gauc']
    differences=[None,None,data['mask_comparison']['groups']['all']['masked_minus_original'],
                 comparison['screening']['0.001']['difference'],comparison['screening']['0.01']['difference']]
    rows=[];curve_rows=[]
    for (name,*_),r,d in zip(specs,runs,differences):
        best=max(r['epochs'],key=lambda x:x['dev']['gauc'])
        assert abs(best['dev']['gauc']-r['best_dev']['gauc'])<1e-10
        if d is not None:
            assert abs(d['mean_difference']-(r['best_dev']['gauc']-base))<1e-10
        interval='—' if d is None else f'[{d["ci95"][0]*100:.3f}, {d["ci95"][1]*100:.3f}]'
        delta='—' if d is None else f'{d["mean_difference"]*100:+.3f}'
        rows.append([name,str(best['epoch']),f'{best["dev"]["gauc"]:.5f}',f'{best["dev"]["log_loss"]:.5f}',delta,interval])
        for point in r['epochs']:
            curve_rows.append(dict(model=name,seed=2026,epoch=point['epoch'],train_bce=point['train_loss'],
                                   train_penalty=point.get('train_regularization',0.0),
                                   dev_bce=point['dev']['log_loss'],dev_gauc=point['dev']['gauc']))
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    fig=plt.figure(figsize=(8.8,7.0),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.2])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    for (name,_,_,color,marker,style),r in zip(specs,runs):
        curve=r['epochs'];x=[v['epoch'] for v in curve]
        values=[[v['train_loss'] for v in curve],[v['dev']['log_loss'] for v in curve],[v['dev']['gauc'] for v in curve]]
        for ax,y in zip(axes,values):
            ax.plot(x,y,label=name,color=color,marker=marker,linestyle=style,ms=4,lw=1.6)
        best=max(curve,key=lambda v:v['dev']['gauc'])
        axes[2].scatter([best['epoch']],[best['dev']['gauc']],s=130,facecolors='none',edgecolors=color,lw=1.2)
    for ax,title,ylabel in zip(axes,['a  训练误差继续下降','b  开发误差回升','c  开发排序效果：改动后仍低于B0'],
                               ['训练预测损失（BCE）','开发预测损失（BCE）','开发GAUC']):
        ax.set_title(title,loc='left',fontsize=11,pad=12)
        ax.set_xlabel('训练轮数');ax.set_ylabel(ylabel);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(.8,5.2);ax.grid(axis='y',color='#e5e9ec',lw=.7);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
    fig.savefig(figdir/'history_followup_curves.png',dpi=300,bbox_inches='tight')
    fig.savefig(figdir/'history_followup_curves.svg',bbox_inches='tight')
    fig.savefig(figdir/'history_followup_curves.pdf',bbox_inches='tight')
    plt.close(fig)
    with (figdir/'history_followup_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curve_rows[0]));writer.writeheader();writer.writerows(curve_rows)
    with (OUT/'comparison_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['方案','最佳轮','开发GAUC','开发BCE','相对原B1变化_百分点','95%区间_百分点']);writer.writerows(rows)
    (figdir/'source_manifest.json').write_text(json.dumps({k:{'file':p.relative_to(ROOT).as_posix(),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in sources.items()},indent=2),encoding='utf-8')
    headers=['方案','最佳轮','开发GAUC','开发预测损失','相对原B1变化（百分点）','95%区间（百分点）']
    md_table='| '+' | '.join(headers)+' |\n| --- | ---: | ---: | ---: | ---: | --- |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
    scope='本表统一使用种子2026和同一批1,642,223条常规开发曝光。差值区间按用户配对抽样2,000次，保留每个用户的全部记录；它描述本次训练下的用户抽样波动。原三种子主实验另行报告。'
    steps=[
        ('发现问题','B1训练预测损失从0.45539降至0.40276，开发GAUC从0.68730降至0.68446；B1第一轮也低于B0第一轮的0.69165。因此，后续过拟合是已观察现象，第一轮的模型差距还需要单独解释。'),
        ('检查输入','开发历史中，约34.28%的条目能使用经过训练的ID向量，候选中约30.16%。其余ID使用零向量并保留标签。这个覆盖差异提供了排查方向。'),
        ('尝试一：屏蔽ID','训练时随机屏蔽50%的历史视频ID向量，保留标签、历史长度和有效位置。开发预测损失从0.48154降到0.46849，但GAUC下降0.126个百分点。概率误差改善与组内排序改善是两个目标。'),
        ('尝试二：保留ID并加正则','对当前批次实际使用的视频ID向量加L2惩罚，限制参数过度拟合。0.001和0.01两组都在第一轮达到最佳开发GAUC；0.01提高0.026个百分点，差值区间包含零。'),
        ('最终取舍','事先约定至少提高0.05个百分点且区间下界高于零，才补另外两个种子和同正则的B0。两组均未达到条件，本轮结束，继续采用B0。')]
    formula='L_total = L_BCE + λ × mean(E_used²)'
    text=['# 历史模型效果下降：两次改进与最终取舍','',CONCLUSION,'','## 对照结果','',scope,'',md_table,
          '', '## 训练曲线','', '![历史模型排查训练曲线](figures/history_followup_curves.png)','',CAPTION,
          '', '## 排查过程']
    for title,body in steps:text+=['',f'### {title}','',body]
    text+=['','## 正则化的计算','',r'$$',r'L_{\mathrm{total}}=L_{\mathrm{BCE}}+\lambda\frac{1}{|S|d}\sum_{i\in S}\sum_{j=1}^{d}E_{ij}^{2}',r'$$','',
           'S是本批次去重后的有效视频ID集合，d=16为ID向量维度，E是视频ID向量表，λ为正则强度。S为空时惩罚为0。这个批次内平方均值方法与DIN按频率修正的MBA不同。',
           '训练日志分别记录BCE、加权正则项和总目标；开发预测损失只计算BCE。数据划分、学习率、20条历史、批次大小和提前停止规则保持一致。']
    text+=['','## 数据与复现','',
           '训练3,764,530条曝光；全部实验最多5轮，连续2轮开发GAUC未改善即停止。新实验只使用开发集选择设置；本轮没有追加种子或测试评价。',
           '原三种子主实验的B0开发GAUC均值为0.69490、测试均值为0.69076；上表0.69461是种子2026的开发结果。Pure融合Recall@100属于另一套实验。',
           '运行 `python scripts/report_history_followup.py` 可从原始汇总JSON重建本案例、表格与曲线，无需重新训练。CSV提供全部轮次，SHA-256清单记录来源；五组共有17个轮次点，全部展示。',
           '参考：[DIN原论文，第5.1节和表4](https://arxiv.org/html/1706.06978)。本表数值均来自本项目实验。']
    (OUT/'history_followup_case.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    assert len(curve_rows)==17, 'Review new epochs before updating the case text'
    image='data:image/svg+xml;base64,'+base64.b64encode((figdir/'history_followup_curves.svg').read_bytes()).decode()
    tr=lambda values:'<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
    parts=['<section id="history-followup"><div class="section-head"><h2>历史模型排查：两次尝试，保留B0</h2></div>',
           '<div class="finding"><p>'+CONCLUSION+'</p></div>', '<p>'+scope+'</p>',
           '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>',
           *[tr(r) for r in rows],'</tbody></table></div>',
           '<figure><img class="chart" src="'+image+'" alt="五种设置在种子2026下的训练预测损失、开发预测损失和开发GAUC曲线"><figcaption>'+CAPTION+'</figcaption></figure>']
    for title,body in steps:parts+=['<h3>'+title+'</h3><p>'+body+'</p>']
    parts+=['<details><summary>正则化怎么算？</summary><p><code>'+formula+'</code></p><p>S为本批次去重后的有效视频ID，E_used为对应的16维向量。候选和历史共享向量表；排除补位与未知ID。开发损失只计BCE。采用批次内L2平方均值，DIN原文另含频率修正。</p></details>']
    parts+=['<p>原三种子主实验中，B0测试GAUC均值为0.69076。上表为种子2026的开发对照；Pure召回结果在前文单独报告。</p></section>']
    section='\n'.join(parts)
    (OUT/'history_followup_section.html').write_text(section,encoding='utf-8')
    return section


if __name__=='__main__':
    build()
    print('Updated history follow-up case, table and training curves.')
