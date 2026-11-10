# -*- coding: utf-8 -*-
"""Build a self-contained HTML experiment report from existing results.

Python / matplotlib only for scientific figures. No new model fitting.
One observed run (seed 2026); no invented uncertainty intervals.
"""
import base64
import csv
import hashlib
import html
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/release'
OUT = REPORT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
DEST = ROOT / 'deliverables/GameFeed_实验结果图解.html'
SOURCES = ['release_summary.json', 'fusion_selection.json', 'selected_test_group_analysis.json', 'demo_benchmark.json', 'semantic_audit/audit.json', 'final_evaluation/results.json', 'interpretation_checks/results.json']
data = {name: json.loads((REPORT/name).read_text(encoding='utf-8')) for name in SOURCES}
SEQUENCE_REPORT = ROOT / 'reports/kuairand_1k/pilot100/experiment_results.json'
for sequence_path in [ROOT/'reports/kuairand_1k/full1000/experiment_results.json', SEQUENCE_REPORT]:
    if sequence_path.exists():
        sequence_data = json.loads(sequence_path.read_text(encoding='utf-8'))
        if len(sequence_data.get('final_evaluation', {})) == 4:
            SEQUENCE_REPORT = sequence_path
            data['kuairand_1k'] = sequence_data
            break
OPTIMIZATION_REPORT = ROOT / 'reports/optimization_v2/sequence/comparison.json'
if OPTIMIZATION_REPORT.exists():
    optimization_data = json.loads(OPTIMIZATION_REPORT.read_text(encoding='utf-8'))
    if optimization_data.get('confirmation') and optimization_data.get('test'):
        data['kuairand_1k_optimization'] = optimization_data
s = data['release_summary.json']
trials = data['fusion_selection.json']['trials']
history = data['selected_test_group_analysis.json']['history_groups']
bench = data['demo_benchmark.json']
final_eval = data['final_evaluation/results.json']
checks = data['interpretation_checks/results.json']
questions = json.loads((ROOT/'docs/report_questions.json').read_text(encoding='utf-8'))
METHODS = ['popularity', 'itemcf', 'semantic', 'selected']
NAMES = ['热门', 'ItemCF', 'BGE＋历史均值', '选定融合']
COLORS = ['#a6afb9', '#718da8', '#c79d6d', '#176657']
INK = '#1b2e38'
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
    'font.size':9, 'axes.titlesize':10, 'axes.labelsize':9, 'xtick.labelsize':8,
    'ytick.labelsize':9, 'legend.fontsize':8, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':.7, 'axes.edgecolor':'#a5afb8',
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
    'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.unicode_minus':False,
    'savefig.facecolor':'white', 'figure.facecolor':'white'})
records = []

def pct(x):
    return f'{100*x:.2f}%'

def uri(path, mime):
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()

def save(fig, name, book=None):
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight', pad_inches=.16)
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=.16)
    fig.savefig(OUT/f'{name}.png', dpi=300, bbox_inches='tight', pad_inches=.16)
    if book is not None:
        book.savefig(fig, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)

def bars(ax, labels, values, colors, maximum, xlabel, digits=2, suffix='%'):
    ys = np.arange(len(labels))
    ax.barh(ys, values, height=.56, color=colors, edgecolor='white', linewidth=.6)
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum)
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e7ecef', linewidth=.6)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=8)
    for y, value in zip(ys, values):
        ax.text(value + maximum*.018, y, f'{value:.{digits}f}{suffix}', va='center', fontsize=8)

def make_figures():
    with PdfPages(OUT/'GameFeed_实验图表.pdf') as book:
        fig, axes = plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
        for ax, metric, maximum, title in zip(axes,['recall@100','coverage@100'],[32,102],['a  正例进入候选的比例','b  推荐内容覆盖面']):
            values = [100*s['results'][m]['test'][metric] for m in METHODS]
            bars(ax,NAMES,values,COLORS,maximum,{'recall@100':'正例召回率 Recall@100 (%)','coverage@100':'内容覆盖率 (%)'}[metric])
            ax.set_title(title,loc='left',pad=12,fontweight='bold')
            records.extend(dict(figure='01_retrieval',split='test',method=m,metric=metric,value=v/100) for m,v in zip(METHODS,values))
        save(fig,'01_retrieval',book)

        fig, ax = plt.subplots(figsize=(7.2,3.35),layout='constrained')
        labels = ['ItemCF 基线','RRF：语义权重 0','RRF：语义权重 0.2','RRF：语义权重 0.4',
                  '语义召回名额 0 条（选定）','语义召回名额 10 条','语义召回名额 20 条']
        values = [t['metrics']['recall@100']*100 for t in trials]
        bars(ax,labels,values,[COLORS[3] if t['setting']['name']=='quota_0' else COLORS[1] for t in trials],30,'开发集 Recall@100 (%)')
        records.extend(dict(figure='02_fusion',split='dev',method=t['setting']['name'],metric='recall@100',value=v/100) for t,v in zip(trials,values))
        save(fig,'02_fusion',book)

        fig, axes = plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
        for col,(task,label) in enumerate([('long_view','观看达标'),('is_like','点赞')]):
            ax = axes[0,col]
            values = [s['single_rankers'][task]['test']['auc'], s['shared_ranker']['test'][task]['auc']]
            ax.plot(values,[1,0],color='#ccd5da',linewidth=1.5)
            for y,v,c in zip([1,0],values,[COLORS[3],COLORS[1]]):
                ax.scatter(v,y,s=40,color=c,zorder=3)
                ax.text(v+.013,y,f'{v:.4f}',va='center',fontsize=8)
            ax.set(yticks=[1,0],yticklabels=['独立模型','共享底层'],xlim=(.5,.95),ylim=(-.6,1.6),xlabel='AUC（点图轴从0.5开始）')
            ax.set_title(f'{"ab"[col]}  {label} · 测试集',loc='left',fontweight='bold',pad=12)
            ax.grid(axis='x',color='#e7ecef'); ax.tick_params(axis='y',length=0)
            ax.spines['left'].set_visible(False)
            records.extend(dict(figure='03_rankers',split='test',method=m,metric=task+'_auc',value=v) for m,v in zip(['single','shared'],values))
            ax = axes[1,col]
            single = [r['dev_auc'] for r in s['single_rankers'][task]['epochs']]
            shared = [r['dev_long_auc' if task=='long_view' else 'dev_like_auc'] for r in s['shared_ranker']['epochs']]
            ax.plot([1,2,3],single,'o-',color=COLORS[3],label='独立模型',markersize=4)
            ax.plot([1,2,3],shared,'s--',color=COLORS[1],label='共享底层',markersize=4)
            ax.set(xticks=[1,2,3],ylim=(.65,.9),xlabel='训练轮次（epoch）',ylabel='开发集 AUC')
            ax.set_title(f'{"cd"[col]}  {label} · 学习过程',loc='left',fontweight='bold',pad=12)
            ax.grid(axis='y',color='#e7ecef'); ax.legend(loc='lower right',frameon=False)
            for method,curve in [('single',single),('shared',shared)]:
                records.extend(dict(figure='03_rankers',split='dev',method=method,metric=f'{task}_auc_epoch_{e}',value=v) for e,v in enumerate(curve,1))
        save(fig,'03_rankers',book)

        fig, axes = plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
        fit = s['validation']['small_fit']
        vals = [fit['initial_loss'],fit['final_loss']]
        axes[0].scatter([0,1],vals,s=55,c=[COLORS[1],COLORS[3]],zorder=3)
        axes[0].set(yscale='log',ylim=(.00005,10),xlim=(-.6,1.6),xticks=[0,1],xticklabels=['训练前','150步后'],ylabel='训练损失（对数坐标）')
        for x,v in enumerate(vals):
            axes[0].annotate(f'{v:.6f}',(x,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
        axes[0].grid(axis='y',color='#e7ecef')
        axes[0].set_title('a  100条样本拟合检查',loc='left',fontweight='bold',pad=12)
        values = [s['validation'][m]['retrieval']['dev']['recall@100']*100 for m in ['with_history','without_history']]
        bars(axes[1],['有历史','去掉历史'],values,[COLORS[3],COLORS[1]],6,'开发集 Recall@100 (%)')
        axes[1].set_title('b  历史特征对照',loc='left',fontweight='bold',pad=12)
        records.extend(dict(figure='04_diagnostics',split='train_100',method='two_tower',metric=m,value=v) for m,v in zip(['initial_loss','final_loss'],vals))
        records.extend(dict(figure='04_diagnostics',split='dev',method=m,metric='recall@100',value=v/100) for m,v in zip(['with_history','without_history'],values))
        save(fig,'04_diagnostics',book)

        fig, axes = plt.subplots(1,2,figsize=(7.2,2.85),layout='constrained')
        game = [s['groups'][m]['game']['macro_recall']*100 for m in METHODS]
        unseen = [s['groups'][m]['unseen']['hits'] for m in METHODS]
        bars(axes[0],NAMES,game,COLORS,38,'游戏正例 Recall@100 (%)')
        axes[0].set_title('a  游戏组：905个评估样本',loc='left',fontweight='bold',pad=12)
        bars(axes[1],NAMES,unseen,COLORS,3,'20个训练中未出现的正例：命中数',digits=0,suffix=' / 20')
        axes[1].set_xticks([0,1,2,3])
        axes[1].set_title('b  训练中未出现的内容：20个正例',loc='left',fontweight='bold',pad=12)
        records.extend(dict(figure='05_groups',split='test',method=m,metric='game_macro_recall',value=v/100) for m,v in zip(METHODS,game))
        records.extend(dict(figure='05_groups',split='test',method=m,metric='unseen_hits_of_20',value=v) for m,v in zip(METHODS,unseen))
        save(fig,'05_groups',book)

        fig, ax = plt.subplots(figsize=(7.2,2.4),layout='constrained')
        labels = ['无历史记录 · 47个评估样本','1–5条 · 724个评估样本','6–20条 · 6351个评估样本']
        vals = [r['recall@100']*100 for r in history.values()]
        bars(ax,labels,vals,[COLORS[3]]*3,36,'选定融合 · 测试集 Recall@100 (%)')
        records.extend(dict(figure='06_history',split='test',method='selected',metric=name,value=v/100) for name,v in zip(history,vals))
        save(fig,'06_history',book)

        fusion = final_eval['fusion']['test']
        fig, axes = plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        values = [fusion['improved'], fusion['harmed'], fusion['unchanged']]
        bars(axes[0],['召回率提高','召回率下降','召回率相同'],values,
             [COLORS[3],COLORS[2],COLORS[0]],max(values)*1.2,'评估组数（按用户和日期分组）',digits=0,suffix='')
        axes[0].set_title('a  哪些评估样本发生变化',loc='left',fontweight='bold',pad=12)
        for y,(name,ci) in enumerate([('总体',fusion['paired_user_interval']),('游戏正例',fusion['game']['paired_user_interval'])]):
            mid,lo,hi = [100*ci[k] for k in ('estimate','lower','upper')]
            axes[1].errorbar(mid,y,xerr=[[max(0,mid-lo)],[max(0,hi-mid)]],fmt='o',color=COLORS[3],capsize=4)
            axes[1].text(mid,y+.18,f'{mid:+.3f} [{lo:+.3f}, {hi:+.3f}]',ha='center',fontsize=7)
            records.append(dict(figure='07_fusion_attribution',split='test',method=name,metric='recall100_delta',value=ci['estimate']))
        axes[1].axvline(0,color='#88939b',linestyle='--',linewidth=.8)
        axes[1].set(yticks=[0,1],yticklabels=['总体','游戏正例'],ylim=(-.5,1.6),xlabel='融合 − ItemCF（百分点）')
        axes[1].set_title('b  按用户重复抽样95%区间',loc='left',fontweight='bold',pad=12)
        save(fig,'07_fusion_attribution',book)

        stages=final_eval['lists']['test']['stages']
        stage_ids=['recall','ranked','author_cap','author_and_caption']
        labels=['融合RRF顺序','模型排序','限制同作者数量','去除重复文案']
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
        for ax,metric,title in zip(axes.flat,['recall10','ndcg10','repeat_author_ratio','repeat_caption_ratio'],
                ['a  正例召回率 Recall@10','b  排序质量 NDCG@10','c  同作者重复占比','d  重复文案占比']):
            values=[stages[x][metric]*100 for x in stage_ids]
            bars(ax,labels,values,[COLORS[1],COLORS[2],COLORS[0],COLORS[3]],max(max(values)*1.28,.1),
                 'NDCG × 100' if metric=='ndcg10' else '比例 (%)',digits=3,suffix='' if metric=='ndcg10' else '%')
            ax.set_title(title,loc='left',fontweight='bold',pad=12)
            records.extend(dict(figure='08_final_lists',split='test',method=x,metric=metric,value=stages[x][metric]) for x in stage_ids)
        save(fig,'08_final_lists',book)

    for step in range(5):
        fig, ax = plt.subplots(figsize=(7.2,1.65))
        ax.set(xlim=(0,10),ylim=(0,2)); ax.axis('off')
        labels = ['此前观看记录\n最近20条达标记录','召回候选\n每种方法100条','融合与去重\n热门10条＋补齐','预测行为\n观看达标、点赞','限制同作者数量\n最多返回10条']
        for i,label in enumerate(labels):
            active = i==step
            ax.add_patch(FancyBboxPatch((i*2+.06,.55),1.78,.9,boxstyle='round,pad=0.04,rounding_size=0.07',
                facecolor=COLORS[3] if active else '#f1f5f7',edgecolor=COLORS[3] if active else '#c9d4db',linewidth=1))
            ax.text(i*2+.95,1,label,ha='center',va='center',fontsize=9,color='white' if active else INK,linespacing=1.7)
            if i<4:
                ax.annotate('',xy=(i*2+2.02,1),xytext=(i*2+1.86,1),arrowprops=dict(arrowstyle='->',color='#70818a',lw=1))
        fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.97)
        save(fig,f'flow_{step}')

    fig, ax = plt.subplots(figsize=(7.2,4.25))
    ax.set(xlim=(0,10),ylim=(0,8.4)); ax.axis('off')
    def box(x,y,width,label,color='#eef3f6'):
        ax.add_patch(FancyBboxPatch((x,y),width,.8,boxstyle='round,pad=0.03',
            facecolor=color,edgecolor='#aabdc8',linewidth=.8))
        ax.text(x+width/2,y+.4,label,ha='center',va='center',fontsize=9)
    def arrow(x,y1,y2):
        ax.annotate('',xy=(x,y2),xytext=(x,y1),arrowprops=dict(arrowstyle='->',color='#617987',lw=1))
    ax.text(2.5,8.1,'用户塔',ha='center',fontsize=12,fontweight='bold')
    ax.text(7.5,8.1,'物品塔',ha='center',fontsize=12,fontweight='bold')
    box(.3,6.8,4.4,'用户ID 16维 ＋ 历史均值64维\n＋ 历史长度1维')
    box(5.3,6.8,4.4,'内容ID 32维 ＋ 标签均值16维\n＋ 时长1维')
    for x,label in [(2.5,'拼接：81维'),(7.5,'拼接：49维')]:
        arrow(x,6.75,6.3);box(x-2.2,5.45,4.4,label)
        arrow(x,5.4,4.95);box(x-2.2,4.1,4.4,'全连接128维 → ReLU → Dropout')
        arrow(x,4.05,3.6);box(x-2.2,2.75,4.4,'全连接64维 → L2归一化','#dceee7')
    for x in [2.5,7.5]:
        ax.annotate('',xy=(5,1.8),xytext=(x,2.7),arrowprops=dict(arrowstyle='->',color='#617987',lw=1))
    box(3,1,4,'点积匹配 → 组内交叉熵','#dceee7')
    ax.text(5,.3,'历史中的每条内容，也通过右侧同一个物品编码器；用mask排除PAD后取均值。',
        ha='center',fontsize=8,color='#526672')
    fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.03)
    save(fig,'two_tower_structure')

FORMULAS = [
    ('recall', 'Recall@100：当天观看达标的内容找回了多少', r'$\mathrm{Recall@100}=\frac{1}{|Q|}\sum_{q\in Q}\frac{|R_{q,100}\cap P_q|}{|P_q|}$',
     'Q是评估组的集合，每组对应一名用户在一天内的记录；P是该组观看达标的不同内容；R是推荐的前100条。当天观看达标4条、命中1条，这组召回率就是25%，再对所有组取平均。'),
    ('dot', '双塔：用向量方向衡量匹配', r'$s(u,i)=\hat{\mathbf{u}}^{T}\hat{\mathbf{v}}_i,\qquad \hat{\mathbf{v}}=\frac{\mathbf{v}}{\|\mathbf{v}\|_2}$',
     'u是用户向量，v是内容向量；除以向量长度后，点积等于余弦相似度。当前两侧都是64维。'),
    ('softmax', '双塔损失：让正例超过抽到的负例', r'$L=-\log\frac{e^{s(u,i^+)/\tau}}{e^{s(u,i^+)/\tau}+\sum_{j\in N_u}e^{s(u,j)/\tau}}$',
     'i⁺是观看达标正例；N是最多4条曝光负例；温度τ=0.1。分母比较同一组候选，正例分数越突出，损失越小。'),
    ('sigmoid', 'Sigmoid：把模型分数变成0到1', r'$p=\sigma(z)=\frac{1}{1+e^{-z}}$',
     'z是网络原始输出，称为logit。z=0时预测概率p=0.5；z越大，p越接近1。'),
    ('bce', '二分类与多任务损失', r'$\mathrm{BCE}(y,p)=-y\log p-(1-y)\log(1-p)$',
     '真实标签y为0或1，p为预测概率。共享模型总损失为L观看达标＋λ×L点赞，λ=1。训练使用BCEWithLogitsLoss计算。'),
    ('rrf', 'RRF：按名次融合多路结果', r'$\mathrm{RRF}(i)=\sum_{c\in C_i}\frac{w_c}{60+\mathrm{rank}_c(i)}$',
     'Cᵢ是把内容i召回的路线，rank从1开始。当前权重为ItemCF 0.8、热门0.2、语义0.2；先确定100条集合，再用这个分数排列。'),
    ('itemcf', 'ItemCF：共同观看达标的用户越多，内容越相近', r'$s(i,j)=\frac{\sum_{u\in U_i\cap U_j}1/\log(1+|I_u|)}{\sqrt{|U_i|\,|U_j|}}$',
     'Uᵢ是观看达标内容i的用户，Iᵤ是用户u观看达标的不同内容。活跃用户降权，分母减弱热门内容用户多的影响。每个内容保留前200个相似邻居。'),
    ('score', '综合得分：同时考虑观看达标与点赞', r'$S=\alpha p_{\mathrm{long}}+(1-\alpha)p_{\mathrm{like}},\qquad \alpha=0.7$',
     'α控制排序时两个目标的比重，越大越重视观看达标。λ则控制训练时两项损失的比重。前面的交互演示可以调整α，观察推荐顺序如何变化。'),
    ('scale', '数值缩放：按训练数据统一尺度', r'$x^{\prime}=\frac{\log(1+x)-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}$',
     '时长和历史长度先log1p，再减训练均值、除训练标准差。开发和测试继续用同一组参数；标准差为0时用1。')]

def make_formulas():
    for name,title,expression,explanation in FORMULAS:
        fig = plt.figure(figsize=(7.2,.85))
        fig.text(.02,.5,expression,fontsize=16,va='center',ha='left',color=INK)
        save(fig,'formula_'+name)

def picture(name, alt, css='chart'):
    path = OUT / (name + '.png')
    width, height = struct.unpack('>II', path.read_bytes()[16:24])
    return f'<img class="{css}" src="{uri(path,"image/png")}" width="{width}" height="{height}" alt="{html.escape(alt)}" loading="lazy">'

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def chart(name, title, caption):
    return '<figure>'+picture(name,title)+'<figcaption><b>'+title+'</b><p>'+caption+'</p></figcaption></figure>'


def final_evaluation_body():
    f=final_eval['fusion']['test']
    ci=f['paired_user_interval']
    stages=final_eval['lists']['test']['stages']
    names={'recall':'融合RRF顺序','ranked':'模型排序','author_cap':'排序＋作者限制','author_and_caption':'排序＋作者限制＋文案去重'}
    direction='提高' if stages['ranked']['recall10']>stages['recall']['recall10'] else '下降'
    parts=['<section id="final-evaluation"><div class="section-head"><h2>提升来自哪里，前10条效果如何</h2></div>',
       chart('07_fusion_attribution','图7｜融合与ItemCF逐组比较。',
             f"比较同一批{f['queries']}组记录。按用户配对抽样2,000次，同一用户的全部日期一起抽取，计算召回率差值的95%区间。"),
       f"<p>融合新增{f['gained']}次正例命中、丢失{f['lost']}次。{f['newly_hit']}组由零命中变成有命中，{f['lost_all']}组丢失全部命中。新增命中中，{f['gained_from_popular_top10']}次来自热门前10条。总体Recall差值为{100*f['delta']:.3f}个百分点，用户重复抽样95%区间为[{100*ci['lower']:.3f}, {100*ci['upper']:.3f}]个百分点。</p>",
       table(['历史长度','评估组数','提高组数','下降组数','Recall差值（百分点）'],[
           [name,g['queries'],g['improved'],g['harmed'],f"{100*g['delta']:+.3f}"] for name,g in f['history_groups'].items()]),
       f"<p>以每名用户各日期的平均变化分类：{f['users_by_mean_change']['improved']}名用户提高，{f['users_by_mean_change']['harmed']}名下降，{f['users_by_mean_change']['unchanged']}名不变。游戏正例组新增{f['game']['gained']}次命中、丢失{f['game']['lost']}次；游戏Recall差值为{100*f['game']['delta']:+.3f}个百分点。每组的正例数不同，命中次数和平均Recall分别统计。</p>",
       '<details><summary>具体哪些用户、哪些日期发生变化？</summary>',
       table(['类型','用户ID','日期','当天正例数','新增命中','丢失命中','组内Recall变化'],[
           [kind,r['user_id'],r['date'][:10],r['positives'],r['gained'],r['lost'],f"{100*r['delta']:+.2f}个百分点"]
           for kind,key in [('提高较大的例子','largest_improvements'),('下降较大的例子','largest_losses')] for r in f[key]]),
       '<p>上表列出召回率变化最大的记录。完整明细见 reports/release/final_evaluation。</p></details>',
       '<h3>开发集：不同方案的前10条对照</h3>',
       table(['方案','开发Recall@10','开发NDCG@10'],[
           [r['label'], f"{r['recall10']:.3%}", f"{r['ndcg10']:.5f}"] for r in checks['dev_top10']]),
       '<p>在相同5,410组记录中，融合的两项指标均最高，Recall@10比热门高约0.045个百分点。融合先选出100条候选，再按RRF得分排序，因此热门内容也按得分决定位置。</p>',
       chart('08_final_lists','图8｜同一批100条候选，最终前10条的变化。',
             '排序时场景设为tab=0，观看达标和点赞权重分别为0.7、0.3。Recall和NDCG按当天日志中全部观看达标内容计算，包括未召回的正例。'),
       f"<p class='takeaway'>模型排序后的测试Recall@10{direction}：{stages['recall']['recall10']:.3%} → {stages['ranked']['recall10']:.3%}。再限制同作者数量、去除重复文案后为{stages['author_and_caption']['recall10']:.3%}。因此默认推荐保留融合后的RRF顺序。</p>"]
    for split in ('dev','test'):
        parts.append('<h3>'+('开发集' if split=='dev' else '测试集')+'前10条结果</h3>')
        parts.append(table(['阶段','Recall@10','NDCG@10','同作者重复占比','重复文案占比','覆盖率','不足10条的组数'],[
            [names[name],f"{v['recall10']:.3%}",f"{v['ndcg10']:.5f}",f"{v['repeat_author_ratio']:.3%}",f"{v['repeat_caption_ratio']:.3%}",f"{v['coverage']:.3%}",v['fewer_than10']]
            for name,v in final_eval['lists'][split]['stages'].items()]))
    parts += ['<details><summary>重复比例怎样计算？</summary><p>同作者重复占比统计每位作者首次出现后的额外条数；每位作者最多2条时，仍允许一次重复。比较文案前合并连续空白；空文案各自保留，缺失作者按不同视频分别计数。</p></details></section>']
    return '\n'.join(parts)

def architecture_body():
    """Keep figure explanations and exports together; reuse the verified figure sources."""
    folder = REPORT / 'architecture'
    entries = json.loads((folder / 'sources.json').read_text(encoding='utf-8'))
    ids = ['arch-fusion', 'arch-tower', 'arch-ranking', 'arch-bge']
    labels = ['候选怎样合并', '双塔怎样计算', '如何预测观看与点赞', '文案怎样表示用户兴趣']
    targets = ['fusion', 'diagnostics', 'ranking', 'retrieval']
    roles = ['默认推荐路径', '召回对照实验', '排序对照实验', '语义召回对照实验']
    pieces = ['<section id="architecture"><div class="section-head"><h2>模型结构与计算过程</h2></div>',
              '<p class="arch-intro">沿箭头查看输入、计算过程和输出。点击图片可放大，图下附有符号解释和代码位置。</p>',
              '<div class="architecture-tabs" role="tablist" aria-label="选择模型结构图">']
    for i, (name, label) in enumerate(zip(ids, labels)):
        pieces.append(f'<button type="button" role="tab" id="tab-{name}" data-arch-tab="{name}" aria-controls="{name}" aria-selected="{str(i == 0).lower()}" tabindex="{0 if i == 0 else -1}">{label}</button>')
    pieces.append('</div>')
    for i, entry in enumerate(entries):
        svg = uri(folder / (entry['stem']+'.svg'), 'image/svg+xml')
        png = uri(folder / (entry['stem']+'.png'), 'image/png')
        title = html.escape(entry['title'])
        pieces.append(f'<article class="arch-panel" id="{ids[i]}" role="tabpanel" aria-labelledby="tab-{ids[i]}"><figure><img class="chart" src="{svg}" alt="{title}"><figcaption><b>结构图 {i+1} · {title}</b><p>{roles[i]}</p></figcaption></figure>')
        pieces.append(f'<div class="arch-caption"><div><h3>这张图怎么读</h3><p>{html.escape(entry["caption"])}</p></div><div class="arch-links"><a href="#{targets[i]}">查看对应实验结果 →</a><a href="{svg}" download="{entry["stem"]}.svg">下载 SVG 矢量图</a><a href="{png}" download="{entry["stem"]}.png">下载 PNG 图片</a></div></div>')
        pieces.append('<details><summary>符号含义与代码位置</summary><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in entry['notes'])+'</ul><p class="code">对应实现：'+html.escape('；'.join(entry['sources']))+'</p></details></article>')
    pieces.append('</section>')
    return '\n'.join(pieces)


def build_body():
    selected = s['selected_test']['recall@100']; base = s['popularity_test']['recall@100']
    cf = s['results']['itemcf']['test']['recall@100']
    audit = data['semantic_audit/audit.json']
    blocks = [f'''<section class="hero" id="overview">
    <h1>GameFeed 推荐实验</h1><p class="lead">从用户观看记录出发，比较候选召回、行为预测和前10条推荐结果。</p>
    <div class="kpis"><div><small>融合方案 · 测试 Recall@100</small><strong>{pct(selected)}</strong><span>7,122组评估记录 · 每组100条候选</span></div>
    <div><small>相比 ItemCF</small><strong>+{100*(selected-cf):.2f}<em>个百分点</em></strong><span>{pct(cf)} → {pct(selected)} · 相对提升{100*(selected/cf-1):.2f}%</span></div>
    <div><small>相比热门推荐</small><strong>+{100*(selected-base):.2f}<em>个百分点</em></strong><span>{pct(base)} → {pct(selected)} · 相对提升{100*(selected/base-1):.2f}%</span></div></div>
    <div class="finding"><b>Pure默认推荐方案</b><p>保留10条热门内容，由ItemCF去重补齐100条，再按RRF得分推荐前10条。开发集上，接入当前排序模型后效果下降，因此默认推荐保留融合顺序。</p></div>
    <div class="reading-paths" aria-label="阅读入口"><a href="#architecture">看模型结构 →</a><a href="#retrieval">查实验结果 →</a><a href="#questions">看技术问答 →</a></div></section>''']
    blocks += ['''<section id="methods"><div class="section-head"><h2>数据与评价方法</h2></div>
    <p>数据来自KuaiRand-Pure，选取5,000名用户、272,354条展示记录。召回检查观看达标的视频有没有进入100条候选；行为预测检查模型能否区分观看达标、点赞的正负样本。</p>
    <div class="two-col"><div><h3>“观看达标”具体怎么算？</h3><p>对应原始字段long_view。视频长度不超过18秒：看完才算达标；视频超过18秒：观看至少18秒算达标。例如10秒的视频看了8秒不达标，60秒的视频看了20秒达标。达标记1，否则记0。</p><p>本项目直接使用数据中的long_view标签。</p></div><div><h3>“评估样本”具体指什么？</h3><p>召回评估按用户和日期分组：张三4月25日算一组，张三4月26日算另一组，李四4月25日再算一组。每组用前一天结束前的历史推荐100条，再检查是否找回当天观看达标的视频。</p><p>下文的7,122组评估样本，均按“用户＋日期”计数；同一用户在不同日期分别计算。</p></div></div>''',
    table(['集合','时间范围（2022年）','视频展示记录数','召回评估样本数（按用户和日期分组）'],[['训练','4/8–4/21','217,873','用于训练'],['开发','4/22–4/28','23,461','5,410组'],['测试','4/29–5/8','31,020','7,122组']]),
    '''<div class="two-col"><div><h3>一条模型样本</h3><p>用户ID＋此前观看达标的视频记录＋一条候选内容＋场景编号 → 观看达标标签、点赞标签。历史取前一天结束前最近20条；当天反馈用作答案。</p></div><div><h3>一次召回评价</h3><p>同一用户在一天内观看达标的不同内容组成正例集合。各方案从相同当天可用库返回100条，保留多正例，统一允许重复推荐历史内容。</p></div></div>
    <details><summary>实验设置与结果适用范围</summary><ul><li>使用standard展示日志；内容从上传日期次日起进入可用库，缺失上传日期时取日志首次出现的次日。</li><li>ItemCF、模型、词表和标准化参数由训练期数据确定；热门与用户历史每天更新到前一天。</li><li>文案和类别来自官方补充表，其中739条属于游戏类别。</li><li>神经网络训练一次，随机种子为2026。按用户重复抽样得到的区间反映评价人群变化，不包含重新训练的波动。</li><li>7种融合设置按开发集选择。测试集曾用于此前迭代，这里的测试结果是已有数据上的比较。</li><li>前10条评价衡量日志中已观察正例的找回能力；未展示内容的反馈未知，新策略的线上收益需要另做实验。</li></ul></details></section>''']
    blocks += [architecture_body()]
    blocks += ['<section id="retrieval"><div class="section-head"><h2>召回效果与内容覆盖率</h2></div><a class="arch-crosslink" href="#arch-bge">回看结构图：文案编码与历史聚合 ↗</a>',
      chart('01_retrieval','图1｜相同候选数下的总体召回对照。','测试集共7,122组记录。左图比较平均召回率，右图比较推荐内容覆盖率。'),
      '<p class="takeaway">热门、ItemCF、融合的Recall@100分别为22.22%、24.63%、25.32%；融合相对ItemCF提高0.69个百分点。融合和ItemCF的覆盖率均为66.95%，热门为1.38%。BGE＋历史均值覆盖81.48%，Recall为3.32%。</p>',
      table(['方案','开发 Recall@100','测试 Recall@50','测试 Recall@100','测试 HitRate@100','测试覆盖率'],[[NAMES[i],pct(s['results'][m]['dev']['recall@100'])]+[pct(s['results'][m]['test'][k]) for k in ['recall@50','recall@100','hitrate@100','coverage@100']] for i,m in enumerate(METHODS)]),
      '<p>正例召回率：先算每组当天观看达标的视频中有多少被找回，再平均。HitRate：至少命中一条正例的组数÷全部评估组数。内容覆盖率：推荐结果出现过的不同视频数÷7,583。</p>',
      '<h3>BGE召回率偏低，检查了哪些原因？</h3><p>本项目用<strong>BGE-small-zh-v1.5编码视频文案</strong>，取CLS位置的输出并归一化，得到512维向量。再将最近20条历史内容的向量取平均，表示用户兴趣。下面依次检查编号对应、编码结果和文本长度。</p>',
      table(['核查项目','实测结果'],[
        ['向量和视频编号','7,583行对应一致；向量均为有限值且长度约为1'],
        ['与官方CLS计算方式对照',f"32条抽样文案，相同预处理后最大绝对误差{audit['official_cls_max_abs_error']:.2g}"],
        ['重新编码与保存向量对照',f"32条抽样文案最小余弦相似度{audit['cache_reencode_min_cosine']:.8f}"],
        ['文本截断数量',f"{audit['truncated_at_128']} / {audit['items']}条文案超过128个文本单元（token）"]]),
      table(['开发集对照（5,410组）','正例召回率 Recall@100'],[
        ['原方案：最多128个token＋历史均值',f"{100*audit['comparisons']['original_128_mean']['recall@100']:.3f}%"],
        ['最大文本长度改为512个token',f"{100*audit['comparisons']['length_512_mean']['recall@100']:.3f}%"],
        ['最多128个token＋提高近期历史权重',f"{100*audit['comparisons']['original_128_recent_weight']['recall@100']:.3f}%"],
        ['每日可用库随机选100条的数学期望',f"{100*checks['random']['dev']['expected_recall_at_100']:.4f}%"]]),
      '<h3>随机推荐基线</h3>',
      table(['集合','评价样本数','每日可用内容数范围','不可用正例出现次数','随机Recall@100期望'],[
          [label, r['queries'], f"{r['min_available_items']}至{r['max_available_items']}", r['unavailable_positive_occurrences'], f"{100*r['expected_recall_at_100']:.4f}%"]
          for split,label in [('dev','开发'),('test','测试')] for r in [checks['random'][split]]]),
      '<p>每组随机Recall期望＝min(100,当天可用内容数)÷当天可用内容数×可用正例数÷全部正例数，然后逐组平均。开发和测试期全部7583条内容都已可用，没有不可用正例，因此结果恰好等于100÷7583。</p>',
      '<p>这两项开发集对照都未提高召回率：单独增加文本长度或提高近期历史权重，效果均接近原方案。下一步可检查<strong>重复文案和不同兴趣混在一起的影响</strong>。</p>',
      '<div class="example"><b>实际抽查：</b>视频6讲“王者荣耀妲己”，相近视频7381、3515也讲妲己，说明主题相似检索有效。视频88和2714都带一大段“评论星座”文案，余弦相似度约0.982；重复宣传文字会影响检索。再把不同兴趣的视频向量平均，可能削弱具体兴趣。可进一步比较清理宣传文字、分别表示不同兴趣后的效果。</div>',
      '</section>']
    blocks += ['<section id="fusion"><div class="section-head"><h2>融合方案如何选择</h2></div>',
      chart('02_fusion','图2｜七种融合设置在开发集上的效果。','开发集共5,410组。各方案返回100条不同内容，优先比较Recall@100，相同时比较Recall@50。'),
      '<p>选定方案先取热门前10条，再从ItemCF依次补齐100条，重复只算一次。随后用RRF排列集合内内容。BGE不单独占候选名额；已经入选的内容若也被BGE召回，会累加其RRF分数。</p>',
      '<p class="takeaway">把语义召回名额从0增加到10、20时，开发Recall依次从24.70%降到23.23%、21.98%。因此保留当前名额分配；测试结果相对ItemCF提高0.69个百分点。</p></section>']
    blocks += [final_evaluation_body()]
    blocks += ['<section id="ranking"><div class="section-head"><h2>观看达标与点赞预测</h2></div>',
      chart('03_rankers','图3｜独立模型与共享底层模型。','上排比较31,020条测试记录的AUC，下排展示3轮训练的开发集AUC。模型使用相同特征、随机种子和批次顺序，每个模型训练3轮。'),
      '<p class="takeaway">独立观看达标／点赞模型测试AUC为0.7303／0.8375，共享模型为0.7271／0.7859。前10条推荐效果见<a href="#final-evaluation">列表对照</a>。</p>',
      table(['任务／模型','测试AUC','测试GAUC','测试AP','测试LogLoss'],[[label+' / '+model]+[f'{r[k]:.4f}' for k in ['auc','gauc','average_precision','log_loss']] for task,label in [('long_view','观看达标'),('is_like','点赞')] for model,r in [('独立',s['single_rankers'][task]['test']),('共享',s['shared_ranker']['test'][task])]]),
      '<h3>GAUC覆盖了哪些用户？</h3>',
      table(['集合／任务','正例率','有效／全部用户','用户覆盖率','有效／全部曝光','曝光覆盖率'],[
          [label+'／'+task_label,pct(r['positive_rate']),f"{r['valid_users']}/{r['users']}",pct(r['user_coverage']),f"{r['covered_rows']}/{r['rows']}",pct(r['row_coverage'])]
          for split,label in [('dev','开发'),('test','测试')] for task,task_label in [('long_view','观看达标'),('is_like','点赞')] for r in [checks['gauc'][split][task]]]),
      '<p>同一用户同时有正例和负例才能计算用户内AUC，GAUC按这些用户的曝光记录数加权。测试点赞正例523/31020；312名有效用户覆盖全部4402名用户的7.09%、曝光记录的11.89%。全负例4078名、全正例12名用户被排除。独立与共享模型使用同一有效人群。</p>',
      '<p>总体AUC会比较不同用户的记录，GAUC则在同一用户内部比较。点赞任务中，两者分别为0.8375和0.6149。判断差距是否来自用户自身的点赞习惯，可增加“只按用户历史点赞率打分”的对照。</p>',
      '<p>独立模型在本次设置下更好。若要查明原因，可以比较模型参数量、任务权重和训练曲线。</p></section>']
    blocks += ['<section id="diagnostics"><div class="section-head"><h2>双塔的拟合检查与历史特征对照</h2></div>',
      chart('04_diagnostics','图4｜小样本拟合与历史消融。','左图比较100条样本训练前后的损失，纵轴采用对数刻度；训练150步后，全部正例排在组内第一。右图比较保留历史和同时移除历史、历史长度的两次训练。'),
      '<p>小样本检查通过，说明这条训练路径能够学习。完整开发集上，有历史4.46%，去历史4.35%，历史增加约0.11个百分点；ItemCF开发Recall为24.38%，因此双塔保留为对照。</p>',
      table(['轮次','有历史：训练损失','有历史：开发损失','去历史：训练损失','去历史：开发损失'],[[i+1]+[f'{s["validation"][m]["epochs"][i][k]:.4f}' for m in ['with_history','without_history'] for k in ['train_loss','dev_loss']] for i in range(3)]),
      '<p>训练每组比较1个正例和最多4个曝光负例，检索时则要从数千条内容中选择。可以通过扩大负例覆盖范围、调整负例难度，检验这种差异对结果的影响。</p></section>']
    blocks += ['<section id="groups"><div class="section-head"><h2>游戏内容与不同历史长度的效果</h2></div>',
      chart('05_groups','图5｜游戏组和训练中未出现的内容。','游戏组有905组记录、1,019个正例；训练未见内容组有20组记录、20个正例。两组均评价原推荐列表的前100条。'),
      '<p>游戏组中，融合29.62%，ItemCF29.91%，热门16.47%。融合方案的总体召回率最高，在游戏组则略低于ItemCF。语义召回找回了20个训练未见正例中的2个，其他方法为0；这组样本较少，适合用来观察具体案例。</p>',
      chart('06_history','图6｜选定融合按历史长度分组。','历史长度分别为0条、1至5条、6至20条，对应47、724和6,351组记录。'),
      '<p>空历史使用热门兜底。短历史组Recall为28.69%，较长历史组24.93%；可以继续检查各组正例数量、重复观看和内容热门程度。</p></section>']
    blocks += ['''<section id="flow"><div class="section-head"><h2>交互演示：从历史记录到推荐列表</h2></div><p>以下用教学样本演示包含模型排序的实验流程。默认推荐在融合后按RRF顺序取前10条。</p>
      <div class="flow-controls"><button id="previous" type="button">上一步</button><button id="next" type="button">下一步</button><button id="play" type="button">自动演示</button><span id="stage-count" aria-live="polite"></span></div>
      <div id="flow-image">'''+picture('flow_0','五阶段推荐流程：历史、召回、融合、排序、作者控制')+'''</div><div class="stage-copy" id="stage-copy" aria-live="polite"></div>
      <h3>亲手改一次排序权重</h3><p>以下六条内容使用教学示例概率。拖动滑块调整观看达标与点赞的权重，勾选作者限制后观察哪些内容入选。</p>
      <label class="slider-label" for="alpha">观看达标权重 α：<output id="alpha-value">0.7</output></label><input id="alpha" type="range" min="0" max="1" step="0.1" value="0.7">
      <label class="check"><input id="cap" type="checkbox"> 同作者最多2条，返回前4条</label><div id="toy-table"></div><p id="toy-explain" class="takeaway" aria-live="polite"></p></section>''']
    blocks += ['<section id="formulas"><div class="section-head"><h2>公式与参数说明</h2></div><div class="formula-grid">']
    for name,title,expression,explanation in FORMULAS:
        blocks.append(f'<article class="formula-card"><h3>{html.escape(title)}</h3>'+picture('formula_'+name,expression,'formula')+f'<p>{html.escape(explanation)}</p></article>')
    blocks += ['</div>',table(['参数','当前值','它控制什么'],[
        ['历史长度','20','保留多少条此前观看达标记录'],['双塔输出维度','64','用户／内容表征长度'],['批次大小 batch_size','128','每次更新读入的样本数'],['学习率 learning_rate','0.001','AdamW每次更新的步幅'],['训练轮数 epochs','3','遍历训练数据的次数'],['随机屏蔽比例 dropout','0.1','训练时随机屏蔽部分神经元'],['温度参数 temperature','0.1','双塔组内分数的集中程度'],['每个正例的负例上限','4','每个观看达标正例最多配的曝光负例'],['候选数量 K','100','每次召回的不同内容数'],['排序权重 α','0.7','演示排序对观看达标目标的重视程度']]),'</section>']
    blocks += ['''<section id="questions"><div class="section-head"><h2>技术问答</h2></div><p>围绕项目实现、实验结果和常见问题展开。标为“扩展知识”的内容用于讨论后续方案。</p>
      <div class="qa-controls"><label for="qa-search">查找问题</label><input id="qa-search" type="search" placeholder="例如：双塔、负例、AUC、冷启动"><button id="expand" type="button">展开全部</button><span id="qa-count" role="status"></span></div>''']
    for q in questions:
        blocks.append(f'<details class="qa" data-search="{html.escape(q["q"]+q["a"])}"><summary><span>Q{q["id"]:02d}</span> {html.escape(q["q"])} <small>{q["tag"]}</small></summary><div class="answer"><p><b>回答：</b>{html.escape(q["a"])}</p><p class="example"><b>举例：</b>{html.escape(q["example"])}</p><details class="code-ref"><summary>查看代码位置</summary><p class="code">{html.escape(q["code"])}</p></details></div></details>')
        if q['id']==3:
            blocks[-1]=blocks[-1].replace('</div></details>','<p><a class="arch-crosslink" href="#arch-tower">结合双塔结构图讲述：81维用户输入、49维视频输入与64维输出 ↗</a></p></div></details>')
    blocks += ['</section><section id="glossary"><div class="section-head"><h2>缩写速查与项目讲述</h2></div>',
    table(['术语','英文全称','直白解释'],[
        ['ItemCF','Item-based Collaborative Filtering','共同看过这些内容的人很多，就认为它们相近'],['BGE','BAAI General Embedding','把文案转成可比较的向量'],['MLP','Multi-Layer Perceptron','多层全连接神经网络'],['BCE','Binary Cross-Entropy','预测0/1标签的常用损失'],['RRF','Reciprocal Rank Fusion','按名次合并多个候选列表'],['AUC','Area Under the ROC Curve','随机正负样本对中正例得分更高的能力'],['GAUC','Group AUC','按用户等分组计算，再加权平均'],['AP','Average Precision','按分数检出正例时的精确程度'],['NDCG','Normalized Discounted Cumulative Gain','重视列表靠前位置的排序质量'],['ANN','Approximate Nearest Neighbor','加速相似向量检索'],['HNSW','Hierarchical Navigable Small World','用分层图进行近似检索'],['MMoE','Multi-gate Mixture-of-Experts','每个任务选择不同专家组合'],['CTR / CVR','Click-Through Rate / Conversion Rate','点击率／转化率，先讲清分母'],['LLM / LTV','Large Language Model / Lifetime Value','大语言模型／用户长期价值'],['PAD / UNK','Padding / Unknown','补位／训练未见ID']]),
    '''<div class="talk"><h3>两分钟讲述顺序</h3><ol><li><b>任务：</b>输入此前观看历史，输出当天推荐列表；使用真实展示、观看达标和点赞数据。</li><li><b>方法：</b>时间切分、热门与ItemCF、语义和双塔对照、观看达标／点赞排序、同作者控制。</li><li><b>结果：</b>热门22.22%、ItemCF24.63%、融合25.32%；融合超过ItemCF0.69个百分点。测试前10条Recall从融合RRF顺序4.71%降至模型排序3.23%，因此默认保留融合候选的RRF顺序。</li><li><b>问题：</b>统一召回过滤、正反馈定义和训练比较条件，修正后重新报告数字。</li><li><b>取舍：</b>总体融合效果最好；游戏组ItemCF略高；语义覆盖更广。根据目标选择方案。</li></ol></div></section>''']
    payload = json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8')
    raw_uri = 'data:application/json;base64,'+base64.b64encode(payload).decode()
    pdf_uri = uri(OUT/'GameFeed_实验图表.pdf','application/pdf')
    blocks += [f'''<section id="sources"><div class="section-head"><h2>结果来源与导出</h2></div><p>下载实验图表和完整数据，或将报告保存为PDF。</p>
      <div class="downloads"><a class="button" href="{pdf_uri}" download="GameFeed_实验图表.pdf">下载Pure图表PDF</a><a class="button secondary" href="{raw_uri}" download="GameFeed_实验结果.json">下载结果JSON</a><button id="print-report" type="button">打印／保存报告PDF</button></div>
      <details><summary>本地耗时测量：环境与计时范围</summary><p>CPU顺序执行，预热5次、测量30次，中位数{bench['median_ms']:.1f}ms，P95为{bench['p95_ms']:.1f}ms。</p><p>计时包括缓存候选读取、两个模型打分、元数据查询和作者限制；不含模型加载、召回生成及HTTP传输。测量版本早于文案去重和默认推荐顺序调整。</p></details>
      <ul class="references"><li><a href="https://github.com/chongminggao/KuaiRand">KuaiRand官方代码与字段说明</a>：数据来源。</li><li><a href="https://kuairand.com/">KuaiRand项目主页</a>：数据版本与标签。</li><li><a href="https://huggingface.co/BAAI/bge-small-zh-v1.5">BGE-small-zh-v1.5模型卡</a>：实际使用的文本编码器。</li><li><a href="https://github.com/facebookresearch/faiss/wiki/Getting-started">Faiss官方入门</a>：向量检索扩展阅读。</li><li><a href="https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/">MMoE论文</a>：任务关系与专家门控扩展阅读。</li></ul>
      <details><summary>报告生成方式与文件位置</summary><p>运行 scripts/build_experiment_html.py 生成本页。实验结果位于 reports/release，训练配置位于 configs/release.yaml，问答位于 docs/report_questions.json。图表和公式随HTML保存，可离线阅读。</p></details></section>''']
    result = '\n'.join(blocks)
    if audit.get('publication_redacted'):
        result = re.sub(r'<details><summary>具体哪些用户、哪些日期发生变化？</summary>.*?</details>', '', result, flags=re.S)
        result = re.sub(r'<div class="example"><b>实际抽查：</b>.*?</div>', '', result, flags=re.S)
    for heading, target, label in [
        ('融合方案如何选择', 'arch-fusion', '候选名额与RRF排序'),
        ('观看达标与点赞预测', 'arch-ranking', '独立模型与共享模型'),
        ('双塔的拟合检查与历史特征对照', 'arch-tower', '双塔输入与历史计算'),
    ]:
        anchor = f'<h2>{heading}</h2></div>'
        result = result.replace(anchor, anchor+f'<a class="arch-crosslink" href="#{target}">回看结构图：{label} ↗</a>')
    return result

def sequence_section():
    """Append completed 1K results using the existing report's table/figure style."""
    r = data.get('kuairand_1k')
    if not r:
        return ''
    names = {'B0':'基础模型', 'B1':'历史均值', 'B2':'候选相关加权'}
    rows = []
    for mode in names:
        dev = r['final_evaluation']['dev_standard']['summary'][mode]
        test = r['final_evaluation']['test_standard']['summary'][mode]
        rows.append([names[mode], f'{dev["gauc"]["mean"]:.5f}',
                     f'{test["gauc"]["mean"]:.5f} ± {test["gauc"]["std"]:.5f}', f'{test["auc"]["mean"]:.5f}'])
    delta = r['final_evaluation']['test_standard']['B2_minus_B1']
    lo, hi = delta['ci95']
    chosen = names[r['selection']['chosen']]
    users = r['preparation']['sample_users']
    train_rows = next(s['rows'] for s in r['preparation']['splits'] if s['split']=='train' and s['random']==0)
    population = f'覆盖KuaiRand-1K全部{users}名日志用户' if users==1000 else f'从KuaiRand-1K历史期用户中固定选取{users}名'
    parts = ['<section id="history-1k"><div class="section-head"><h2>1K历史基线对照：B0、B1与B2</h2></div>',
        f'<p>{population}，保留其全部选定时期记录。使用{train_rows:,}条常规训练曝光，预测观看是否达标。三个模型都使用相同的用户、视频ID、视频标签、场景和时间特征，区别在于怎样使用最近20条历史。</p>',
        '<p>基础模型不输入历史内容；历史均值模型对历史内容向量求平均；候选相关加权模型让当前候选参与历史权重计算。每种模型运行三个配对种子，最多5轮，按开发集保存最佳模型。</p>',
        table(['模型','开发GAUC均值','测试GAUC均值 ± 种子标准差','测试AUC均值'],rows),
        f'<div class="finding"><p>这一历史基线对照的开发集选择：<b>{chosen}</b>。加权相对均值的测试GAUC差为{delta["mean_difference"]*100:.3f}个百分点，按用户配对抽样2,000次的95%区间为[{lo*100:.3f}, {hi*100:.3f}]个百分点。</p></div>']
    for name, caption in [('01_three_seed_comparison','三个点分别对应三个训练种子，横线表示均值。'),
                          ('02_learning_curves','种子2026的训练误差与开发效果；各模型均展示全部已运行轮次。'),
                          ('03_scenario_differences','按训练曝光占比5%分成两个面板，纵轴分别显示。点是相同场景、相同种子的加权减均值GAUC差；n为测试曝光数。')]:
        image_path = SEQUENCE_REPORT.parent / 'figures' / f'{name}.svg'
        parts.append(f'<figure><img class="chart" src="{uri(image_path,"image/svg+xml")}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    c=r['training']['B2_seed2026']['dev_coverage']
    parts.append(f'<p>开发候选中，{c["candidate_trained"]:.1%}的视频ID参与过有效训练，{c["tag_available"]:.1%}具有可用标签。未训练ID使用零向量并保留标签，避免把随机向量误当成已学到的内容特征。</p>')
    report_file=SEQUENCE_REPORT.with_suffix('.md').relative_to(ROOT).as_posix()
    parts.append(f'<p>这轮评价对象是{users}用户的实际曝光记录。Pure部分评价召回列表，两组数字分别解释。完整随机展示、场景、反馈延迟检查和复现步骤见本地 {report_file}。</p></section>')
    return '\n'.join(parts)


def optimization_section():
    """Show completed paired-seed follow-up results alongside historical baselines."""
    r = data.get('kuairand_1k_optimization')
    if not r:
        return ''
    names = {'full_author':'全部作者特征', 'high_author':'高频作者特征',
             'high_author_small':'高频作者与小初始化', 'history_matches':'显式历史匹配'}
    chosen = names[r['selected_variant']]
    adopted = f'B0＋{chosen}' if r.get('decision') == 'prefer_'+r['selected_variant'] else 'B0'
    rows = []
    for label, key in [('开发集', 'confirmation'), ('测试集', 'test')]:
        stage = r[key]
        interval = stage['difference']
        low, high = interval['ci95']
        rows.append([label, f"{np.mean([p['baseline']['gauc'] for p in stage['per_seed']]):.5f}",
                     f"{np.mean([p['candidate']['gauc'] for p in stage['per_seed']]):.5f}",
                     f"{100*interval['mean_difference']:+.3f}", f"[{100*low:+.3f}, {100*high:+.3f}]"])
    parts = ['<section id="optimization-1k"><div class="section-head"><h2>1K后续优化：三种子验证结果</h2></div>',
             f'<div class="finding"><p>本轮优选：<b>{adopted}</b>。以下对照采用相同样本、共享初始权重和评价频率；结果来自种子2026、2027、2028。</p></div>']
    if r['selected_variant'] == 'history_matches':
        parts.append('<p>在B0中增加三项数值：最近20条有效历史中的同作者比例、是否出现过同作者、候选标签与历史标签的匹配比例。它们接入原64维隐藏层，共增加3×64＝192个权重，不增加作者Embedding表。</p>')
    parts.append(table(['集合','B0 GAUC均值',f'B0＋{chosen} GAUC均值','差值（百分点）','用户配对95%区间（百分点）'], rows))
    seed_rows = []
    for seed, dev, test in zip(r['confirmation']['seeds'], r['confirmation']['per_seed'], r['test']['per_seed']):
        seed_rows.append([seed, f"{dev['baseline']['gauc']:.5f}", f"{dev['candidate']['gauc']:.5f}",
                          f"{test['baseline']['gauc']:.5f}", f"{test['candidate']['gauc']:.5f}"])
    parts.append(table(['种子','B0开发GAUC','入选模型开发GAUC','B0测试GAUC','入选模型测试GAUC'], seed_rows))
    parts.append('<p>先用开发集筛选五个设置，再对入选设置补两个配对种子，之后统一评价测试集。区间按用户整体配对抽样2,000次计算，反映所评三次训练下的用户差异。该测试集曾用于项目迭代，本轮为回顾性比较。</p>')
    for name, caption in [('screening_learning_curves','种子2026的全部五组对照；显示全部已完成评价点。'),
                          ('confirmation_learning_curves',f'B0与{chosen}的三个配对种子；显示全部已完成评价点。')]:
        image_path = ROOT/'reports/optimization_v2/figures'/f'{name}.png'
        parts.append(f'<figure><img class="chart" src="{uri(image_path, "image/png")}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    parts.append('<p>详细分组、作者频次实验和复现步骤见 reports/optimization_v2/summary.md。Pure召回与排序继续按前文的独立评价结果使用原RRF顺序。</p></section>')
    return '\n'.join(parts)


def main():
    from report_history_followup import build as build_history_followup
    followup = build_history_followup()
    subprocess.run([sys.executable, str(ROOT/'scripts/build_architecture_figures.py')], check=True)
    make_figures()
    make_formulas()
    with (OUT/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['figure','split','method','metric','value']);writer.writeheader();writer.writerows(records)
    provenance = {name: hashlib.sha256((REPORT/name).read_bytes()).hexdigest() for name in SOURCES}
    provenance['questions_sha256'] = hashlib.sha256((ROOT/'docs/report_questions.json').read_bytes()).hexdigest()
    if 'kuairand_1k' in data:
        provenance['kuairand_1k_sha256'] = hashlib.sha256(SEQUENCE_REPORT.read_bytes()).hexdigest()
    if 'kuairand_1k_optimization' in data:
        provenance['kuairand_1k_optimization_sha256'] = hashlib.sha256(OPTIMIZATION_REPORT.read_bytes()).hexdigest()
    if followup:
        provenance['history_followup_sources'] = json.loads((ROOT/'reports/kuairand_1k/item_l2/figures/source_manifest.json').read_text(encoding='utf-8'))
    (OUT/'source_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    template = (ROOT/'src/gamefeed/report_web/template.html').read_text(encoding='utf-8')
    flow = [uri(OUT/f'flow_{i}.png','image/png') for i in range(5)]
    body = build_body()
    if 'kuairand_1k' in data:
        body = body.replace('<section id="questions">', sequence_section()+followup+'<section id="questions">', 1)
        template = template.replace('<a href="#diagnostics">', '<a href="#history-1k">1K历史基线对照</a><a href="#diagnostics">', 1)
        if followup:
            template = template.replace('<a href="#diagnostics">', '<a href="#history-followup">历史模型排查案例</a><a href="#diagnostics">', 1)
    if 'kuairand_1k_optimization' in data:
        before = '<section id="history-1k">' if 'kuairand_1k' in data else '<section id="questions">'
        body = body.replace(before, optimization_section()+before, 1)
        template = template.replace('<a href="#diagnostics">', '<a href="#optimization-1k">1K优选与三种子验证</a><a href="#diagnostics">', 1)
    document = template.replace('__BODY__',body).replace('__FLOW_IMAGES__',json.dumps(flow))
    web = ROOT/'src/gamefeed/report_web'
    document = document.replace('</style>', (web/'editorial.css').read_text(encoding='utf-8')+'\n</style>', 1)
    document = document.replace('</body>', '<script>'+ (web/'editorial.js').read_text(encoding='utf-8')+'</script></body>')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    DEST.write_text(document,encoding='utf-8')
    print(json.dumps({'html':str(DEST),'bytes':DEST.stat().st_size,'figures':8,'architecture_figures':4,'formulas':len(FORMULAS),'questions':len(questions)},ensure_ascii=False))

if __name__ == '__main__':
    main()
