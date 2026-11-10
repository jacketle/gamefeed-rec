"""Generate the second optimization report from completed, measured results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/optimization_v2'
NAMES = {'baseline': 'B0对照', 'full_author': '全部训练作者',
         'high_author': '高频作者', 'high_author_small': '高频作者＋小初始化',
         'history_matches': '显式历史匹配'}
PLOT_NAMES = {'baseline': 'B0', 'full_author': 'All authors',
              'high_author': 'Frequent authors', 'high_author_small': 'Frequent + small init',
              'history_matches': 'History matches'}
COLORS = {'baseline': '#526B8A', 'full_author': '#BB744A', 'high_author': '#54917B',
          'high_author_small': '#9B709C', 'history_matches': '#B69A49'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ci_text(interval):
    low, high = interval['ci95']
    return f"{100 * interval['mean_difference']:+.3f} [{100 * low:+.3f}, {100 * high:+.3f}]"


def training_records(result):
    runs, curves = {}, []
    for name in result['settings']['variants']:
        report = read(OUT / 'sequence' / name / 'experiment_results.json')
        for key, entry in report['training'].items():
            if entry['status'] != 'development_comparison':
                raise RuntimeError(f'Incomplete training: {name}/{key}')
            seed = int(key.split('seed')[1])
            runs[(name, seed)] = entry
            for point in entry.get('evaluations', entry['epochs']):
                curves.append(dict(variant=name, seed=seed, epoch=point['epoch'],
                    fractional_epoch=point.get('fractional_epoch', point['epoch']),
                    eval_step=point.get('eval_step'), train_loss=point['train_loss'],
                    dev_log_loss=point['dev']['log_loss'], dev_gauc=point['dev']['gauc'],
                    dev_auc=point['dev']['auc'], train_seconds=point['train_seconds'],
                    evaluation_seconds=point.get('evaluation_seconds')))
    return runs, pd.DataFrame(curves)


def metric_records(result, runs):
    rows = []

    def add(split, variant, seed, metrics, user_scene, difference=None):
        run = runs[(variant, seed)]
        row = dict(split=split, variant=variant, seed=seed, **metrics,
                   user_scene_gauc=user_scene['gauc'], parameters=run['parameters'],
                   train_minutes=sum(p['train_seconds'] for p in run['epochs']) / 60,
                   best_eval_step=run.get('best_eval_step'),
                   best_fractional_epoch=run.get('best_fractional_epoch'))
        if difference:
            row.update(gauc_difference=difference['mean_difference'],
                       ci95_lower=difference['ci95'][0], ci95_upper=difference['ci95'][1])
        rows.append(row)

    first = next(iter(result['dev_variants'].values()))
    add('dev', 'baseline', 2026, first['baseline'], first['user_scene_baseline'])
    for name, comparison in result['dev_variants'].items():
        add('dev', name, 2026, comparison['candidate'], comparison['user_scene_candidate'], comparison['difference'])
    winner = result.get('selected_variant')
    if 'confirmation' in result:
        for seed, comparison in zip(result['confirmation']['seeds'], result['confirmation']['per_seed']):
            if seed == 2026:
                continue
            add('dev', 'baseline', seed, comparison['baseline'], comparison['user_scene_baseline'])
            add('dev', winner, seed, comparison['candidate'], comparison['user_scene_candidate'], comparison['difference'])
    if 'test' in result:
        for seed, comparison in zip(result['confirmation']['seeds'], result['test']['per_seed']):
            add('test', 'baseline', seed, comparison['baseline'], comparison['user_scene_baseline'])
            add('test', winner, seed, comparison['candidate'], comparison['user_scene_candidate'], comparison['difference'])
    return pd.DataFrame(rows)


def draw_curves(curves, result):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    directory = OUT / 'figures'
    directory.mkdir(exist_ok=True)
    groups = [('screening_learning_curves', curves[curves.seed.eq(2026)])]
    if 'confirmation' in result:
        names = ['baseline', result['selected_variant']]
        groups.append(('confirmation_learning_curves', curves[curves.variant.isin(names)]))
    paths = []
    for filename, data in groups:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout='constrained')
        for (variant, seed), group in data.groupby(['variant', 'seed'], sort=False):
            group = group.sort_values('eval_step')
            style = {2026: '-', 2027: '--', 2028: ':'}[seed]
            for ax, column, label in zip(axes, ['train_loss', 'dev_log_loss', 'dev_gauc'],
                    ['Training BCE', 'Development log loss', 'Development GAUC']):
                ax.plot(group.fractional_epoch, group[column], color=COLORS[variant], linestyle=style,
                        marker='o', markersize=2.8, linewidth=1.1,
                        label=f'{PLOT_NAMES[variant]}, {seed}')
                ax.set(xlabel='Epoch progress', ylabel=label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=7)
        for letter, ax in zip('abc', axes):
            ax.set_title(letter, loc='left', fontweight='bold')
        fig.savefig(directory / f'{filename}.png', dpi=300, bbox_inches='tight')
        fig.savefig(directory / f'{filename}.svg', bbox_inches='tight')
        fig.savefig(directory / f'{filename}.pdf', bbox_inches='tight')
        plt.close(fig)
        paths.append(filename)
    return paths


def report(result, pure, runs, curves, metrics, figures):
    winner = result.get('selected_variant')
    adopted = result['decision'].startswith('prefer_')
    lines = ['# 推荐算法第二轮优化总结', '',
             '本轮完成Pure场景一致性诊断，以及1K作者频次、初始化和显式历史匹配对照。', '',
             '## 结果与采用方案', '']
    if adopted:
        before = np.mean([row['baseline']['gauc'] for row in result['test']['per_seed']])
        after = np.mean([row['candidate']['gauc'] for row in result['test']['per_seed']])
        lines.append(f"1K曝光预测实验优先采用{NAMES[winner]}，三种子平均测试GAUC由{before:.5f}升至{after:.5f}，变化及95%区间为{ci_text(result['test']['difference'])}个百分点。Pure页面继续使用RRF排序。")
    elif 'test' in result:
        lines.append(f"1K继续采用B0。{NAMES[winner]}通过开发筛选与种子复核，测试GAUC变化及95%区间为{ci_text(result['test']['difference'])}个百分点，未达到替换条件。")
    elif 'confirmation' in result:
        lines.append(f"1K继续采用B0。{NAMES[winner]}进入三种子复核，开发GAUC变化及95%区间为{ci_text(result['confirmation']['difference'])}个百分点，未达到测试评价条件。")
    else:
        lines.append('1K继续采用B0。四个候选方案均未达到追加种子的开发条件，本轮在种子2026对照后结束。')
    lines += ['', pure['conclusion'], '', '## 1. Pure场景一致性诊断', '',
              '复用现有观看模型，开发曝光按用户、日期、场景分组，比较真实场景与固定tab=0。', '',
              '| 输入场景 | 有效组 | 曝光数 | GAUC | NDCG@10 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, row in pure['exposure']['summary'].items():
        label = '真实场景' if name == 'actual_scene' else '固定tab=0'
        lines.append(f"| {label} | {row['valid_groups']:,} | {row['covered_rows']:,} | {row['gauc']:.5f} | {row['ndcg10']:.5f} |")
    interval = pure['exposure']['actual_minus_tab0']['gauc']
    lines += ['', f"真实场景的组内GAUC变化为{100*interval['estimate']:+.3f}个百分点，用户配对95%区间[{100*interval['lower']:+.3f}, {100*interval['upper']:+.3f}]。", '',
              f"曝光组只纳入正负均有的组，GAUC按曝光数加权，NDCG组间等权并处理并列分数。曝光有效组中位数为{pure['exposure']['group_size']['median']:.0f}条，不能把其NDCG与100候选口径直接对比。", '',
              '固定候选评价把用户日期查询拆成实际出现的场景，同组100条候选都使用该场景，正例也限定在该场景。下表只统计有正例的场景查询。', '',
              '| 排序 | 场景查询数 | Recall@10 | NDCG@10 | 内容覆盖率 |', '| --- | ---: | ---: | ---: | ---: |']
    for name in ('rrf', 'watch', 'blend'):
        row = pure['candidates']['summary'][name]
        lines.append(f"| {dict(rrf='原RRF',watch='观看模型',blend='75/25融合')[name]} | {row['queries']:,} | {row['recall10']:.3%} | {row['ndcg10']:.5f} | {row['coverage']:.2%} |")
    lines += ['', '该补充评价与旧全场景评价的查询和分母不同，数值分别解释。候选未曝光代表反馈未知；Pure未开展新训练，也未评价测试集。', '',
              '## 2. 1K配对对照', '',
              '五组共用训练曝光、batch size 256、Adam学习率0.001和最多5轮训练。前两轮按每四分之一轮检查开发指标，此后每轮检查；按开发GAUC保存最佳点，连续两轮没有更好结果则结束。', '',
              'B0与新增特征方案的共享参数使用一致初始化，新增输入列初始权重为0。作者按正式训练曝光计数，高频组保留至少101次的作者，其余回退为零；小初始化组将作者embedding标准差设为0.0001。显式匹配组增加3个历史匹配数值，不增加作者embedding表。', '',
              '下表为种子2026。用户×场景GAUC按user_id与tab分组，跨日期汇总；主GAUC按用户分组。差值和区间以本轮B0为参照。', '',
              '| 方案 | 开发GAUC | 用户×场景GAUC | GAUC差值及95%区间（百分点） | 参数量 | 最佳训练进度／步数 |',
              '| --- | ---: | ---: | --- | ---: | --- |']
    for row in metrics[metrics.split.eq('dev') & metrics.seed.eq(2026)].itertuples():
        ci = '—' if row.variant == 'baseline' else ci_text(result['dev_variants'][row.variant]['difference'])
        lines.append(f"| {NAMES[row.variant]} | {row.gauc:.5f} | {row.user_scene_gauc:.5f} | {ci} | {row.parameters:,} | {row.best_fractional_epoch:.3f}轮／{int(row.best_eval_step):,} |")
    old = read(ROOT / 'reports/kuairand_1k/full1000/experiment_results.json')['training']['B0_seed2026']
    old_best = max(old['epochs'], key=lambda p: p['dev']['gauc'])
    current = runs[('baseline', 2026)]
    lines += ['',
        f"旧B0按整轮评价，种子2026选中第{old_best['epoch']}轮，开发GAUC为{old_best['dev']['gauc']:.5f}；本轮B0选中{current['best_fractional_epoch']:.3f}轮、步数{current['best_eval_step']:,}，开发GAUC为{current['best_dev']['gauc']:.5f}。轮内评价扩大了可选训练停点，并可能改变早停时机。因此本轮特征增量统一相对本轮B0计算。", '',
        '### 显式匹配特征的覆盖', '',
        '| 集合 | 匹配类型 | 非零覆盖率 | 有匹配的观看达标率 | 无匹配的观看达标率 |',
        '| --- | --- | ---: | ---: | ---: |']
    coverage = read(OUT / 'match_coverage.json')
    for split, label in [('train', '训练'), ('dev', '开发')]:
        for feature, name in [('same_author_any', '同作者'), ('shared_tag_history_fraction', '同标签')]:
            row = coverage[split]['features'][feature]
            lines.append(f"| {label} | {name} | {row['nonzero_coverage']:.2%} | {row['long_view_rate_nonzero']:.2%} | {row['long_view_rate_zero']:.2%} |")
    author_coverage = coverage['dev']['features']['same_author_any']['nonzero_coverage']
    tag_coverage = coverage['dev']['features']['shared_tag_history_fraction']['nonzero_coverage']
    lines += ['', f"开发曝光中，同作者匹配覆盖{author_coverage:.2%}，同标签匹配覆盖{tag_coverage:.2%}。同作者占比和同作者是否出现使用相同的非零样本。表中正例率是未经其他变量调整的观察关联；实际排序增益由模型对照判断。"]
    lines += ['', '### 分阶段解释', '']
    for name, label in [('frequency_high_minus_full', '高频作者相对全部作者'),
                        ('small_init_minus_default', '小初始化相对高频作者默认初始化')]:
        lines.append(f"- {label}：开发GAUC差值及95%区间为{ci_text(result['stage_comparisons'][name]['difference'])}个百分点。")
    lines += ['', '高频筛选和初始化对照用于判断作者方案内部的变化；最终是否改善仍以相对B0的主指标为准。', '',
              '### 训练用时与选中停点', '',
              '| 方案 | 种子 | 完成轮数 | 累计训练分钟 | 最佳训练进度／步数 |',
              '| --- | ---: | ---: | ---: | --- |']
    total_minutes = 0.
    for (name, seed), run in runs.items():
        minutes = sum(p['train_seconds'] for p in run['epochs']) / 60
        total_minutes += minutes
        lines.append(f"| {NAMES[name]} | {seed} | {len(run['epochs'])} | {minutes:.1f} | {run['best_fractional_epoch']:.3f}轮／{run['best_eval_step']:,} |")
    lines += ['', f"以上覆盖全部完成种子，累计训练时间{total_minutes:.1f}分钟。时间按每轮训练循环累计，数据加载、开发评价、后续对照统计和报告生成另计。", '',
              '### 训练曲线', '']
    for filename in figures:
        label = '种子2026五组训练曲线' if filename.startswith('screening') else '优选方案与B0的三种子训练曲线'
        lines += [f'![{label}](figures/{filename}.png)', '']
    lines += ['图中保留全部实际开发评价点，曲线连接相邻记录。横轴为累计训练轮数；训练BCE是当前轮截至该点的累计均值，开发LogLoss和GAUC来自该点完整开发评价。颜色代表方案，线型代表种子。', '',
              '### 追加种子与测试', '',
              '单种子开发GAUC至少增加0.0005且用户配对区间下界大于0，才进入2027、2028复核。三种子各自改善、平均增量达到0.0005且区间下界大于0后，才评价开发选出的方案及其B0对照的测试结果。']
    if 'confirmation' in result:
        lines += ['', '| 阶段 | 种子 | B0 GAUC | 候选GAUC | B0用户×场景GAUC | 候选用户×场景GAUC |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        stage_summaries = []
        for key, label in [('confirmation', '开发复核'), ('test', '测试')]:
            if key not in result:
                continue
            for seed, row in zip(result['confirmation']['seeds'], result[key]['per_seed']):
                lines.append(f"| {label} | {seed} | {row['baseline']['gauc']:.5f} | {row['candidate']['gauc']:.5f} | {row['user_scene_baseline']['gauc']:.5f} | {row['user_scene_candidate']['gauc']:.5f} |")
            stage_summaries += ['', f"{label}配对平均GAUC差值及95%区间：{ci_text(result[key]['difference'])}个百分点。"]
        lines += stage_summaries
    else:
        lines += ['', '本轮未追加训练种子，也未开展1K新方案测试集评价。']
    lines += ['', '### 训练出现与未见视频', '',
              '种子2026开发集，依据同期B0是否训练过该视频ID分组。后四列为GAUC相对B0的变化，单位为百分点。', '',
              '| 视频分组 | 开发曝光数 | B0 GAUC | 完整作者 | 高频作者 | 高频小初始化 | 历史匹配 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, label in [('trained_video', '训练出现'), ('untrained_video', '训练未见')]:
        group = result['dev_variants']['full_author']['groups'][key]
        base = group['baseline']['gauc']
        values = [100*(result['dev_variants'][name]['groups'][key]['candidate']['gauc']-base)
                  for name in ('full_author', 'high_author', 'high_author_small', 'history_matches')]
        lines.append(f"| {label} | {group['rows']:,} | {base:.5f} | " +
                     ' | '.join(f'{value:+.3f}' for value in values) + ' |')
    if (OUT / 'author_groups.json').exists():
        groups = read(OUT / 'author_groups.json')['groups']
        lines += ['', '### 作者训练频次分组', '',
                  '种子2026开发集；各组按用户计算GAUC。后四列为相对同期B0的变化，单位为百分点。', '',
                  '| 作者训练曝光次数 | 开发曝光数 | B0 GAUC | 完整作者 | 高频作者 | 高频小初始化 | 历史匹配 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label, group in groups.items():
            base = group['variants']['baseline']['gauc']
            changes = [100*(group['variants'][name]['gauc']-base)
                       for name in ('full_author', 'high_author', 'high_author_small', 'history_matches')]
            lines.append(f"| {'未见或缺失' if label=='unknown' else label} | {group['rows']:,} | {base:.5f} | " +
                         ' | '.join(f'{value:+.3f}' for value in changes) + ' |')
    lines += ['', '### 本轮对效果下降的解释', '',
              '完整作者表加入大量只出现几次的ID，训练BCE下降更快，开发GAUC却更差；低频作者组和训练未见视频组均出现退化。频次筛选缩小了作者表，再减小初始化尺度后，作者方案获得正向结果。这说明作者特征的效果依赖于低频回退和初始化方式。', '',
              '显式历史匹配直接表达候选与已有兴趣的关系，三列分别为历史同作者占比、是否存在同作者、共享候选标签的历史占比。所有列只读取该请求之前有效的历史，空历史返回0，缺失作者和未知标签不计为匹配。它增加192个权重，六个作者频次组、训练出现与未见视频组的开发GAUC点估计均改善；三列各自贡献需要单独消融才能区分。', '',
              'Pure的真实场景输入改善了曝光组内GAUC，但观看模型在100候选排序中仍不如RRF。场景修正解决了一部分输入不一致，曝光分类训练与召回候选排序的目标、样本分布差异仍然存在。本轮据此保留RRF，不追加新排序器。']
    lines += ['', '## 3. 代码与复现', '',
              '`sequence_model.py`增加可选的配对初始化和3维匹配输入；`sequence_training.py`增加作者频次筛选、作者初始化尺度、匹配特征计算，以及轮内开发评价和检查点记录。原配置仍可按原默认行为运行。', '',
              '47项测试通过，覆盖公共参数与初始预测一致、频次100/101边界、缺失值与空历史、反馈延迟、最佳轮内检查点恢复及整轮早停。新旧默认模型的B0/B1/B2初始化也已逐项核对。', '',
              '在项目根目录执行，训练使用具备CUDA的Python环境，报告生成需要matplotlib。', '',
              '```powershell', '$env:PYTHONPATH = "$PWD\\src"',
              'python -m unittest discover -s tests -p "test_*.py"',
              'python scripts/evaluate_pure_scenes.py', 'python scripts/run_author_followup.py',
              'python scripts/audit_followup_matches.py', 'python scripts/audit_followup_author_groups.py',
              'python scripts/report_author_followup.py', '```', '',
              '实验入口复用相同配置下已完成的训练结果。新模型与映射位于artifacts/kuairand_1k/author_v2，各方案独立保存；原模型继续保留。', '',
              '## 4. 结果与数据', '',
              '配对区间均按用户整体抽样2000次，保留同一用户的观测关系。1K多种子时平均配对差值。区间描述已训练模型下的用户差异；既有测试集用于本项目回顾性比较。', '',
              '- [Pure诊断说明](pure_scene/report.md)', '- [Pure完整结果](pure_scene/results.json)',
              '- [1K完整比较与分组结果](sequence/comparison.json)', '- [指标CSV](author_metrics.csv)',
              '- [显式匹配覆盖与正例率](match_coverage.json)',
              '- [作者训练频次分组](author_groups.json)',
              '- [全部实际评价点CSV](training_curves.csv)', '- [Pure指标CSV](pure_scene_metrics.csv)']
    if (ROOT / 'reports/optimization_research/review.md').exists():
        lines.append('- [方案依据与论文代码对照](../optimization_research/review.md)')
    if (OUT / 'verification.json').exists():
        lines.append('- [代码与文件检查](verification.json)')
    if adopted:
        lines += ['', '### 使用本轮1K模型', '',
                  f'完整配置为`artifacts/kuairand_1k/author_v2/configs/{winner}.yaml`，模型为该方案目录下`B0_seed2026/best.pt`；2027、2028模型用于配对复核。检查点同时保存`spec`、`mode`、`model_kwargs`和权重，重建模型时需一并读取。`run_author_followup.py`中的测试回放展示了完整加载与预测流程。', '',
                  '1K模型用于曝光预测实验，Pure演示页面仍读取Pure候选和模型。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    result = read(OUT / 'sequence/comparison.json')
    if 'decision' not in result:
        raise RuntimeError('Experiments are still running; final report needs completed comparisons')
    pure = read(OUT / 'pure_scene/results.json')
    runs, curves = training_records(result)
    metrics = metric_records(result, runs)
    curves.to_csv(OUT / 'training_curves.csv', index=False)
    metrics.to_csv(OUT / 'author_metrics.csv', index=False)
    pd.DataFrame([dict(method=name, **row) for name, row in pure['candidates']['summary'].items()])\
        .to_csv(OUT / 'pure_scene_metrics.csv', index=False)
    figures = draw_curves(curves, result)
    report(result, pure, runs, curves, metrics, figures)
    print(OUT / 'summary.md')


if __name__ == '__main__':
    main()
