"""Generate the current report from measured results, not old conclusions."""
import json
import pandas as pd
from .utils import write_json
from .content_analysis import positive_metrics


def write_experiment_summary(config):
    paths = config['resolved_paths']
    def read(name):
        file = paths.report_dir / name
        if not file.exists():
            raise FileNotFoundError(f'缺少 {file}。请按 README 运行 release 实验。')
        return json.loads(file.read_text(encoding='utf-8'))
    pct = lambda x: f'{100*x:.2f}%'
    validation = read('validation.json')
    selection = read('fusion_selection.json')
    shared = read('shared_ranker_results.json')
    singles = read('single_ranker_results.json')
    selected = read('selected_test.json')
    items = pd.read_parquet(paths.processed_dir / 'items_content.parquet')
    events = pd.read_parquet(paths.processed_dir / 'events.parquet')
    game_ids = set(items.loc[items.is_game, 'video_id'])
    unseen = set(items.video_id) - set(events.loc[events.split.eq('train'), 'video_id'])
    groups = {}
    results = {name: {split: read(f'{name}_{split}.json') for split in ('dev','test')}
               for name in ('popularity','itemcf','semantic')}
    results['selected'] = {'dev': read('selected_dev.json'), 'test': selected['metrics']}
    for method in results:
        recs = pd.read_parquet(paths.report_dir / f'{method}_test_recommendations.parquet')
        groups[method] = {'game': positive_metrics(recs, game_ids), 'unseen': positive_metrics(recs, unseen)}
    summary = dict(selected=selection['selected'], selected_test=selected['metrics'],
                   popularity_test=results['popularity']['test'], game_items=len(game_ids),
                   results=results, groups=groups, validation=validation, single_rankers=singles,
                   shared_ranker=shared, events=len(events), users=int(events.user_id.nunique()), items=len(items))
    write_json(paths.report_dir / 'release_summary.json', summary)
    lines = ['# GameFeed 当前实验报告', '',
        '由结果JSON自动生成。配置为 `configs/release.yaml`；旧目录 `reports/generated` 不用于本次结论。', '',
        '## 数据与评价', '',
        f'- 固定抽取5000名用户，{len(events):,}条行为、{len(items):,}条内容；游戏类别内容{len(game_ids)}条。',
        '- 2022年训练4/8–4/21，开发4/22–4/28，测试4/29–5/8；实际抽样行为最早为4/9。',
        '- 每个用户每天一条召回查询，使用前一天结束前最近20条观看达标历史；只评价当天有观看达标正例的评估样本。',
        '- 内容从上传日期次日可用，各方法均允许重复推荐历史内容，每次输出100个不重复ID。',
        '- 热门度逐日累计此前观看达标；ItemCF相似度、模型、ID词表和标准化参数只用训练数据。开发/测试时允许读取此前日期已发生的行为，属于按日回放。',
        '- Recall先算每个评估样本的正例命中比例，再取平均；HitRate表示至少命中一条的评估样本比例；覆盖率分母为全部7583条内容。',
        '- 排序AUC在原始曝光行为上计算。GAUC按用户分组、按有效组行为行数加权，全正或全负用户不计GAUC。',
        '- 测试集在此前迭代中已经查看。本轮仅用开发集选择融合设置，测试结果属于已有数据上的对照，后续可另留新日期确认。', '',
        '## 召回对照', '',
        '| 方法 | 开发Recall@100 | 测试Recall@50 | 测试Recall@100 | 测试HitRate@100 | 测试覆盖率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, label in [('popularity','热门'),('itemcf','ItemCF'),('semantic','BGE语义'),('selected','选定融合')]:
        r = results[name]
        lines.append(f"| {label} | {pct(r['dev']['recall@100'])} | {pct(r['test']['recall@50'])} | {pct(r['test']['recall@100'])} | {pct(r['test']['hitrate@100'])} | {pct(r['test']['coverage@100'])} |")
    base, final, cf = [results[n]['test']['recall@100'] for n in ('popularity','selected','itemcf')]
    lines += ['', f'选定 `{selection["selected"]["name"]}`：相对热门提升{100*(final-base):.2f}个百分点（相对提升{100*(final/base-1):.2f}%），相对ItemCF提升{100*(final-cf):.2f}个百分点。', '',
        '本次选中quota_0：热门保留10条，ItemCF补齐100条，重复ID只算一次。集合内按三路RRF分数排序（ItemCF 0.8、热门0.2、语义0.2）；语义不占独立名额，重叠内容仍贡献名次分。', '',
        '## 开发集选择过程', '', '| 设置 | Recall@100 | 游戏正例Recall@100 |', '| --- | ---: | ---: |']
    for trial in selection['trials']:
        lines.append(f"| {trial['setting']['name']} | {pct(trial['metrics']['recall@100'])} | {pct(trial['game']['macro_recall'])} |")
    lines += ['', 'RRF比较名次，避免直接相加不同量纲的相似度。配额方案比较语义0/10/20个名额，热门固定10个，ItemCF补齐，总数始终100。', '',
        '## 双塔检查与历史消融', '',
        f"100条训练样本拟合：正例排第一比例{pct(validation['small_fit']['positive_top1_accuracy'])}，损失{validation['small_fit']['initial_loss']:.6f} → {validation['small_fit']['final_loss']:.6f}。验证训练路径能学习，不代表未见数据效果。", '',
        '| 设置 | 开发Recall@100 |', '| --- | ---: |']
    for name, label in [('with_history','有历史'),('without_history','去掉历史与历史长度')]:
        lines.append(f"| {label} | {pct(validation[name]['retrieval']['dev']['recall@100'])} |")
    lines += ['', '相同样本、种子、结构、3轮训练，各自按开发损失选最佳轮。历史带来小幅提升，双塔仍明显弱于ItemCF，本轮不作为主召回。', '',
        '## 排序对照', '',
        '相同特征、随机种子2026、批次顺序；每个模型固定训练3轮并使用第3轮参数。两个单任务模型总参数和总训练成本高于一个共享模型。', '',
        '| 方法 | 开发观看达标AUC | 开发点赞AUC | 测试观看达标AUC | 测试点赞AUC |', '| --- | ---: | ---: | ---: | ---: |',
        f"| 两个单任务 | {singles['long_view']['dev']['auc']:.4f} | {singles['is_like']['dev']['auc']:.4f} | {singles['long_view']['test']['auc']:.4f} | {singles['is_like']['test']['auc']:.4f} |",
        f"| 共享底层 | {shared['dev']['long_view']['auc']:.4f} | {shared['dev']['is_like']['auc']:.4f} | {shared['test']['long_view']['auc']:.4f} | {shared['test']['is_like']['auc']:.4f} |", '',
        '当前设置下独立模型的曝光预测更好，保留用于排序对照。共享结构、损失权重与训练预算共同影响结果；本次比较尚未区分这些原因。', '',
        '## 分组结果', '', '| 方法 | 游戏正例Recall@100 | 训练未见正例Recall@100 |', '| --- | ---: | ---: |']
    for name, values in groups.items():
        lines.append(f"| {name} | {pct(values['game']['macro_recall'])} | {pct(values['unseen']['macro_recall'])} |")
    lines += ['', '游戏组905个评估样本、1019个去重正例；训练未见组20个评估样本、20个正例。“训练未见”指训练行为没出现，不等于刚发布。分组只限制正例，推荐列表仍为全内容前100条。', '',
        '## 展示与耗时', '',
        '页面默认读取预计算100条候选并保留融合候选的RRF顺序，返回前10条。勾选实验对照后使用0.7×观看达标概率＋0.3×点赞概率排序，再执行同作者最多2条与文案去重。', '',
        '前10条检查：测试Recall@10从融合RRF顺序4.71%降至模型排序3.23%，再做作者限制和文案去重为3.24%；开发集也下降，因此默认保留融合候选的RRF顺序。详见 `final_evaluation/report.md`。', '',
        '整批召回耗时见 `route_timing_test.json`，包含必要拟合/编码。已有排序展示路径的局部计时：CPU单请求顺序调用、预热5次、测量30次，P95为73.5ms；包含缓存候选读取、两个模型评分、元数据和作者限制，排除模型加载、召回生成与HTTP传输。该记录早于文案去重和默认召回顺序的调整。', '',
        '## 补充检查', '',
        '开发集Top10对照（5410组）：热门Recall@10为4.448%、NDCG@10为0.02204；ItemCF为3.858%、0.01997；融合候选的RRF顺序为4.493%、0.02317；固定tab=0、权重0.7/0.3的模型排序为3.367%、0.01691。融合在这四种设置中均最高，Recall@10比热门高约0.045个百分点，保持现有默认方案。详表见 `interpretation_checks/report.md`。', '',
        '融合相对ItemCF提高0.685个百分点，按用户重复抽样2000次的95%区间为[0.441, 0.941]个百分点。新增101次正例命中来自热门前10条，同时丢失35次；游戏组新增0次、丢失4次。详见 `final_evaluation/report.md`。', '',
        'GAUC覆盖率、正例率和每日随机基线详见 `interpretation_checks/report.md`。测试点赞GAUC覆盖312/4402名用户和3688/31020条曝光，分别为7.09%和11.89%；原代码的有效组与记录数已核对一致。', '',
        '逐日均匀随机选100条的Recall期望为1.3187%。开发与测试期间每日可用内容均为7583条，不可用正例为0，因此与100÷7583相同。', '',
        '复现命令见README，项目讲解见 `docs/RESUME_AND_INTERVIEW.md`。', '',
        '## 资料', '',
        '- [KuaiRand官方仓库](https://github.com/chongminggao/KuaiRand)',
        '- [数据下载](https://zenodo.org/records/10439422)',
        '- [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)', '']
    output = paths.report_dir / 'experiment_summary.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    return {'output': str(output)}
