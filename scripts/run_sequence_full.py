"""Run the agreed full-1K comparison sequentially and update local deliverables.

Use the CUDA Python interpreter. --report-python can select an environment
that already contains matplotlib. Progress/logs stay under ignored artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/kuairand_1k_full.yaml'
REPORT = ROOT / 'reports/kuairand_1k/full1000'
ARTIFACTS = ROOT / 'artifacts/kuairand_1k/full1000'


def update_docs(result):
    """Replace only this run's summary, retaining the earlier 100-user history."""
    chosen=result['selection']['chosen']
    final=result['final_evaluation']['test_standard']
    names={'B0':'基础模型','B1':'历史均值','B2':'候选相关加权'}
    rows=['| 模型 | 开发GAUC均值 | 测试GAUC均值 |','| --- | ---: | ---: |']
    for m in names:
        dev=result['selection']['mean_dev_gauc'][m]
        test=final['summary'][m]['gauc']['mean']
        rows.append(f'| {m} {names[m]} | {dev:.5f} | {test:.5f} |')
    d=final['B2_minus_B1'];lo,hi=d['ci95']
    body='\n'.join([
        '## 原B0／B1／B2历史模型对照（1,000用户）','',
        '本节记录原三种子历史模型对照；后续作者特征与显式历史匹配实验见各自优化报告。', '',
        '本轮保留全部1,000名日志用户。常规训练曝光3,764,530条、开发1,642,223条、测试1,188,322条；各阶段用户数随活跃情况变化。',
        '',*rows,'',
        f'该历史模型对照的开发集选择{chosen}（{names[chosen]}）。B2相对B1的测试GAUC差为{d["mean_difference"]*100:.3f}个百分点，按用户配对抽样2,000次的95%区间为[{lo*100:.3f}, {hi*100:.3f}]个百分点。',
        '',
        '三个模型使用相同样本、20条历史上限和三个配对种子；按开发集选择最佳轮和方法，再统一评价测试集。100用户阶段用于检查流程，新增用户扩大了共同训练信息；完整实验的测试期包含已经检查过的100用户记录。',
        '',
        '数据处理修正了历史期用户筛选遗漏、未训练用户向量回退，并使用紧凑字段类型和标签组合复用降低内存。原100用户的全部948,314条记录、历史和元数据核对一致，原检查点预测最大差值为0。',
        '',
        '在RTX 4060 Laptop GPU上完成训练。资源试跑的加权模型一轮约483秒，进程峰值约3.33 GiB、PyTorch分配峰值显存约1.09 GiB。实际各轮耗时与内存见结果表。',
        '',
        '已完成常规／随机展示、场景、内容与历史分组、反馈延迟检查及成功／失败案例。50／100条历史和全库召回仍为可选扩展。GitHub公开副本未自动同步。',
    ])
    marker_start='<!-- full1000-results:start -->'
    marker_end='<!-- full1000-results:end -->'
    for relative in ['README.md','docs/KuaiRand_1K后续实验计划.md','docs/RESUME_AND_INTERVIEW.md']:
        p=ROOT/relative
        if relative != 'README.md' and not p.is_file():
            continue
        s=p.read_text(encoding='utf-8')
        link=('' if relative=='README.md' else '../')+'reports/kuairand_1k/full1000/experiment_results.md'
        block=marker_start+'\n'+body+f'\n\n完整结果：[实验报告]({link})。\n'+marker_end
        pattern=re.escape(marker_start)+r'.*?'+re.escape(marker_end)
        s=re.sub(pattern,lambda _:block,s,flags=re.S) if marker_start in s else s.rstrip()+'\n\n'+block+'\n'
        if relative.endswith('KuaiRand_1K后续实验计划.md'):
            s=re.sub(r'^制定及修订日期：.*$',
                '制定及修订日期：2026-11-03。状态：全部1,000用户的主对照已完成，结果见文末；原100用户阶段保留作过程记录。',s,count=1,flags=re.M)
            s=s.replace('## 执行进度（2026-11-03）','## 100用户阶段记录（2026-11-03）')
            s=s.replace('正式任务正在依次训练B0、B1、B2的三个种子，之后按开发集选择方案，再评价测试集并更新报告。',
                'B0、B1、B2的三个种子已训练完成，已按开发集选择方案、评价测试集并更新报告。')
        p.write_text(s,encoding='utf-8')


def main(report_python):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['PYTHONUNBUFFERED']='1'
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    status_path=ARTIFACTS/'run_status.json'
    started=datetime.now(timezone.utc).isoformat()
    def status(state,phase,**extra):
        status_path.write_text(json.dumps(dict(state=state,phase=phase,pid=os.getpid(),
            started_utc=started,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),ensure_ascii=False,indent=2),encoding='utf-8')
    with (ARTIFACTS/'run.log').open('a',encoding='utf-8',buffering=1) as log:
        def command(phase,args):
            status('running',phase)
            print('\n'+datetime.now(timezone.utc).isoformat()+' '+phase,file=log,flush=True)
            subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
        try:
            for seed in (2026,2027,2028):
                r=json.loads((REPORT/'experiment_results.json').read_text(encoding='utf-8'))
                modes=[m for m in ('B0','B1','B2')
                       if not (r.get('training',{}).get(f'{m}_seed{seed}',{}).get('status')=='development_comparison'
                               and all((ARTIFACTS/f'{m}_seed{seed}'/name).is_file()
                                       for name in ('best.pt','dev_predictions.parquet')))]
                if modes:
                    command(f'train_seed_{seed}',[sys.executable,'-m','gamefeed.sequence_training','--config',CONFIG,'--seed',str(seed),'--modes',*modes])
            command('final_evaluation',[sys.executable,'-m','gamefeed.sequence_evaluation','--config',CONFIG])
            command('plots_and_report',[report_python,'scripts/report_sequence_experiment.py','--report-dir',str(REPORT.relative_to(ROOT))])
            result=json.loads((REPORT/'experiment_results.json').read_text(encoding='utf-8'))
            assert len(result['training'])==9 and len(result['final_evaluation'])==4
            status('running','update_local_docs');update_docs(result)
            command('offline_html',[report_python,'scripts/build_experiment_html.py'])
            html=(ROOT/'deliverables/GameFeed_实验结果图解.html').read_text(encoding='utf-8')
            assert '覆盖KuaiRand-1K全部1000名日志用户' in html
            assert 'reports/kuairand_1k/full1000/experiment_results.md' in html
            status('complete','complete',chosen=result['selection']['chosen'],
                test_gauc={m:v['gauc']['mean'] for m,v in result['final_evaluation']['test_standard']['summary'].items()})
            print('Complete: full comparison, evaluation, figures, local docs and HTML.',file=log,flush=True)
        except BaseException as exc:
            traceback.print_exc(file=log)
            status('failed','failed',error=str(exc))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-python',default=sys.executable)
    main(p.parse_args().report_python)
