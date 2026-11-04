"""Run the agreed full-1K comparison sequentially and update local deliverables.

Use the CUDA Python interpreter. --report-python can select an environment
that already contains matplotlib. Progress/logs stay under ignored artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/kuairand_1k_full.yaml'
REPORT = ROOT / 'reports/kuairand_1k/full1000'
ARTIFACTS = ROOT / 'artifacts/kuairand_1k/full1000'


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
                       if r.get('training',{}).get(f'{m}_seed{seed}',{}).get('status')!='development_comparison'
                       or not all((ARTIFACTS/f'{m}_seed{seed}'/name).is_file()
                                  for name in ('best.pt','dev_predictions.parquet'))]
                if modes:
                    command(f'train_seed_{seed}',[sys.executable,'-m','gamefeed.sequence_training','--config',CONFIG,'--seed',str(seed),'--modes',*modes])
            command('final_evaluation',[sys.executable,'-m','gamefeed.sequence_evaluation','--config',CONFIG])
            command('plots_and_report',[report_python,'scripts/report_sequence_experiment.py','--report-dir',str(REPORT)])
            result=json.loads((REPORT/'experiment_results.json').read_text(encoding='utf-8'))
            assert len(result['training'])==9 and len(result['final_evaluation'])==4
            status('complete','complete',chosen=result['selection']['chosen'],
                test_gauc={m:v['gauc']['mean'] for m,v in result['final_evaluation']['test_standard']['summary'].items()})
            print('Complete: full comparison, evaluation and figures.',file=log,flush=True)
        except BaseException as exc:
            traceback.print_exc(file=log)
            status('failed','failed',error=str(exc))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-python',default=sys.executable)
    main(p.parse_args().report_python)
