"""Reconcile saved metric denominators, without training new models."""
from gamefeed.config import load_config
from gamefeed.report_checks import run

if __name__ == '__main__':
    run(load_config('configs/release.yaml'))
    print('Saved reports/release/interpretation_checks/report.md')
