"""Run saved-release attribution and final-list evaluation without retraining."""
from gamefeed.config import load_config
from gamefeed.final_evaluation import run

if __name__ == '__main__':
    result = run(load_config('configs/release.yaml'))
    print('Saved reports/release/final_evaluation/report.md', flush=True)
