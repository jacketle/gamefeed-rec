from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

from .analysis import analyze_retrieval_groups
from .config import load_config
from .demo import run_demo
from .supplement import prepare_supplement
from .text_retrieval import run_text
from .pipeline import run_fusion, run_inspection, run_itemcf, run_popularity, run_prepare
from .reporting import write_experiment_summary
from .training import train_shared_ranker, train_single_rankers, train_two_tower


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gamefeed", description="GameFeed-Rec command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "prepare", "prepare-content", "text-dev", "semantic-dev", "train-two-tower", "train-ranker", "train-single-rankers", "report"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/release.yaml" if command == "report" else "configs/base.yaml")
    for command in ("validate-models", "optimize-fusion", "serve"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/release.yaml")
        if command == "serve":
            child.add_argument("--port", type=int, default=8765)
    popularity = subparsers.add_parser("popularity")
    popularity.add_argument("--config", default="configs/base.yaml")
    popularity.add_argument("--split", choices=("dev", "test"), default="dev")
    itemcf = subparsers.add_parser("itemcf")
    itemcf.add_argument("--config", default="configs/base.yaml")
    itemcf.add_argument("--split", choices=("dev", "test"), default="dev")
    fusion = subparsers.add_parser("fusion")
    fusion.add_argument("--config", default="configs/base.yaml")
    fusion.add_argument("--split", choices=("dev", "test"), default="dev")
    analyze = subparsers.add_parser("analyze-groups")
    analyze.add_argument("--config", default="configs/base.yaml")
    analyze.add_argument("--method", choices=("popularity", "itemcf", "fusion", "selected", "semantic"), default="fusion")
    analyze.add_argument("--split", choices=("dev", "test"), default="test")
    demo = subparsers.add_parser("demo")
    demo.add_argument("--config", default="configs/base.yaml")
    demo.add_argument("--user-id")
    demo.add_argument("--date")
    demo.add_argument("--top-k", type=int, default=10)
    demo.add_argument("--split", choices=("dev", "test"), default="test")
    subparsers.add_parser("test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "test":
        root = Path(__file__).resolve().parents[2]
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    config = load_config(args.config)
    if args.command == "serve":
        from .serving import serve
        serve(config, args.port)
        return 0
    elif args.command == "validate-models":
        from .validation import run_validation
        result = run_validation(config)
    elif args.command == "optimize-fusion":
        from .fusion_experiment import run_fusion_experiment
        result = run_fusion_experiment(config)
    elif args.command == "inspect":
        result = run_inspection(config)
    elif args.command == "prepare-content":
        result = prepare_supplement(config)
    elif args.command == "text-dev":
        result = run_text(config, 'dev')
    elif args.command == "semantic-dev":
        result = run_text(config, 'dev', semantic=True)
    elif args.command == "prepare":
        result = run_prepare(config)
    elif args.command == "train-two-tower":
        result = train_two_tower(config)
    elif args.command == "train-ranker":
        result = train_shared_ranker(config)
    elif args.command == "train-single-rankers":
        result = train_single_rankers(config)
    elif args.command == "itemcf":
        result = run_itemcf(config, args.split)
    elif args.command == "report":
        result = write_experiment_summary(config)
    elif args.command == "fusion":
        result = run_fusion(config, args.split)
    elif args.command == "analyze-groups":
        result = analyze_retrieval_groups(config, args.method, args.split)
    elif args.command == "demo":
        result = run_demo(config, args.user_id, args.date, args.top_k, args.split)
    else:
        result = run_popularity(config, args.split)
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

