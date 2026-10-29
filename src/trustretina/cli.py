"""Implemented foundation, patient data preparation and explicit runner checks."""
import argparse
import json
import sys
from .config import load_config
from .environment import environment_report, require_gpu
from .planning import execution_plan


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trustretina")
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser("check-env", help="Report observed capabilities; does not certify GPU jobs")
    env.add_argument("--network", action="store_true")
    env.add_argument("--require-gpu", action="store_true")
    for name in ("check-config", "plan", "check-development"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
    for name in ('discover-data', 'prepare-data'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', required=True)
        if name == 'prepare-data':
            sub.add_argument('--output', help='Immutable bundle path; default artifacts/data/phase1')
    commands.add_parser('check-runner', help='Read-only Kaggle authentication and connectivity check')
    smoke = commands.add_parser('prepare-runner-smoke', help='Write a nontraining script; never submits')
    smoke.add_argument('--owner', required=True)
    smoke.add_argument('--output', required=True)
    smoke.add_argument('--dataset-config', help='TOML with dataset_ref, input_root and image_directory_name')
    smoke.add_argument('--previous-bundle', help='Read a pinned receipt and prepare a separate new version of that kernel')
    smoke.add_argument('--gpu', action='store_true', help='Prepare a CUDA arithmetic check; execution uses GPU quota')
    run = commands.add_parser('run-runner-smoke', help='Submit prepared smoke, poll and retrieve artifacts')
    run.add_argument('--bundle', required=True)
    run.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check-development':
            from .data.development import load_development_config
            print(json.dumps(load_development_config(args.config),indent=2))
        elif args.command == "check-env":
            report = environment_report(args.network)
            print(json.dumps(report, indent=2))
            if args.require_gpu and not require_gpu(report):
                print("A usable PyTorch CUDA GPU was not verified.", file=sys.stderr)
                return 1
        elif args.command in ('check-runner', 'prepare-runner-smoke', 'run-runner-smoke'):
            from .runner.kaggle import check_runner, prepare_smoke, run_smoke
            if args.command == 'check-runner':
                result = check_runner()
                okay = result['authentication_verified']
            elif args.command == 'prepare-runner-smoke':
                dataset = None
                if args.dataset_config:
                    import tomllib
                    with open(args.dataset_config, 'rb') as handle:
                        dataset = tomllib.load(handle)
                result = prepare_smoke(args.output, args.owner, args.gpu, dataset, args.previous_bundle)
                okay = result['prepared']
            else:
                result = run_smoke(args.bundle, args.timeout_seconds)
                okay = result['smoke_executed'] and 'blocker' not in result
            print(json.dumps(result, indent=2))
            return 0 if okay else 1
        elif args.command in ('discover-data', 'prepare-data'):
            from pathlib import Path
            from .data.validation import discover
            from .data.splits import prepare_data
            config = load_config(args.config)
            if args.command == 'discover-data':
                result = discover(config)
            else:
                output = args.output or str(Path(config['paths']['artifacts']) / 'data' / 'phase1')
                result = prepare_data(config, output)
            print(json.dumps(result, indent=2))
        else:
            config = load_config(args.config)
            print(json.dumps(config if args.command == "check-config" else execution_plan(config), indent=2))
    except ImportError:
        print('Optional dependency missing; install the data or runner extra for this command.', file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
