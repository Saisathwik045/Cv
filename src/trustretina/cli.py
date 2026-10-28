"""Only implemented Phase 0 operations are exposed as commands."""
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
    for name in ("check-config", "plan"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-env":
            report = environment_report(args.network)
            print(json.dumps(report, indent=2))
            if args.require_gpu and not require_gpu(report):
                print("A usable PyTorch CUDA GPU was not verified.", file=sys.stderr)
                return 1
        else:
            config = load_config(args.config)
            print(json.dumps(config if args.command == "check-config" else execution_plan(config), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
