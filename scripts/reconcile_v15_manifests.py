#!/usr/bin/env python3
"""Public planning and separately authorized offline original V4 reconciliation."""
import argparse
import json
from pathlib import Path
from trustretina_v15.reconciliation import authorized_reconcile, canonical, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('plan', help='Public source/hash identities only; no private access')
    run = commands.add_parser('reconcile', help='Requires separate explicit private reconstruction approval')
    for key in ('input-root', 'output', 'approval-file'):
        run.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'plan':
            expected = plan()
            print(json.dumps({'plan': expected, 'plan_sha256': canonical(expected)}, indent=2, sort_keys=True))
        else:
            result = authorized_reconcile(args.input_root, args.output,
                                          json.loads(args.approval_file.read_bytes()))
            print(json.dumps({'status': result['status'], 'v15_fitting_authorized': False}))
    except Exception:
        print(json.dumps({'status': 'blocked', 'reason': 'Reconciliation failed; inspect private diagnostics if created'}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
