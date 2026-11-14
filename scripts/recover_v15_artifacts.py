"""Plan V15 selective recovery; real downloads need a separate exact approval."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from trustretina_v15.recovery import public_plan, identity, recover, RecoveryError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'recover'))
    parser.add_argument('--profile', choices=('metadata', 'analysis'), default='metadata')
    parser.add_argument('--plan-file', type=Path)
    parser.add_argument('--approval-file', type=Path)
    parser.add_argument('--private-root', type=Path)
    args = parser.parse_args(argv)
    try:
        expected = public_plan(args.profile)
        if args.action == 'plan':
            print(json.dumps({'plan_sha256': identity(expected), 'plan': expected}, indent=2, sort_keys=True))
            return 0
        if not all((args.plan_file, args.approval_file, args.private_root)):
            raise PermissionError('Recovery requires a reviewed plan, separate approval and approved private root')
        plan = json.loads(args.plan_file.read_text())
        # Accept the exact plan output envelope or its plan member alone.
        if 'plan' in plan:
            if plan.get('plan_sha256') != identity(plan['plan']):
                raise RecoveryError('Plan envelope digest mismatch')
            plan = plan['plan']
        if plan != expected:
            raise RecoveryError('Plan does not match the requested pinned profile')
        approval = json.loads(args.approval_file.read_text())
        result = recover(plan, approval, args.private_root)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (RecoveryError, PermissionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (Exception, KeyboardInterrupt):
        print('Recovery blocked; no provider details or private contents printed', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
