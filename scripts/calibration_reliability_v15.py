"""V15 software readiness and synthetic checks only. No real-data CLI execution."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trustretina_v15.v15_inputs import REGISTRY_SHA256, FROZEN_PLAN_SHA256, verified_path, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('readiness', 'synthetic-check'))
    parser.add_argument('--output', type=Path, help='Fresh synthetic-only report directory')
    args = parser.parse_args(argv)
    try:
        verified_path(ROOT, 'results/candidate-v14/verified-artifact-registry.json', REGISTRY_SHA256)
        verified_path(ROOT, 'results/candidate-v14/frozen-plan.json', FROZEN_PLAN_SHA256)
        if args.command == 'readiness':
            print(json.dumps({'software_scope': 'CPU software and synthetic verification',
                              'public_v14_metadata_verified': True, 'real_v15_analysis_executed': False,
                              'real_execution_ready': False, 'deployment_default': 'REVIEW REQUIRED',
                              'blockers': ['Separate real-data authorization and exact manifest binding required',
                                           'Private saved artifacts and original role manifests not checked by this command',
                                           'Independent all-eight confirmation unavailable']}))
            return 0
        from trustretina_v15.v15 import analyze, save_report
        from trustretina_v15.v15_fixture import create_fixture
        protocol = sha256(ROOT/'docs/CALIBRATION_RELIABILITY_V15.md')
        with tempfile.TemporaryDirectory(prefix='trustretina-v15-synthetic-') as tmp:
            path, digest, approval = create_fixture(Path(tmp)/'inputs', protocol)
            report = analyze(path, digest, protocol, approval, fixture=True)
            if args.output:
                save_report(report, args.output)
        print(json.dumps({'synthetic_check_passed': True, 'data_kind': report['data_kind'],
                          'real_v15_results': False, 'deployment_default': report['deployment_default']}))
        return 0
    except (ValueError, PermissionError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
