#!/usr/bin/env python3
"""Read-only exact-version retrieval after queue restore. Never submits a job."""
import fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import controlled_experiment as c


def main():
    p=json.loads((c.BASE/'registry.json').read_text());api=c.transport.api_client();retrieved=[]
    for run in p['runs']:
        for attempt in run['attempts']:
            if attempt.get('status') not in ('complete','paused','failed'):continue
            folder=ROOT/attempt['bundle'];receipt=json.loads((folder/'submission.json').read_text())
            if not receipt.get('kernel_version'):raise ValueError('Ambiguous receipt: recover version before retrieval')
            report,_,v=c.transport.retrieve(api,folder)
            retrieved.append({'run':run['id'],'version':v['kernel_version'],'status':report['status'],'report_sha256':v['artifact_sha256']})
    print(json.dumps({'verified':retrieved,'job_submissions':0},indent=2))


if __name__=='__main__':
    with (c.BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
