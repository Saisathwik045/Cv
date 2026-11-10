#!/usr/bin/env python3
"""Orchestrate the unchanged approved v13 controller; persistent IDs, bounded polls."""
import contextlib,datetime,fcntl,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import resolution_experiment as c


class ObservedAPI:
    def __init__(self,api):self.api=api;self.last_quota=None
    def __getattr__(self,name):return getattr(self.api,name)
    def quota_view(self):
        result=self.api.quota_view();raw=json.loads(str(result));now=datetime.datetime.now(datetime.timezone.utc)
        snapshot={'observed_utc':now.isoformat(),'raw':raw}
        folder=c.BASE/'accounting';folder.mkdir(exist_ok=True)
        c.write(folder/('quota-'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'),snapshot)
        self.last_quota=(time.time(),raw);return result
    def kernels_push(self,folder,*,timeout,acc):
        if not self.last_quota or time.time()-self.last_quota[0]>60:raise ValueError('Fresh observed quota required')
        gpu=self.last_quota[1]['gpuQuota']
        if gpu.get('isPayToScaleEnabled') is not False or c.seconds(gpu['timeReserved'])!=0 or acc!='NvidiaTeslaT4' or str(timeout) not in ('2400','4200'):
            raise ValueError('Exact free accelerator configuration required')
        folder=Path(folder);meta=json.loads((folder/'kernel-metadata.json').read_text())
        if not meta['is_private'] or not meta['enable_gpu'] or meta['enable_tpu'] or meta['dataset_sources']:
            raise ValueError('Unexpected training resources/privacy/data mount')
        c.write(folder/'provider-push-start.json',{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_accelerator':acc,'timeout_seconds':int(timeout)})
        try:
            result=self.api.kernels_push(str(folder),timeout=timeout,acc=acc)
            c.write(folder/'provider-response.json',{'ref':result.ref,'version_number':result.version_number,'url':result.url})
            return result
        except Exception as e:
            response=getattr(e,'response',None)
            c.write(folder/'provider-error.json',{'type':type(e).__name__,'status':getattr(response,'status_code',None),'response':getattr(response,'text','')[:10000]})
            raise


def snapshot(p,state):
    value={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,
           'plan_sha256':p['plan_sha256'],'source_sha256':p['experiment_source_sha256'],
           'runs':[{'id':r['id'],'status':r['status'],'versions':[a.get('kernel_version',a['kernel_id']) for a in r['attempts']]} for r in p['runs']],
           'reserved_budget_seconds':sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts']),
           'submitted_or_reserved_attempts':sum(len(r['attempts']) for r in p['runs'])}
    c.write(c.BASE/'driver-latest.json',value);print(json.dumps(value),flush=True)


def main():
    approval_path=c.BASE/'approval.json';approval=json.loads(approval_path.read_text());started=time.monotonic()
    with (c.BASE/'execution-driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        api=ObservedAPI(c.transport.api_client())
        while time.monotonic()-started<43200:
            p=json.loads((c.BASE/'registry.json').read_text());c.require_approval(p,approval);c.verify_plan(p)
            if (c.BASE/'STOP_DRIVER').exists():snapshot(p,'stopped by explicit local flag; no continued monitoring');return
            remaining=p.get('last_advance_unix',0)+180-time.time()
            if remaining>0:time.sleep(min(30,remaining));continue
            if all(r['status']=='complete' for r in p['runs']):
                q=c.quota(api);c.write(c.BASE/'final-quota.json',q)
                if not q['paid_disabled']:raise ValueError('Paid scaling unexpectedly enabled')
                if q['reserved_seconds']>0:
                    p['last_advance_unix']=time.time();c.write(c.BASE/'registry.json',p);snapshot(p,'all fits complete; accelerator accounting not settled');continue
                charged=q['used_seconds']-p['quota_baseline']['used_seconds']
                if not 0<=charged<=21600:raise ValueError('Actual charged quota outside approved cap')
                snapshot(p,'all six complete and retrieved; final quota settled');return
            try:
                with (c.BASE/'controller.lock').open('a') as controller_lock:
                    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    try:c.advance(p,api,submit=True,approval=approval)
                    finally:c.write(c.BASE/'registry.json',p);c.export(p)
            except ValueError as e:
                qpath=c.BASE/'quota-latest.json';q=json.loads(qpath.read_text()) if qpath.exists() else {}
                if str(e)=='Paid/reserved accelerator quota blocks submission' and q.get('paid_disabled') and q.get('reserved_seconds',0)>0:
                    snapshot(p,'previous GPU reservation not settled; bounded wait');continue
                c.write(c.BASE/'driver-stop.json',{'type':type(e).__name__,'reason':str(e),'no_automatic_resubmission':True})
                snapshot(p,'stopped safely; inspect saved reason and receipts');return
            except Exception as e:
                c.write(c.BASE/'driver-error.json',{'type':type(e).__name__,'reason':str(e),'no_automatic_resubmission':True})
                snapshot(p,'error preserved; reconcile exact remote versions before continuation');raise
            snapshot(p,'authorized sequential execution active')
        snapshot(p,'driver wall limit reached; no continued monitoring')


if __name__=='__main__':main()
