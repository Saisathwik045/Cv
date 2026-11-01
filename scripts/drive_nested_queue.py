#!/usr/bin/env python3
"""Approved v11 execution only; bounded polling and charged-quota audit outside worker."""
import argparse,datetime,fcntl,json,math,re,time
from pathlib import Path
import nested_experiment as c


class AccountingPending(RuntimeError):pass
class AccountingBudgetStop(RuntimeError):pass


def seconds(value):
    match=re.fullmatch(r'(\d+(?:\.\d+)?)(?:\.0)?s',str(value))
    if not match:raise ValueError('Unrecognized accelerator-quota duration')
    return float(match.group(1))


def projected_charge(rate):
    # The approved 300-second overhead remains separate from charged runtime.
    return max(2400,2100*rate+300)


def observed_rate_floor_with_guard(charged,worker_seconds):
    # Do not equate worker time and quota. Infer a conservative integer guard
    # only after subtracting the separately reserved allowance from API charges.
    if charged<0 or worker_seconds<=0:raise ValueError('Invalid charged accounting interval')
    return max(1,math.ceil(max(0,charged-300)/worker_seconds))


def next_required(plan,used,rate):
    attempts=[a for r in plan['runs'] for a in r['attempts']]
    reserved=sum(a['reserved_seconds'] for a in attempts)
    next_run=next((r for r in plan['runs'] if r['status']!='complete'),None)
    if next_run is not None and next_run['status']=='prepared':reserved-=next_run['attempts'][-1]['reserved_seconds']
    return max(reserved,used-plan['quota_baseline']['used_seconds'])+projected_charge(rate)


def snapshot(raw,context):
    value={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'context':context,'raw':json.loads(str(raw))}
    folder=c.BASE/'accounting';folder.mkdir(exist_ok=True)
    c.write(folder/('quota-'+value['observed_utc'].replace(':','-')+'.json'),value)
    return value


def record_completed(plan,view):
    folder=c.BASE/'accounting';folder.mkdir(exist_ok=True);path=folder/'ledger.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'intervals':[], 'rate_guard':1,'requested_accelerator':'NvidiaTeslaT4','initial_rate_basis':'Same-shape v9 charged GPU quota evidence; no SDK per-shape multiplier is exposed. API used/reserved counters, not worker time, drive the budget.'}
    gpu=view['raw']['gpuQuota'];used=seconds(gpu['timeUsed'])
    for run in plan['runs']:
        for attempt in run['attempts']:
            if attempt.get('status') not in ('complete','paused','failed') or 'elapsed_seconds' not in attempt:continue
            if any(row['kernel_version']==attempt['kernel_version'] for row in ledger['intervals']):continue
            before=json.loads((c.ROOT/attempt['bundle']/'quota-before.json').read_text())
            # Counter interval ends before the next push; possible billing lag is
            # documented. Final global counter delta is authoritative for total.
            delta=used-before['used_seconds'];rate=observed_rate_floor_with_guard(delta,attempt['elapsed_seconds'])
            ledger['rate_guard']=max(ledger['rate_guard'],rate)
            ledger['intervals'].append({'kernel_version':attempt['kernel_version'],'run':run['id'],
                'gpu_quota_used_before':before['used_seconds'],'gpu_quota_used_at_reconciliation':used,
                'charged_counter_interval_seconds':delta,'worker_elapsed_seconds':attempt['elapsed_seconds'],
                'rate_guard':rate,'counter_attribution':'Account counter interval; may include delayed billing. Not a per-job invoice.',
                'observed_utc':view['observed_utc']})
    c.write(path,ledger);return ledger


def compact_provider_metadata(plan,run,bundle,receipt):
    """Provider identifiers only; scientific run ID, nonce and code stay identical."""
    meta=json.loads((bundle/'kernel-metadata.json').read_text())
    if len(meta['title'])<=50:return
    if receipt['submission_attempted']:raise ValueError('Attempted save requires explicit remote-state recovery before adaptation')
    before_hash=c.file_hash(bundle/'execute.py')
    owner=meta['id'].split('/')[0];arm='sqrtbce' if run['arm']=='sqrt_ratio_bce' else 'ratiobce'
    name=f"trustretina-nested-v11-f{run['outer_fold']}-{arm}-{receipt['nonce'][:8]}"
    if not 5<=len(name)<=50:raise ValueError('Invalid compact provider identifier')
    c.write(bundle/'provider-original-metadata.json',meta)
    previous=receipt['kernel_id'];meta.update(id=owner+'/'+name,title=name);receipt['kernel_id']=meta['id']
    run['attempts'][-1]['kernel_id']=meta['id']
    c.write(bundle/'kernel-metadata.json',meta);c.write(bundle/'submission.json',receipt)
    plan.setdefault('provider_metadata_corrections',[]).append({'run':run['id'],'original_provider_id':previous,'actual_provider_id':meta['id'],'reason':'Compact provider title/slug to <=50 characters; scientific identity/source unchanged'})
    if before_hash!=c.file_hash(bundle/'execute.py') or before_hash!=receipt['source_sha256']:raise ValueError('Metadata adaptation changed source')


def prepare_next(plan,guard,approval):
    run=next((r for r in plan['runs'] if r['status']!='complete'),None)
    if run is None or run['status'] not in ('queued','paused'):return
    if len([a for r in plan['runs'] for a in r['attempts']])>=6:
        raise AccountingBudgetStop('Six reserved sessions reached; no seventh session')
    existing={ref for ref in c.inventory(guard) if 'trustretina-nested-v11-' in ref}
    known={a['kernel_id'] for r in plan['runs'] for a in r['attempts']}
    if existing-known:raise AccountingBudgetStop('Unregistered provider job; recover, never duplicate')
    guard.quota_view()
    bundle,receipt=c.build(plan,run,approval)
    compact_provider_metadata(plan,run,bundle,receipt)
    plan['last_advance_unix']=time.time()
    c.write(c.BASE/'registry.json',plan)


class GuardedAPI:
    def __init__(self,api):self.api=api;self.plan=None;self.before=None
    def __getattr__(self,key):return getattr(self.api,key)
    def quota_view(self):
        raw=self.api.quota_view();view=snapshot(raw,'before_next_submission');gpu=view['raw']['gpuQuota']
        baseline=self.plan['quota_baseline']
        if view['raw'].get('quotaRefreshTime')!=baseline['refresh_time']:raise AccountingBudgetStop('Quota window changed; review accounting')
        if gpu.get('isPayToScaleEnabled') is not False:raise AccountingBudgetStop('Paid scaling is enabled')
        if seconds(gpu['timeReserved'])>0:raise AccountingPending('Previous accelerator reservation has not settled; no new job pushed')
        ledger=record_completed(self.plan,view)
        needed=next_required(self.plan,seconds(gpu['timeUsed']),ledger['rate_guard'])
        if needed>14400:raise AccountingBudgetStop('Actual charged-quota/rate guard cannot cover next session within 14400 seconds')
        if seconds(gpu['totalTimeAllowed'])-seconds(gpu['timeUsed'])-seconds(gpu['timeReserved'])<projected_charge(ledger['rate_guard']):raise AccountingBudgetStop('Free accelerator quota cannot cover guarded next session')
        self.before=view
        return raw
    def kernels_push(self,folder,*,timeout,acc):
        if self.before is None or timeout!='2100' or acc!='NvidiaTeslaT4':raise ValueError('Submission lacks exact accelerator/quota gate')
        folder=Path(folder);g=self.before['raw']['gpuQuota']
        c.write(folder/'quota-before.json',{'observed_utc':self.before['observed_utc'],'raw':self.before['raw'],
            'used_seconds':seconds(g['timeUsed']),'refresh_time':self.before['raw'].get('quotaRefreshTime'),
            'paid_scaling_disabled':g['isPayToScaleEnabled'] is False})
        c.write(folder/'submission-clock.json',{'before_api_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_accelerator':acc,'hard_timeout_seconds':int(timeout)})
        try:
            result=self.api.kernels_push(str(folder),timeout=timeout,acc=acc)
        except Exception as error:
            response=getattr(error,'response',None)
            c.write(folder/'provider-save-error.json',{'error_type':type(error).__name__,'http_status':getattr(response,'status_code',None),'response':getattr(response,'text','')[:10000]})
            raise
        c.write(folder/'submission-response.json',{'ref':result.ref,'version_number':result.version_number,'url':result.url})
        self.before=None;return result


def main(approval_file):
    p=json.loads((c.BASE/'registry.json').read_text());approval=json.loads(approval_file.read_text())
    c.require_approval(p,approval);c.verify_plan(p)
    api=c.transport.api_client();guard=GuardedAPI(api);started=time.monotonic();previous=None
    if 'quota_baseline' not in p:
        raw=api.quota_view();v=snapshot(raw,'initial_baseline');p['quota_baseline']={'used_seconds':seconds(v['raw']['gpuQuota']['timeUsed']),'refresh_time':v['raw'].get('quotaRefreshTime')};c.write(c.BASE/'registry.json',p)
    while time.monotonic()-started<21600:
        if (c.BASE/'STOP_DRIVER').exists():break
        p=json.loads((c.BASE/'registry.json').read_text())
        remaining=p.get('last_advance_unix',0)+180-time.time()
        if remaining>0:
            time.sleep(min(30,remaining));continue
        with (c.BASE/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            p=json.loads((c.BASE/'registry.json').read_text());guard.plan=p
            try:
                active=any(r['status'] in ('running','submitting','ambiguous') for r in p['runs'])
                pending=next((r for r in p['runs'] if r['status']!='complete'),None)
                if active:c.advance(p,guard,submit=False,approval=approval)
                elif pending is not None and pending['status']=='prepared':c.advance(p,guard,submit=True,approval=approval)
                elif pending is None:p['queue_blocker']=None
                elif pending['status'] not in ('queued','paused'):raise AccountingBudgetStop('Unresolved failed/ambiguous state blocks queue')
                if not any(r['status'] in ('running','submitting','ambiguous') for r in p['runs']):prepare_next(p,guard,approval)
                raw=api.quota_view();view=snapshot(raw,'after_reconciliation_or_submission');record_completed(p,view)
                used=seconds(view['raw']['gpuQuota']['timeUsed'])-p['quota_baseline']['used_seconds']
                if used>14400:raise AccountingBudgetStop('Charged counter exceeds approved budget; no further submissions')
                p['charged_quota_seconds_observed']=used
            except AccountingPending as error:
                p['queue_blocker']='Waiting for accelerator accounting settlement'
                print(str(error),flush=True)
            except Exception as error:
                c.write(c.BASE/'driver-error.json',{'type':type(error).__name__,'reason':str(error),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'action':'Driver stopped without resubmission; inspect exact saved receipts/remote versions.'})
                p['queue_blocker']='Driver stopped: '+type(error).__name__
            finally:
                c.write(c.BASE/'registry.json',p);c.export(p)
            state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[{'id':r['id'],'status':r['status'],'epochs':r.get('completed_epochs'),'versions':[a.get('kernel_version',a['kernel_id']) for a in r['attempts']]} for r in p['runs']],
                   'charged_quota_seconds':p.get('charged_quota_seconds_observed'),'reserved_budget_seconds':sum(a['reserved_seconds'] for r in p['runs'] for a in r['attempts']),'blocker':p['queue_blocker']}
            c.write(c.BASE/'driver-latest.json',state)
            compact=json.dumps({k:v for k,v in state.items() if k!='utc'},sort_keys=True)
            if compact!=previous:print(json.dumps(state),flush=True);previous=compact
        if all(r['status']=='complete' for r in p['runs']) or (p['queue_blocker'] and p['queue_blocker']!='Waiting for accelerator accounting settlement'):break
    print('Bounded driver returned; saved registry/receipts remain. No monitoring after exit.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--approval-file',type=Path,required=True);args=parser.parse_args();main(args.approval_file)
