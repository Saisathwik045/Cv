"""Safe local preflight; no model deserialization, private rows or external probes."""
import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def loopback_health(port):
    if type(port) is not int or not 1024 <= port <= 65535:
        return {'status': 'FAIL', 'detail': 'Invalid local port; no request made'}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as response:
            if response.status == 200 and response.read(64).strip() == b'ok':
                return {'status': 'PASS', 'detail': 'Loopback health replied ok; this alone does not prove process identity or external access'}
    except (OSError, urllib.error.URLError):
        pass
    return {'status': 'FAIL', 'detail': 'No expected loopback health response; redirects are not followed'}


def dependency_check(root=ROOT, version=importlib.metadata.version):
    lock = Path(root) / 'requirements' / ('app-macos-arm64.lock' if platform.system() == 'Darwin' else 'app-cpu.lock')
    mismatches = []
    count = 0
    for line in lock.read_text().splitlines():
        if not line or line.startswith(('#', '--')):
            continue
        name, expected = line.split('==', 1)
        count += 1
        try:
            actual = version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = 'missing'
        if actual != expected:
            mismatches.append({'package': name, 'expected': expected, 'actual': actual})
    return {'status': 'PASS' if not mismatches else 'FAIL', 'checked_pins': count, 'mismatches': mismatches}


def bundle_check(bundle):
    from trustretina.inference.bundle import verify_bundle
    try:
        c = verify_bundle(bundle)  # Hashes/contract JSON only; never torch.load.
    except (OSError, ValueError):
        return {'status': 'FAIL', 'detail': 'Original bundle missing or invalid; no fallback or automatic recovery'}
    return {'status': 'PASS', 'verified_files': 5, 'model': c['selected_backbone'],
            'epoch': c['best_epoch'], 'resolution': c['preprocessing']['size'],
            'temperature': c['temperature'], 'threshold': c['threshold'],
            'decision': c['deployment_default']}


def collect_readiness(bundle, *, root=ROOT, port=8501, address='127.0.0.1', cors=True, xsrf=True):
    from dashboard.academic import load_academic_evidence
    try:
        load_academic_evidence(root)
        evidence = {'status': 'PASS', 'detail': 'Public V15 aggregate and four reviewed figures match their hashes'}
    except (OSError, ValueError, KeyError):
        evidence = {'status': 'FAIL', 'detail': 'Public V15 evidence unavailable or invalid'}
    checks = {'python': {'status': 'PASS' if sys.version_info[:2] in ((3, 11), (3, 12)) else 'FAIL', 'version': platform.python_version()},
              'dependencies': dependency_check(root), 'bundle': bundle_check(bundle),
              'public_evidence': evidence, 'loopback_health': loopback_health(port),
              'server_binding': {'status': 'PASS' if address == '127.0.0.1' else 'FAIL', 'detail': 'Configured loopback-only address required'},
              'cors': {'status': 'PASS' if cors is True else 'FAIL'},
              'xsrf': {'status': 'PASS' if xsrf is True else 'FAIL'}}
    passed = all(c['status'] == 'PASS' for c in checks.values())
    return {'schema': 'cloud_demo_preflight_v1', 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'local_preflight': 'PASS' if passed else 'FAIL', 'checks': checks,
            'free_disk_bytes': shutil.disk_usage(root).free,
            'checkpoint_loading': 'NOT RUN', 'inference': 'NOT RUN',
            'external_viewing_route': 'NOT VERIFIED', 'uptime': 'NOT GUARANTEED',
            'live_demonstration_readiness': 'CONDITIONAL' if passed else 'BLOCKED',
            'clinical_validation': False, 'decision': 'REVIEW REQUIRED'}


def render_readiness_page():
    import os
    import pandas as pd
    import streamlit as st
    from trustretina.inference.bundle import DEFAULT_BUNDLE
    st.title('Cloud readiness')
    st.warning('Local software checks are not clinical validation, proof of external access or an uptime guarantee.')
    st.write('Check the original five-file bundle, exact dependency pins, public V15 evidence, local health response and server protections before a presentation. This reads model bytes for hashing only; it never loads weights, fits, runs inference or downloads missing artifacts.')
    if st.button('Run local readiness checks', type='primary'):
        st.session_state.pop('readiness_report', None)
        try:
            with st.spinner('Checking local integrity and loopback health…'):
                st.session_state['readiness_report'] = collect_readiness(
                    os.environ.get('TRUSTRETINA_BUNDLE', str(DEFAULT_BUNDLE)),
                    port=st.get_option('server.port'), address=st.get_option('server.address'),
                    cors=st.get_option('server.enableCORS'), xsrf=st.get_option('server.enableXsrfProtection'))
        except (OSError, ValueError):
            st.error('Local checks could not complete. No readiness result is assumed; inspect the local setup without sharing credentials or private data.')
    report = st.session_state.get('readiness_report')
    if report:
        if report['local_preflight'] == 'PASS':
            st.success('Local preflight PASS. Live demonstration remains CONDITIONAL on a working viewing route and current runtime.')
        else:
            st.error('Local preflight FAIL. Resolve the failed checks before the live demonstration.')
        st.dataframe(pd.DataFrame([{'Check': name, 'Status': value['status']} for name, value in report['checks'].items()]), hide_index=True, width='stretch')
        st.caption('Snapshot time (UTC): ' + report['observed_utc'] + '. Conditions can change after this check.')
        st.json(report)
        st.download_button('Download local readiness snapshot', json.dumps(report, indent=2),
                           file_name='trustretina-local-preflight.json', mime='application/json')
    st.subheader('Viewing and presentation checklist')
    st.write('External viewing route: NOT VERIFIED by this page. Open the app through your cloud Desktop viewer if your client supports it, and rehearse that route from the presenting screen. A cloud 127.0.0.1 address opened on your laptop points to your laptop, not this workspace.')
    st.write('Health replies alone do not authenticate a process or test proxy/WebSocket/upload access. This check never requests an arbitrary URL or follows a redirect. No tunnel or public deployment is created.')
    st.write('Use Guided synthetic demo for the current operational model check. Keep non-patient inputs visibly labelled, test export/reset, and present the recorded V15 findings separately. Native Mac behavior, durable backup and clinical validation are not checked here.')
    st.write('The app may stop when the cloud runtime pauses or shuts down. No permanent hosting, durable storage, background monitoring or guaranteed uptime is configured by these checks.')
