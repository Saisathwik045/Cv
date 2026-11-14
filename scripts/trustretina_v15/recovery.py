"""Selective, separately authorized V14 recovery. No fitting or job operations.

This module is stdlib-only until an explicitly authorized transport is created.
"""
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import stat
import uuid

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_SHA256 = 'da8fc216144e5f233c4e58363632c53d7a43d1d8c6e07300ca366d965f7aafce'
FROZEN_SHA256 = '43351a86118e6465a0e24939faddecd9b09d8c8af49d1122c8f0f47f197749e9'
SCOPE = 'v15_selective_artifact_recovery'
FIXTURE_SCOPE = 'v15_synthetic_artifact_recovery'


class RecoveryError(ValueError):
    """Only static messages may cross the CLI/log boundary."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_plan(registry, frozen, *, profile='metadata', fixture=False):
    """Public inventory only. The existing loader's extra hash dependencies remain explicit."""
    if profile not in ('metadata', 'analysis'):
        raise RecoveryError('Unknown recovery profile')
    objects = []; runs = []
    receipts = list(frozen['receipts'].values())
    for fit in registry['fits']:
        version = fit['kernel_version']
        receipt = next(r for r in receipts if r['kernel_version'] == version)
        checkpoint = fit['artifact_sha256']['best.pt']
        runs.append({'run_id': fit['run_id'], 'version': version, 'source_commit': fit['source_commit'],
                     'nonce': receipt['nonce'], 'checkpoint_sha256': checkpoint,
                     'artifact_sha256': fit['artifact_sha256']})
        files = {'stage-report.json': fit['report_sha256'], **fit['artifact_sha256']}
        for name, sha in files.items():
            if profile == 'metadata' and name in ('best.pt', 'last.pt'):
                continue
            objects.append({'version': version, 'remote': 'full/'+name,
                            'local': fit['run_id']+'/'+name, 'sha256': sha,
                            'purpose': 'opaque_checkpoint_hash_required_by_existing_v15_loader' if name.endswith('.pt')
                            else 'v15_prediction_quality_or_integrity_dependency'})
    return {'schema': 1, 'data_kind': 'synthetic_fixture_not_remote_evidence' if fixture else 'real_v14',
            'profile': profile, 'runs': runs, 'objects': objects,
            'role_manifest_sha256': frozen['registry']['role_manifest_sha256'],
            'not_recovered': ['original_role_manifests_and_labeled_rows_require_separate_verified_source',
                              'raw_images', 'training_tensors', 'imagenet_weights', 'historical_final', 'logs'],
            'calibration_fitting_authorized': False}


def public_plan(profile='metadata'):
    base = ROOT/'results/candidate-v14'
    for name, sha in [('verified-artifact-registry.json', REGISTRY_SHA256), ('frozen-plan.json', FROZEN_SHA256)]:
        if digest(base/name) != sha:
            raise RecoveryError('Public V14 registry integrity check failed')
    return build_plan(json.loads((base/'verified-artifact-registry.json').read_text()),
                      json.loads((base/'frozen-plan.json').read_text()), profile=profile)


def authorize(plan, approval, root, *, fixture=False):
    expected = {'approved': True, 'scope': FIXTURE_SCOPE if fixture else SCOPE,
                'plan_sha256': identity(plan), 'private_root': str(Path(root).absolute()),
                'allow_checkpoint_bytes': plan['profile'] == 'analysis',
                'calibration_fitting_authorized': False}
    if not approval or any(approval.get(k) != v for k, v in expected.items()) or not approval.get('user_approval_reference'):
        raise PermissionError('Separate exact recovery authorization required')
    if not fixture and approval.get('credential_access_authorized') is not True:
        raise PermissionError('Credential activation needs explicit recovery authorization')
    for key in ('maximum_file_bytes', 'maximum_transfer_bytes'):
        if type(approval.get(key)) is not int or approval[key] <= 0:
            raise PermissionError('Explicit positive recovery byte budgets required')
    hosts = approval.get('download_hosts')
    if not isinstance(hosts, list) or not hosts or any(not isinstance(h, str) or not h or '/' in h or '*' in h for h in hosts):
        raise PermissionError('Exact download host authorization required')


def checked_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise RecoveryError('Unsafe recovery path')
    p = Path(root)
    for part in relative.parts:
        p = p/part
        if p.is_symlink():
            raise RecoveryError('Recovery symlinks forbidden')
    return p


def private_root(root):
    root = Path(root).absolute()
    if '..' in root.parts:
        raise RecoveryError('Recovery root traversal forbidden')
    if root == ROOT or ROOT in root.parents:
        raise RecoveryError('Private recovery must be outside the repository')
    # Refuse symlink parents as well as a symlink final directory.
    p = Path(root.anchor)
    for part in root.parts[1:]:
        p = p/part
        if p.is_symlink():
            raise RecoveryError('Recovery symlinks forbidden')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise RecoveryError('Recovery directory must be owned by the user with mode 0700')
    return root


def private_write(path, value):
    temp = checked_path(path.parent, '.state-'+uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


@contextlib.contextmanager
def quiet_transport():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def verify_provenance(plan, root):
    for run in plan['runs']:
        folder = root/run['run_id']
        stage = json.loads((folder/'stage-report.json').read_text())
        selected = json.loads((folder/'selected-checkpoint.json').read_text())
        spec = json.loads((folder/'experiment-spec.json').read_text())
        if (stage.get('status') != 'complete' or stage.get('run_id') != run['run_id']
                or stage.get('nonce') != run['nonce'] or stage.get('source_commit') != run['source_commit']
                or stage.get('output_hashes') != run['artifact_sha256']
                or stage.get('selected_checkpoint_sha256') != run['checkpoint_sha256']
                or not all(stage.get(k) is True for k in ('strict_best_reload', 'model_state_invariance', 'calibration_inference_after_freeze'))
                or stage.get('final_evaluation_access') is not False
                or stage.get('calibration_labels_accessed') is not False
                or selected.get('checkpoint_sha256') != run['checkpoint_sha256']
                or selected.get('decision_threshold') != .5
                or selected.get('patient_aggregation') != 'mean_left_right_sigmoid_before_threshold'
                or selected.get('selection_role') != 'validation'
                or stage.get('provenance') != selected.get('provenance')
                or selected.get('provenance', {}).get('run_id') != run['run_id']
                or spec.get('run_id') != run['run_id']
                or spec.get('config', {}).get('class_order') != list('NDGCAHMO')
                or spec.get('role_manifest_sha256') != plan['role_manifest_sha256']):
            raise RecoveryError('Recovered V14 provenance mismatch')


def _execute(plan, expected, approval, root, factory, *, fixture=False):
    if plan != expected:
        raise RecoveryError('Recovery plan differs from trusted pinned inventory')
    authorize(plan, approval, root, fixture=fixture)
    root = private_root(root)
    lockpath = checked_path(root, 'recovery.lock')
    fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RecoveryError('Another recovery holds the directory lock') from None
        statepath = checked_path(root, 'recovery-state.json')
        complete = checked_path(root, 'recovery-complete.json')
        if statepath.exists() and json.loads(statepath.read_text()).get('plan_sha256') != identity(plan):
            raise RecoveryError('Existing private directory belongs to another plan')
        state = {'plan_sha256': identity(plan), 'status': 'incomplete', 'verified': [],
                 'data_kind': plan['data_kind'], 'calibration_fitting_authorized': False,
                 'role_alignment_verified': False, 'v15_fitting_ready': False}
        private_write(statepath, state)
        client = None
        # Conservatively charge retained partial bytes too, across invocations.
        transferred = 0
        for item in plan['objects']:
            target = checked_path(root, item['local'])
            candidates = [target, *target.parent.glob(target.name+'.partial-*')]
            for path in candidates:
                if path.is_symlink():
                    raise RecoveryError('Recovery symlinks forbidden')
                if path.is_file():
                    transferred += path.stat().st_size
        try:
            for item in plan['objects']:
                target = checked_path(root, item['local'])
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if stat.S_IMODE(target.parent.stat().st_mode) != 0o700:
                    raise RecoveryError('Private artifact directory permissions are too broad')
                if not target.exists():
                    for partial in sorted(target.parent.glob(target.name+'.partial-*')):
                        if stat.S_IMODE(partial.stat().st_mode) != 0o600:
                            raise RecoveryError('Private partial permissions are too broad')
                        if digest(partial) == item['sha256']:
                            os.replace(partial, target); break
                if target.exists():
                    if stat.S_IMODE(target.stat().st_mode) != 0o600 or digest(target) != item['sha256']:
                        raise RecoveryError('Existing artifact changed; preserved without overwrite')
                else:
                    if complete.exists():
                        raise RecoveryError('Previously completed recovery lost an artifact; no automatic refetch')
                    remaining = approval['maximum_transfer_bytes']-transferred
                    limit = min(approval['maximum_file_bytes'], remaining)
                    if limit <= 0:
                        raise RecoveryError('Recovery transfer budget exhausted')
                    if client is None:
                        with quiet_transport():
                            client = factory()
                    part = checked_path(root, item['local']+'.partial-'+uuid.uuid4().hex)
                    # Transport must create exactly this file; no arbitrary directory download.
                    with quiet_transport():
                        client.download(item['version'], item['remote'], part, limit)
                    if part.is_symlink() or not part.is_file() or part.stat().st_size > limit:
                        raise RecoveryError('Transport returned an invalid or oversized object')
                    transferred += part.stat().st_size
                    if digest(part) != item['sha256']:
                        raise RecoveryError('Recovered artifact SHA256 mismatch')
                    part.chmod(0o600)
                    os.replace(part, target)
                state['verified'].append(item['local']); private_write(statepath, state)
            verify_provenance(plan, root)
            state['status'] = 'recovered_bytes_and_provenance_verified'
            private_write(statepath, state); private_write(complete, state)
            return {'verified_objects': len(state['verified']), 'profile': plan['profile'],
                    'data_kind': plan['data_kind'], 'recovery_complete': True,
                    'role_alignment_verified': False, 'calibration_fitting_authorized': False}
        except (Exception, KeyboardInterrupt):
            state['status'] = 'interrupted_or_blocked'; private_write(statepath, state)
            # Preserve partial/verified bytes. Never include provider exceptions or URLs.
            raise RecoveryError('Recovery interrupted or blocked; private state preserved') from None


def recover(plan, approval, root, *, transport_factory=None):
    """Real path: exact public inventory and approval checked before SDK construction."""
    expected = public_plan(plan.get('profile'))
    factory = transport_factory or (lambda: KaggleTransport(approval['download_hosts']))
    return _execute(plan, expected, approval, root, factory)


def recover_fixture(plan, approval, root, transport):
    """Synthetic transport only; fixture permission cannot activate real Kaggle."""
    if plan.get('data_kind') != 'synthetic_fixture_not_remote_evidence' or isinstance(transport, KaggleTransport):
        raise RecoveryError('Synthetic recovery requires fixture transport')
    if any(not r['version'].startswith('fixture/') for r in plan['runs']):
        raise RecoveryError('Synthetic source versions required')
    return _execute(plan, plan, approval, root, lambda: transport, fixture=True)


class KaggleTransport:
    """Low-level listing + bounded HTTPS streams. Never calls kernels_output or logs."""
    def __init__(self, hosts):
        self.hosts = hosts
        self.allowed = {(o['version'], o['remote']) for o in public_plan('analysis')['objects']}
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import kaggle
                self.api = kaggle.api
                if getattr(self.api, '_authenticated', False) is not True:
                    raise RecoveryError('No activated Kaggle identity')
        except (Exception, SystemExit):
            raise RecoveryError('Kaggle credential activation unavailable; no download attempted') from None
        finally:
            logging.disable(previous)

    def download(self, version, remote, target, limit):
        if (version, remote) not in self.allowed:
            raise RecoveryError('Transport object is outside the pinned inventory')
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        import requests
        from urllib.parse import urlparse, urljoin
        owner, slug, number = version.split('/')
        token = ''; matches = []
        with self.api.build_kaggle_client() as client:
            for _ in range(10):
                request = ApiListKernelSessionOutputRequest()
                request.user_name = owner; request.kernel_slug = slug
                request.version_label = 'v'+number; request.page_size = 100; request.page_token = token
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                matches.extend(item for item in response.files or [] if item.file_name == remote)
                token = response.next_page_token
                if not token:
                    break
            else:
                raise RecoveryError('Kaggle output pagination exceeded cap')
        if len(matches) != 1:
            raise RecoveryError('Requested pinned object missing or duplicated')
        url = matches[0].url
        # Validate every redirect BEFORE requesting the next host. Never log signed URLs.
        for _ in range(6):
            parsed = urlparse(url)
            if parsed.scheme != 'https' or parsed.hostname not in self.hosts or parsed.username or parsed.password or parsed.port not in (None, 443):
                raise RecoveryError('Download host is outside the approved HTTPS list')
            # Explicit no-op auth prevents requests from consulting ~/.netrc.
            # Inherited proxy/CA settings still apply to the signed HTTPS URL.
            with requests.get(url, stream=True, timeout=(20, 60), allow_redirects=False,
                              auth=lambda request: request) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get('Location', '')); continue
                response.raise_for_status()
                length = response.headers.get('Content-Length')
                if length is not None and (int(length) < 0 or int(length) > limit):
                    raise RecoveryError('Remote object exceeds approved byte limit')
                count = 0
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    for block in response.iter_content(chunk_size=1024*1024):
                        count += len(block)
                        if count > limit:
                            raise RecoveryError('Remote object exceeds approved byte limit')
                        stream.write(block)
                    stream.flush(); os.fsync(stream.fileno())
                if length is not None and count != int(length):
                    raise RecoveryError('Incomplete remote transfer')
                return
        raise RecoveryError('Download redirect cap exceeded')
