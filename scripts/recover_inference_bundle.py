#!/usr/bin/env python3
"""Bounded original-demo recovery only. No Torch, model loading, inference or logs."""
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trustretina.inference.bundle import CONTRACT_PATH, required_files, verified_contract, verify_bundle

HOST = 'www.kaggleusercontent.com'
MAX_TRANSFER = 668_483_988
MIN_FREE = 2 * 1024 ** 3
SIZES = {'best.pt': 334_159_877, 'temperature.json': 1_020,
         'quality-normalizer.npz': 79_547, 'candidate-pipeline.json': 1_550}
VERSIONS = ('saisathwikregula/trustretina-full-v5-convnext-tiny-c349c706/1',
            'saisathwikregula/trustretina-phase6-analysis-ab40b8f2/1')


class RecoveryError(Exception):
    """Controlled diagnostics; never include URLs or third-party exception text."""


def require(condition, message):
    if not condition:
        raise RecoveryError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def build_plan():
    c = verified_contract()
    hashes = required_files(c)
    require((c['selected_checkpoint_kernel_version'], c['artifact_storage']['analysis_kernel_version']) == VERSIONS,
            'Frozen source versions changed')
    record = json.loads((ROOT / 'results/demo-v8/artifact-recovery.json').read_text())
    require(record['files'] == hashes, 'Committed recovery hashes differ')
    return {'objects': [{'name': name, 'remote': 'full/' + name,
                         'version': VERSIONS[0 if name == 'best.pt' else 1],
                         'size': size, 'sha256': hashes[name]} for name, size in SIZES.items()],
            'contract_sha256': hashes['candidate-contract.json'], 'expected_private_bytes': sum(SIZES.values()),
            'recovery_tool_sha256': digest(Path(__file__)),
            'dependency_lock_sha256': digest(ROOT / 'requirements/artifact-recovery.lock'),
            'maximum_transfer_bytes': MAX_TRANSFER, 'minimum_free_bytes': MIN_FREE,
            'maximum_attempts_per_object': 2, 'download_hosts': [HOST]}


def no_symlinks(path):
    for p in [path, *path.parents]:
        require(not os.path.lexists(p) or not p.is_symlink(), 'Symlink path rejected')


def private_path(path, directory=False):
    s = path.lstat()
    require(s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == (0o700 if directory else 0o600)
            and (stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)),
            'Private ownership, type or permissions failed')


def private_directory(path):
    no_symlinks(path)
    missing, p = [], path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for p in reversed(missing):
        p.mkdir(mode=0o700)
    private_path(path, directory=True)


def write_json(path, value):
    if os.path.lexists(path):
        private_path(path)
    temporary = path.with_name(path.name + '.tmp')
    if os.path.lexists(temporary):
        private_path(temporary)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def validate_url(url):
    p = urlparse(url)
    require(p.scheme == 'https' and p.hostname == HOST and not p.username and not p.password
            and p.port in (None, 443), 'Unapproved HTTPS download URL')


class KaggleTransport:
    def __init__(self):
        # Import/activate only after local invocation, plan, disk and storage gates.
        # SDK output and all raw exceptions stay out of terminal and receipts.
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from kaggle import api
                require(getattr(api, '_authenticated', False) is True
                        and api.get_config_value(api.CONFIG_NAME_USER) == 'saisathwikregula',
                        'Kaggle account does not match the approved source owner')
                self.api = api
        except (Exception, SystemExit):
            raise RecoveryError('Kaggle authentication unavailable or account mismatch; no download attempted') from None
        finally:
            logging.disable(previous)
        self.allowed = {(o['version'], o['remote']) for o in build_plan()['objects']}

    def url(self, version, remote):
        require((version, remote) in self.allowed, 'Object outside the pinned recovery scope')
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        owner, slug, number = version.split('/')
        matches, token = [], ''
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.api.build_kaggle_client() as client:
                for _ in range(10):
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    request.version_label, request.page_size, request.page_token = 'v' + number, 100, token
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                    matches.extend(item.url for item in response.files or [] if item.file_name == remote)
                    token = response.next_page_token
                    if not token:
                        break
                else:
                    raise RecoveryError('Pinned metadata pagination limit exceeded')
        finally:
            logging.disable(previous)
        require(len(matches) == 1, 'Pinned object missing or duplicated')
        validate_url(matches[0])
        return matches[0]

    def open(self, url):
        import requests
        validate_url(url)
        class NoAuth(requests.auth.AuthBase):
            def __call__(self, request):
                return request
        # A fresh session has no retry policy or SDK credentials. Keep inherited proxy/CA trust.
        return requests.get(url, headers={'Accept-Encoding': 'identity'}, auth=NoAuth(),
                            stream=True, allow_redirects=False, timeout=(10, 30))


def receive(response, output, obj, state, save):
    # Reject even approved-host redirects: no redirect response body is consumed.
    require(response.status_code == 200, 'Unexpected HTTP status or redirect; body rejected')
    require(response.headers.get('Content-Length') == str(obj['size']), 'Unexpected response size; body rejected')
    require(response.headers.get('Content-Encoding', 'identity').lower() == 'identity'
            and not response.headers.get('Transfer-Encoding'), 'Unexpected encoding; body rejected')
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    h, count = hashlib.sha256(), 0
    with os.fdopen(fd, 'wb') as f:
        while count < obj['size']:
            amount = min(1024 * 1024, obj['size'] - count)
            require(state['charged_bytes'] + amount <= state['maximum_transfer_bytes'], 'Cumulative transfer budget exhausted')
            # Charge before the read. An interrupted read remains conservatively charged.
            state['charged_bytes'] += amount
            save()
            block = response.raw.read(amount, decode_content=False)
            require(block and len(block) <= amount, 'Incomplete or oversized response body')
            state['known_transferred_bytes'] += len(block)
            state['charged_bytes'] -= amount - len(block)
            save()
            f.write(block)
            h.update(block)
            count += len(block)
        f.flush()
        os.fsync(f.fileno())
    require(count == obj['size'] and h.hexdigest() == obj['sha256'], 'Downloaded file SHA256 mismatch')


def recover(destination, *, approved=False, transport_factory=KaggleTransport,
            plan=None, contract_path=CONTRACT_PATH, verifier=verify_bundle):
    require(approved, 'Explicit --allow-recovery invocation required')
    plan = build_plan() if plan is None else plan
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    no_symlinks(destination)
    if os.path.lexists(destination):
        private_path(destination, directory=True)
        for p in destination.iterdir():
            private_path(p)
        verifier(destination)
        return {'status': 'already_verified', 'downloaded_files': 0, 'bundle': str(destination)}
    require(shutil.disk_usage(destination.parent if destination.parent.exists() else Path.home()).free >= plan['minimum_free_bytes'], 'At least 2 GiB free space required before recovery')
    private_directory(destination.parent)
    work = destination.parent / ('.' + destination.name + '-recovery')
    try:
        work.mkdir(mode=0o700)
        fresh = True
    except FileExistsError:
        fresh = False
    no_symlinks(work)
    private_path(work, directory=True)
    lock = work / 'recovery.lock'
    if os.path.lexists(lock):
        private_path(lock)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RecoveryError('Another recovery is already active') from None
        plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
        state_file = work / 'state.json'
        if fresh:
            state = {'plan_sha256': plan_hash, 'destination': str(destination),
                     'known_transferred_bytes': 0, 'charged_bytes': 0,
                     'maximum_transfer_bytes': plan['maximum_transfer_bytes'], 'attempts': {}, 'blocked': False}
            write_json(work / 'approval.json', {'scope': 'original_inference_v8_four_file_recovery',
                       'reference': 'Explicit local --allow-recovery invocation', 'plan_sha256': plan_hash,
                       'destination': str(destination), 'checkpoint_loading_authorized': False,
                       'inference_authorized': False})
            write_json(work / 'plan.json', plan)
            write_json(state_file, state)
        else:
            private_path(state_file)
            state = json.loads(state_file.read_text())
            require(state['plan_sha256'] == plan_hash and state['destination'] == str(destination)
                    and state['maximum_transfer_bytes'] == plan['maximum_transfer_bytes'], 'Recovery state/plan binding changed')
        require(not state['blocked'], 'Earlier integrity/security failure requires review; existing evidence preserved')
        require(0 <= state['known_transferred_bytes'] <= state['charged_bytes'] <= plan['maximum_transfer_bytes']
                and set(state['attempts']).issubset({o['name'] for o in plan['objects']})
                and all(isinstance(v, int) and 0 <= v <= plan['maximum_attempts_per_object']
                        for v in state['attempts'].values()), 'Invalid transfer accounting state')
        save = lambda: write_json(state_file, state)
        inputs = work / 'inputs'
        private_directory(inputs)
        try:
            transport = None
            for obj in plan['objects']:
                name, size, expected = obj['name'], obj['size'], obj['sha256']
                target = inputs / name
                if os.path.lexists(target):
                    private_path(target)
                    require(target.stat().st_size == size and digest(target) == expected, 'Existing recovered artifact mismatch; not overwritten')
                    continue
                attempts = state['attempts'].get(name, 0)
                # A fully downloaded, verified partial can survive a process interruption.
                for i in range(1, attempts + 1):
                    partial = inputs / f'{name}.attempt-{i}.partial'
                    if os.path.lexists(partial):
                        private_path(partial)
                        if partial.stat().st_size == size and digest(partial) == expected:
                            partial.rename(target)
                            break
                if target.exists():
                    continue
                require(attempts < plan['maximum_attempts_per_object'], 'Per-object retry limit reached')
                require(state['charged_bytes'] + size <= plan['maximum_transfer_bytes'], 'Insufficient remaining transfer budget')
                if transport is None:
                    transport = transport_factory()
                url = transport.url(obj['version'], obj['remote'])
                validate_url(url)
                state['attempts'][name] = attempts + 1
                save()
                partial = inputs / f'{name}.attempt-{attempts + 1}.partial'
                with contextlib.closing(transport.open(url)) as response:
                    receive(response, partial, obj, state, save)
                private_path(partial)
                require(digest(partial) == expected, 'On-disk SHA256 mismatch')
                partial.rename(target)
                print('Verified selected file ' + name, flush=True)
            require(digest(contract_path) == plan['contract_sha256'], 'Committed candidate contract changed')
            staging = work / 'bundle-staging'
            private_directory(staging)
            sources = [(inputs / o['name'], o['sha256']) for o in plan['objects']]
            sources.append((Path(contract_path), plan['contract_sha256']))
            for source, expected in sources:
                target = staging / source.name
                if os.path.lexists(target):
                    private_path(target)
                    require(digest(target) == expected, 'Existing staged artifact mismatch')
                else:
                    out_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    with source.open('rb') as src, os.fdopen(out_fd, 'wb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                require(digest(target) == expected, 'Assembled file SHA256 mismatch')
            require({p.name for p in staging.iterdir()} == {p.name for p, _ in sources}, 'Unexpected bundle members')
            verifier(staging)  # Only hashes/contract JSON; no Torch or weight deserialization.
            no_symlinks(destination)
            require(not os.path.lexists(destination), 'Existing final bundle preserved')
            staging.rename(destination)
            receipt = {'status': 'verified', 'bundle': str(destination), 'files': 5,
                       'source_bytes': plan['expected_private_bytes'],
                       'known_transferred_bytes': state['known_transferred_bytes'],
                       'conservatively_charged_bytes': state['charged_bytes'], 'plan_sha256': plan_hash,
                       'checkpoint_loaded': False, 'inference_run': False}
            write_json(work / 'complete.json', receipt)
            return receipt
        except RecoveryError:
            state['blocked'] = True
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Public plan only; no authentication or files')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--allow-recovery', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    try:
        if args.plan:
            print(json.dumps(build_plan(), indent=2))
        else:
            require(args.destination is not None, 'An explicit private destination is required')
            result = recover(args.destination, approved=args.allow_recovery)
            print(json.dumps(result))
        return 0
    except RecoveryError as error:
        print('Recovery stopped: ' + str(error), file=sys.stderr)
    except (Exception, SystemExit):
        print('Recovery stopped: dependency, network or storage failure. No raw URL/credential error is displayed; retain the private recovery state before retrying.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
