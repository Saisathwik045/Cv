#!/usr/bin/env python3
"""Local-only Mac lifecycle/setup. Standard library bootstrap, no inference changes."""
import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trustretina.inference.bundle import verify_bundle
APP_LOCK = 'requirements/app-macos-arm64.lock'
RECOVERY_LOCK = 'requirements/artifact-recovery.lock'
LOG_KEYS = {'event', 'phase', 'code', 'python', 'machine', 'system', 'port', 'pid', 'lock_sha256', 'error_type', 'timestamp'}


class LaunchError(Exception):
    pass


def secure_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def log_event(root, event, **values):
    # Never persist subprocess output, exceptions, environment, URLs, filenames,
    # credentials or uploaded content. Only controlled lifecycle metadata.
    folder = secure_folder(Path(root) / '.trustretina')
    record = {'event': event, 'timestamp': int(time.time()), **values}
    if set(record) - LOG_KEYS:
        raise ValueError('Unexpected diagnostic field')
    path = folder / 'launcher.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (json.dumps(record) + '\n').encode())
    finally:
        os.close(fd)


def require_mac():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise LaunchError('This launcher requires Apple silicon with native ARM64 Python. Intel/Rosetta is unsupported; install native Python 3.12 from python.org.')
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise LaunchError('Install native Python 3.11 or 3.12 from python.org, then double-click Start_TrustRetina.command again.')
    version = platform.mac_ver()[0]
    if not version or int(version.split('.')[0]) < 13:
        raise LaunchError('The locked Mac runtime requires macOS 13 or newer.')


def read_pins(lock):
    pins = {}
    for line in Path(lock).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', line):
            raise LaunchError('Dependency lock must contain exact package pins only.')
        name, version = line.split('==')
        pins[re.sub(r'[-_.]+', '-', name).lower()] = version
    if not pins:
        raise LaunchError('Dependency lock is empty.')
    return pins


def run_checked(root, args, phase, *, env=None):
    print(f'{phase}…', flush=True)
    log_event(root, 'stage_started', phase=phase)
    # Package indexes and third-party exceptions can include credential-bearing
    # URLs. Do not save or display unfiltered child output.
    process = subprocess.Popen(args, cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        code = process.wait()
    finally:
        stop_child(process)
    log_event(root, 'stage_finished', phase=phase, code=code)
    if code:
        raise LaunchError(f'{phase} failed (exit {code}). Check internet access, trusted certificates and available disk space. For artifact recovery also check your Kaggle token/private-job access. Diagnostic stage/exit codes: .trustretina/launcher.jsonl (no raw subprocess output).')


def environment_identity():
    return {'python': '.'.join(map(str, sys.version_info[:3])), 'machine': platform.machine(), 'system': platform.system()}


def environment_matches(python, pins):
    code = '''import importlib.metadata as m,json,re,sys,platform
expected=json.loads(sys.argv[1]);base=json.loads(sys.argv[2])
if list(sys.version_info[:2]) != base['version'] or platform.machine()!=base['machine']:sys.exit(1)
try:
 for name,version in expected.items():
  if m.version(name)!=version:sys.exit(1)
except m.PackageNotFoundError:sys.exit(1)
'''
    try:
        result = subprocess.run([str(python), '-c', code, json.dumps(pins), json.dumps({'version': list(sys.version_info[:2]), 'machine': platform.machine()})], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except OSError:
        return False


def prepare_environment(root, *, recovery=False):
    root = Path(root)
    lock = root / (RECOVERY_LOCK if recovery else APP_LOCK)
    pins = read_pins(lock)
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    folder = root / ('.venv-artifacts-mac' if recovery else '.venv-mac')
    python = folder / 'bin/python'
    stamp = folder / 'trustretina-lock.json'
    identity = {**environment_identity(), 'lock_sha256': digest}
    try:
        previous = json.loads(stamp.read_text())
    except (OSError, ValueError):
        previous = None
    if previous == identity and environment_matches(python, pins):
        log_event(root, 'environment_reused', phase='recovery' if recovery else 'app', lock_sha256=digest)
        return python
    if folder.exists() and not python.exists():
        raise LaunchError(f'Incomplete virtual environment: {folder.name}. Remove that folder and retry; the inference bundle is separate.')
    if not folder.exists():
        print(f'Creating {folder.name}…', flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    else:
        # A different Python/architecture venv must never be silently reused.
        check = subprocess.run([str(python), '-c', 'import json,sys,platform;print(json.dumps([list(sys.version_info[:2]),platform.machine()]))'], capture_output=True, text=True)
        try:
            compatible = json.loads(check.stdout) == [list(sys.version_info[:2]), platform.machine()]
        except ValueError:
            compatible = False
        if not compatible:
            raise LaunchError(f'{folder.name} belongs to another Python/architecture. Remove only that virtual environment and retry.')
    run_checked(root, [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:', '-r', str(lock)], 'Installing locked recovery dependencies' if recovery else 'Installing locked app dependencies')
    run_checked(root, [str(python), '-m', 'pip', 'check'], 'Checking dependency consistency')
    if not recovery:
        run_checked(root, [str(python), '-c', 'import streamlit,torch,torchvision,numpy,PIL,scipy,sklearn'], 'Verifying native runtime imports')
    if not environment_matches(python, pins):
        raise LaunchError('Installed package versions do not match the lock; no server was started.')
    stamp.write_text(json.dumps(identity, indent=2))
    stamp.chmod(0o600)
    return python


@contextlib.contextmanager
def project_lock(root):
    folder = secure_folder(Path(root) / '.trustretina')
    with (folder / 'launch.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LaunchError('Another setup/launcher is already starting. Wait for its Terminal window; a duplicate server was not started.') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def health_ok(port):
    try:
        # Explicit loopback only; no forwarding, remote destination or proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
            return response.status == 200 and response.read(64).strip() == b'ok'
    except (OSError, urllib.error.URLError):
        return False


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(.3)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def owned_process(record, root):
    if record.get('root') != str(Path(root).resolve()) or not isinstance(record.get('pid'), int):
        return False
    pid = record['pid']
    if pid <= 1:
        return False
    result = subprocess.run(['ps', '-ww', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    command = result.stdout.strip()
    # macOS framework Python can replace argv[0] with Python.app's executable.
    # Identify the saved PID by its exact module/app/loopback/port invocation,
    # rather than requiring the virtualenv symlink to appear as argv[0].
    marker = f" -m streamlit run {Path(root).resolve() / 'streamlit_app.py'} "
    return result.returncode == 0 and marker in command and '--server.address 127.0.0.1 ' in command and f"--server.port {record.get('port')} " in command


def release_identity(root):
    root = Path(root)
    return {'lock_sha256': hashlib.sha256((root / APP_LOCK).read_bytes()).hexdigest(),
            'app_sha256': hashlib.sha256((root / 'streamlit_app.py').read_bytes()).hexdigest()}


def existing_server(root, port):
    state = Path(root) / '.trustretina/server.json'
    try:
        record = json.loads(state.read_text())
    except (OSError, ValueError):
        record = {}
    if owned_process(record, root):
        if any(record.get(k) != v for k, v in release_identity(root).items()):
            raise LaunchError('App entry/dependency lock changed since this server started. Stop its original Terminal before launching the updated version.')
        if record.get('port') != port:
            raise LaunchError(f'This checkout already has a launcher-managed server on port {record["port"]}. Close its Terminal before changing ports.')
        if health_ok(port):
            return record
        raise LaunchError('The existing TrustRetina process is not healthy yet. Inspect its Terminal or stop it before retrying; no duplicate was started.')
    if port_busy(port):
        raise LaunchError(f'Local port {port} is already occupied by an unverified/manual instance. Stop that instance or choose --port; a health response alone does not identify TrustRetina.')
    return None


def wait_for_health(child, port, *, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise LaunchError(f'Streamlit exited before startup (exit {child.returncode}). Check dependencies and whether the local port is occupied.')
        if health_ok(port):
            return
        time.sleep(.25)
    raise LaunchError('Streamlit did not become healthy within 60 seconds. The new process will be stopped; no browser was opened.')


def browser_open(port):
    url = f'http://127.0.0.1:{port}'
    print(f'Local app ready: {url}', flush=True)
    if not webbrowser.open(url, new=2):
        print('Automatic browser opening was unavailable. Open the local address above on this Mac.', flush=True)


def runtime_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('KAGGLE_'):
            del env[key]
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def bundle_location(root):
    path = Path(os.environ.get('TRUSTRETINA_BUNDLE', str(Path(root) / 'artifacts/inference-v8'))).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def start(root, *, port=8501, open_browser=True):
    root = Path(root).resolve()
    child = None
    state = root / '.trustretina/server.json'
    try:
        with project_lock(root):
            bundle = bundle_location(root)
            try:
                verify_bundle(bundle)
            except (OSError, ValueError) as error:
                raise LaunchError('Frozen inference bundle is missing or failed its checksums. Run Setup_TrustRetina.command once, or copy the original verified inference-v8 folder into artifacts. No random weights or server will be used.') from error
            record = existing_server(root, port)
            if record:
                log_event(root, 'server_reused', pid=record['pid'], port=port)
                if open_browser:
                    browser_open(port)
                return 'reused'
            python = prepare_environment(root)
            args = [str(python), '-m', 'streamlit', 'run', str(root / 'streamlit_app.py'), '--server.address', '127.0.0.1', '--server.port', str(port), '--server.headless', 'true', '--server.enableCORS', 'true', '--server.enableXsrfProtection', 'true', '--server.baseUrlPath', '', '--browser.gatherUsageStats', 'false']
            # Raw server output may contain user-supplied exception messages.
            # Keep controlled lifecycle logs only, never raw app/server logs.
            server_env = runtime_environment()
            server_env['TRUSTRETINA_BUNDLE'] = str(bundle)
            child = subprocess.Popen(args, cwd=root, env=server_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record = {'pid': child.pid, 'port': port, 'root': str(root), 'python': str(python), **release_identity(root)}
            state.write_text(json.dumps(record, indent=2))
            state.chmod(0o600)
            log_event(root, 'server_started', pid=child.pid, port=port)
            wait_for_health(child, port)
            log_event(root, 'server_healthy', pid=child.pid, port=port)
            if open_browser:
                browser_open(port)
        print('Keep this Terminal window open. Press Control-C to stop TrustRetina. No public hosting is used.', flush=True)
        code = child.wait()
        if code:
            raise LaunchError(f'The local server stopped (exit {code}); see .trustretina/launcher.jsonl for lifecycle diagnostics.')
        return 'started'
    finally:
        if child is not None:
            stop_child(child)
            try:
                if json.loads(state.read_text()).get('pid') == child.pid:
                    state.unlink()
            except (OSError, ValueError):
                pass
            log_event(root, 'server_stopped', pid=child.pid, port=port)


def check_credential_permissions():
    config = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle'))).expanduser()
    for name in ('access_token', 'kaggle.json'):
        path = config / name
        if path.exists() and (path.is_symlink() or path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077):
            raise LaunchError('Existing Kaggle credential file is not private/owned by this user. Restrict it to your account (chmod 600) or use the hidden token prompt; never commit credentials.')


def setup_artifacts(root):
    root = Path(root).resolve()
    with project_lock(root):
        bundle = bundle_location(root)
        if bundle.exists():
            try:
                verify_bundle(bundle)
            except (OSError, ValueError) as error:
                raise LaunchError('Existing bundle failed verification. Preserve/move that folder aside and restore the exact pinned bundle; it is never overwritten silently.') from error
            print('Frozen bundle already verified. Double-click Start_TrustRetina.command.', flush=True)
            return
        python = prepare_environment(root, recovery=True)
        print('Use a Kaggle API token from your authorized account (Kaggle Settings → API). It is not saved or logged.', flush=True)
        if not sys.stdin.isatty():
            raise LaunchError('Private setup needs an interactive Terminal for hidden token entry. Double-click Setup_TrustRetina.command, or copy the verified bundle without credentials.')
        token = getpass.getpass('Paste API token (hidden), or press Return to use existing credentials: ').strip()
        env = os.environ.copy()
        if token:
            env['KAGGLE_API_TOKEN'] = token
        else:
            check_credential_permissions()
        try:
            run_checked(root, [str(python), str(root / 'scripts/install_inference_bundle.py'), '--download', '--destination', str(bundle)], 'Restoring pinned private inference artifacts', env=env)
        finally:
            env.pop('KAGGLE_API_TOKEN', None)
            token = None
        verify_bundle(bundle)
        log_event(root, 'bundle_verified')
        print('Frozen bundle installed and verified. Double-click Start_TrustRetina.command. Inference no longer requires Kaggle credentials.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-artifacts', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    try:
        require_mac()
        if not 1024 <= args.port <= 65535:
            raise LaunchError('Choose a local port from 1024 through 65535.')
        log_event(ROOT, 'launcher_started', **environment_identity())
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        if args.setup_artifacts:
            setup_artifacts(ROOT)
        else:
            start(ROOT, port=args.port, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print('\nTrustRetina stopped. No training or calibration occurred.', flush=True)
        return 0
    except LaunchError as error:
        log_event(ROOT, 'launcher_failed', error_type=type(error).__name__)
        print(f'\nTrustRetina: {error}', file=sys.stderr)
        return 1
    except Exception as error:
        log_event(ROOT, 'launcher_failed', error_type=type(error).__name__)
        print(f'\nTrustRetina setup/start failed ({type(error).__name__}). Check disk space, file permissions and .trustretina/launcher.jsonl. Raw exceptions are omitted to protect credentials and image contents.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
