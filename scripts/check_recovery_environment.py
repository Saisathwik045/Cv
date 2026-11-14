"""Verify locked recovery packages and SDK imports without credential discovery.

SDK authentication is replaced BEFORE kaggle.__init__ can invoke it. The child
has a new empty HOME/config directory and no inherited credential variables.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECK = r'''
import importlib.abc, importlib.machinery, sys, socket, json, contextlib, io, inspect
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Network disabled'))
class Loader(importlib.abc.Loader):
    def __init__(self, wrapped): self.wrapped = wrapped
    def create_module(self, spec): return self.wrapped.create_module(spec)
    def exec_module(self, module):
        self.wrapped.exec_module(module)
        def disabled(self): raise PermissionError('Credential discovery disabled')
        module.KaggleApi.authenticate = disabled
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kaggle.api.kaggle_api_extended':
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            spec.loader = Loader(spec.loader)
            return spec
sys.meta_path.insert(0, Finder())
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle, kagglesdk
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    request = ApiListKernelSessionOutputRequest()
    request.version_label = 'v1'; request.page_size = 100; request.page_token = ''
    from kaggle.cli import main
    sys.argv = ['kaggle', '--help']
    try: main()
    except SystemExit as exc:
        if exc.code not in (None, 0): raise
assert not kaggle.api._authenticated
assert KaggleApi._kernel_version_label('1') == 'v1'
assert 'file_pattern' in inspect.signature(KaggleApi.kernels_output).parameters
from unittest.mock import patch
import requests
with patch('requests.sessions.get_netrc_auth', side_effect=AssertionError('netrc forbidden')):
    prepared = requests.Session().prepare_request(requests.Request(
        'GET', 'https://storage.googleapis.com/fixture-only', auth=lambda request: request))
assert 'Authorization' not in prepared.headers
print(json.dumps({'sdk_imports': True, 'cli_help': True, 'version_label': request.version_label,
                  'authentication_disabled': True, 'network_disabled': True,
                  'signed_http_without_netrc': True}))
'''


def main():
    missing = []; mismatched = []
    for line in (ROOT/'requirements/artifact-recovery.lock').read_text().splitlines():
        match = re.match(r'^([A-Za-z0-9_.-]+)==([^ ;]+)', line)
        if not match:
            continue
        package, version = match.groups()
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package); continue
        if actual != version:
            mismatched.append(package)
    if missing or mismatched:
        print(json.dumps({'lock_matches': False, 'missing': missing, 'mismatched': mismatched}))
        return 1
    with tempfile.TemporaryDirectory(prefix='trustretina-sdk-sterile-') as folder:
        config = Path(folder)/'config'; config.mkdir(mode=0o700)
        env = {'HOME': folder, 'KAGGLE_CONFIG_DIR': str(config), 'XDG_CONFIG_HOME': folder,
               'PATH': str(Path(sys.executable).parent)+os.pathsep+'/usr/bin:/bin',
               'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', '-c', CHECK], env=env,
                                capture_output=True, text=True, timeout=45)
        if result.returncode:
            print(json.dumps({'lock_matches': True, 'isolated_sdk_check': False}))
            return 1
        print(json.dumps({'lock_matches': True, 'kaggle': importlib.metadata.version('kaggle'),
                          'kagglesdk': importlib.metadata.version('kagglesdk'),
                          'isolated_check': json.loads(result.stdout)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
