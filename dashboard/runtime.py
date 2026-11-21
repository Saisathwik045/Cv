"""Small, non-sensitive CPU startup diagnostics; never loads a checkpoint."""
import importlib
import sys


def cpu_runtime_check():
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        return {'status': 'FAIL', 'code': 'unsupported_python',
                'detail': 'Select Python 3.12 in the hosting settings, save and rebuild. This frozen CPU environment supports Python 3.11/3.12.'}
    for name in ('torch', 'torchvision'):
        try:
            importlib.import_module(name)
        except ImportError:
            return {'status': 'FAIL', 'code': 'missing_cpu_dependency', 'package': name,
                    'detail': 'Install the existing requirements/app-cpu.lock in Python 3.12. On Community Cloud use deploy/community_cloud/app.py as the main file and rebuild.'}
        except (OSError, RuntimeError):
            return {'status': 'FAIL', 'code': 'cpu_dependency_import_failed', 'package': name,
                    'detail': 'The CPU packages could not initialize. Rebuild the Python 3.12 environment using requirements/app-cpu.lock; do not replace the frozen model.'}
    return {'status': 'PASS', 'code': 'cpu_imports_ready',
            'detail': 'CPU imports succeeded; model hashes, checkpoint loading and inference are separate checks.'}
