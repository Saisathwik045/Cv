#!/usr/bin/env python3
"""Explicitly gated offline operational benchmark; no clinical evaluation."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def private_output(path):
    path = Path(path)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError('Require a fresh absolute output filename')
    if path.resolve().is_relative_to(ROOT):
        raise ValueError('Operational output must stay outside the checkout')
    if any(p.is_symlink() for p in path.parents):
        raise ValueError('Output symlinks are forbidden')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 0o077:
        raise ValueError('Require an owner-only output directory')
    return path


def peak_rss_bytes():
    # Linux ru_maxrss is KiB. This CLI intentionally supports Linux only.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def resource_context():
    result = {'os': platform.system(), 'architecture': platform.machine(),
              'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
              'affinity_cpus': len(os.sched_getaffinity(0))}
    for name, filename in (('cgroup_cpu_max', 'cpu.max'), ('cgroup_memory_max', 'memory.max')):
        p = Path('/sys/fs/cgroup') / filename
        result[name] = p.read_text().strip() if p.is_file() else 'unavailable'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-synthetic-cpu', action='store_true', help='Explicitly authorize the fixed four-prediction local benchmark')
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True, help='Fresh JSON outside the checkout in an owner-only directory')
    args = parser.parse_args()
    if not args.run_synthetic_cpu:
        parser.error('No execution without --run-synthetic-cpu')
    if platform.system() != 'Linux':
        parser.error('This measurement harness supports Linux only; no native Mac claims')
    os.umask(0o077)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    def deny_network(*a, **kw):
        raise RuntimeError('Benchmark networking is disabled')
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    def deadline(*a):
        raise TimeoutError('Benchmark deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    try:
        output = private_output(args.output)
        from trustretina.inference.bundle import verify_bundle, required_files, CONTRACT_SHA256
        started = time.perf_counter()
        contract = verify_bundle(args.bundle)
        verification_seconds = time.perf_counter() - started
        started = time.perf_counter()
        import torch
        from trustretina.inference.service import InferenceService
        from trustretina.runner.reliability_runtime import state_hash
        import_seconds = time.perf_counter() - started
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from dashboard.performance import measure
        baseline_peak = peak_rss_bytes()
        started = time.perf_counter()
        measured = measure(lambda: InferenceService(args.bundle, device='cpu'), contract, state_hash)
        execution_wall = time.perf_counter() - started
        after_peak = peak_rss_bytes()
        verify_bundle(args.bundle)
        report = {'schema': 'synthetic_cpu_performance_v1', 'status': 'PASS',
                  'input_kind': 'generated_geometric_non_patient', 'research_evaluation': False,
                  'clinical_validation': False, 'decision': 'REVIEW REQUIRED',
                  'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'environment': resource_context(), 'torch_threads': torch.get_num_threads(),
                  'torch_interop_threads': torch.get_num_interop_threads(),
                  'dependencies': {name: importlib.metadata.version(name) for name in ('torch','torchvision','numpy','Pillow')},
                  'initial_five_file_verification_seconds': verification_seconds,
                  'dependency_import_seconds': import_seconds, 'measurement_wall_seconds': execution_wall,
                  'memory': {'process_peak_rss_before_model_bytes': baseline_peak,
                             'process_peak_rss_after_measurement_bytes': after_peak,
                             'interpretation': 'Process lifetime high-water RSS including imports, verification, temporary copies and Grad-CAM; not isolated model size or current RSS'},
                  'measurements': measured,
                  'provenance': {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                 'bundle_hashes': required_files(contract), 'contract_sha256': CONTRACT_SHA256,
                                 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'measurement_module_sha256': hashlib.sha256((ROOT/'dashboard/performance.py').read_bytes()).hexdigest()},
                  'limits': ['One process, one generated pair, no clinical interpretation',
                             'First prediction is not a cold OS-cache measurement; no cache flushing',
                             'Three repeated samples are not a p95, confidence interval or throughput benchmark',
                             'Includes decode/preprocessing, 20 head-only MC passes, quality and frozen temperature in predict',
                             'No browser, upload, download, concurrent-user, native Mac or independent clinical validation',
                             'No predictions, images or explanation arrays retained; network disabled; 180-second deadline']}
        with output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
        output.chmod(0o600)
        print('PASS: aggregate-only CPU timings saved privately; four predictions and one explanation.')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, TimeoutError):
        print('FAIL: benchmark stopped; verify approved assets, private output permissions and CPU environment. No fallback model or reported success.', file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
