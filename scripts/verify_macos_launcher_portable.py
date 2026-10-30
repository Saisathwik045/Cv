#!/usr/bin/env python3
"""Portable real Streamlit lifecycle fixture; no macOS/native/model evaluation claim."""
import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import macos_launcher as launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    launcher.verify_bundle(ROOT / 'artifacts/inference-v8')
    with tempfile.TemporaryDirectory(prefix='Trust Retina portable ', dir=ROOT / 'artifacts/demo-v8') as tmp:
        root = Path(tmp).resolve()
        (root / 'streamlit_app.py').write_text("import streamlit as st\nst.title('Portable lifecycle fixture')\nst.file_uploader('Left eye')\nst.file_uploader('Right eye')\n")
        (root / 'requirements').mkdir()
        (root / launcher.APP_LOCK).write_bytes((ROOT / launcher.APP_LOCK).read_bytes())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        worker = root / 'worker.py'
        worker.write_text(f"import sys\nfrom pathlib import Path\nsys.path.insert(0,{str(ROOT / 'scripts')!r})\nimport macos_launcher as m\nm.verify_bundle=lambda folder: None\nm.prepare_environment=lambda root: Path({str(args.python)!r})\ntry:\n print(m.start(Path({str(root)!r}),port={port},open_browser=False),flush=True)\nexcept KeyboardInterrupt:pass\n")
        process = subprocess.Popen([str(args.python), str(worker)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and not launcher.health_ok(port):
                if process.poll() is not None:
                    raise RuntimeError('Fixture owner exited before health')
                time.sleep(.25)
            assert launcher.health_ok(port)
            record = json.loads((root / '.trustretina/server.json').read_text())
            assert launcher.owned_process(record, root)
            second = subprocess.run([str(args.python), str(worker)], capture_output=True, text=True, timeout=15)
            assert second.returncode == 0 and 'reused' in second.stdout
            assert json.loads((root / '.trustretina/server.json').read_text())['pid'] == record['pid']
            process.send_signal(signal.SIGINT)
            process.wait(timeout=20)
            assert not (root / '.trustretina/server.json').exists()
            assert not launcher.port_busy(port)
            events = [json.loads(x)['event'] for x in (root / '.trustretina/launcher.jsonl').read_text().splitlines()]
            assert events.count('server_started') == 1 and events.count('server_reused') == 1 and events.count('server_stopped') == 1
            report = {'scope': 'Portable real Streamlit fixture on current host, not macOS testing', 'system': launcher.platform.system(), 'macos_native_execution': False, 'macos_dependency_installation': False, 'finder_gatekeeper_browser_open': 'unverified on macOS', 'path_with_spaces': True, 'loopback_http_health': True, 'process_identity': True, 'second_launch_reused_same_pid': True, 'one_server_only': True, 'control_c_stops_owned_server_and_cleans_receipt': True, 'raw_server_output_persisted': False, 'model_inference_or_evaluation': False, 'frozen_bundle_hash_verification': True}
            output = ROOT / 'results/demo-v8/macos-portable-verification.json'
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=20)


if __name__ == '__main__':
    main()
