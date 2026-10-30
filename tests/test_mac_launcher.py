"""Portable lifecycle tests. They do not claim native macOS execution."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('macos_launcher', ROOT / 'scripts/macos_launcher.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class MacLauncherTests(unittest.TestCase):
    def test_platform_checks_native_supported_python_and_os(self):
        with patch.object(launcher.platform, 'system', return_value='Darwin'), patch.object(launcher.platform, 'machine', return_value='arm64'), patch.object(launcher.platform, 'mac_ver', return_value=('13.0', (), '')), patch.object(launcher.sys, 'version_info', (3, 12, 12)):
            launcher.require_mac()
            with patch.object(launcher.platform, 'machine', return_value='x86_64'):
                with self.assertRaisesRegex(launcher.LaunchError, 'Rosetta'):
                    launcher.require_mac()
            with patch.object(launcher.platform, 'mac_ver', return_value=('12.0', (), '')):
                with self.assertRaisesRegex(launcher.LaunchError, '13'):
                    launcher.require_mac()
            with patch.object(launcher.sys, 'version_info', (3, 13, 0)):
                with self.assertRaisesRegex(launcher.LaunchError, '3.12'):
                    launcher.require_mac()

    def test_locks_keep_selected_runtime_versions_without_linux_build_suffix(self):
        mac = launcher.read_pins(ROOT / launcher.APP_LOCK)
        linux = dict(line.split('==') for line in (ROOT / 'requirements/app-cpu.lock').read_text().splitlines() if line and not line.startswith(('#', '--')))
        self.assertEqual(mac['torch'], '2.7.1')
        self.assertEqual(mac['torchvision'], '0.22.1')
        self.assertNotIn('kaggle', mac)
        for name, version in mac.items():
            self.assertEqual(version, linux[name].removesuffix('+cpu'))
        self.assertEqual(launcher.read_pins(ROOT / launcher.RECOVERY_LOCK)['kaggle'], '2.2.5')

    def test_lock_rejects_unpinned_or_shell_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            for text in ('torch>=2', '--index-url https://secret.example', 'x==1; malicious'):
                path.write_text(text)
                with self.assertRaises(launcher.LaunchError):
                    launcher.read_pins(path)

    def test_runtime_drops_kaggle_credentials_and_preserves_parent(self):
        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'test-secret', 'KAGGLE_CONFIG_DIR': 'private', 'PATH': '/safe'}, clear=True):
            env = launcher.runtime_environment()
            self.assertNotIn('KAGGLE_API_TOKEN', env)
            self.assertNotIn('KAGGLE_CONFIG_DIR', env)
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'test-secret')
            self.assertEqual(env['PATH'], '/safe')

    def test_log_is_private_and_rejects_arbitrary_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher.log_event(tmp, 'server_healthy', pid=123, port=8501)
            path = Path(tmp) / '.trustretina/launcher.jsonl'
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(json.loads(path.read_text())['event'], 'server_healthy')
            with self.assertRaises(ValueError):
                launcher.log_event(tmp, 'bad', token='test-secret')
            self.assertNotIn('test-secret', path.read_text())

    def test_concurrent_launch_lock_prevents_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with launcher.project_lock(tmp):
                with self.assertRaisesRegex(launcher.LaunchError, 'duplicate'):
                    with launcher.project_lock(tmp):
                        self.fail('Second launcher acquired lock')

    def test_health_response_alone_cannot_identify_existing_server(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'port_busy', return_value=True):
            with self.assertRaisesRegex(launcher.LaunchError, 'unverified/manual'):
                launcher.existing_server(tmp, 8501)

    def test_verified_existing_server_reused_without_install_or_spawn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'verify_bundle'), patch.object(launcher, 'existing_server', return_value={'pid': 123}), patch.object(launcher, 'prepare_environment') as prepare, patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher, 'browser_open') as browser:
            self.assertEqual(launcher.start(tmp), 'reused')
            prepare.assert_not_called()
            spawn.assert_not_called()
            browser.assert_called_once_with(8501)

    def test_missing_bundle_fails_before_install_or_server(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'prepare_environment') as prepare, patch.object(launcher.subprocess, 'Popen') as spawn, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(launcher.LaunchError, 'Setup_TrustRetina'):
                launcher.start(tmp, open_browser=False)
            prepare.assert_not_called()
            spawn.assert_not_called()

    def test_process_identity_rejects_other_project(self):
        record = {'pid': 123, 'root': '/other', 'python': '/other/python', 'port': 8501}
        self.assertFalse(launcher.owned_process(record, ROOT))
        record['root'] = str(ROOT)
        with patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0, stdout='unrelated http server')):
            self.assertFalse(launcher.owned_process(record, ROOT))

    def test_environment_installs_once_then_reuses_and_repairs_changed_lock(self):
        with tempfile.TemporaryDirectory(prefix='Trust Retina ') as tmp:
            root = Path(tmp)
            (root / 'requirements').mkdir()
            lock = root / launcher.APP_LOCK
            lock.write_text('sample-package==1.0\n')
            def create(folder):
                (folder / 'bin').mkdir(parents=True)
                (folder / 'bin/python').touch()
            identity = json.dumps([list(sys.version_info[:2]), launcher.platform.machine()])
            with patch.object(launcher.venv.EnvBuilder, 'create', side_effect=create), patch.object(launcher, 'run_checked') as run, patch.object(launcher, 'environment_matches', return_value=True), patch.object(launcher.subprocess, 'run', return_value=Mock(stdout=identity)):
                python = launcher.prepare_environment(root)
                self.assertTrue(python.exists())
                self.assertEqual(run.call_count, 3)
                launcher.prepare_environment(root)
                self.assertEqual(run.call_count, 3)
                lock.write_text('sample-package==2.0\n')
                launcher.prepare_environment(root)
                self.assertEqual(run.call_count, 6)

    def test_unhealthy_existing_managed_process_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = launcher.secure_folder(Path(tmp) / '.trustretina')
            (folder / 'server.json').write_text(json.dumps({'pid': 123, 'port': 8501, 'lock_sha256': 'fixture'}))
            with patch.object(launcher, 'owned_process', return_value=True), patch.object(launcher, 'release_identity', return_value={'lock_sha256': 'fixture'}), patch.object(launcher, 'health_ok', return_value=False):
                with self.assertRaisesRegex(launcher.LaunchError, 'no duplicate'):
                    launcher.existing_server(tmp, 8501)

    def test_failed_startup_never_opens_browser(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'verify_bundle'), patch.object(launcher, 'existing_server', return_value=None), patch.object(launcher, 'prepare_environment', return_value=Path(sys.executable)), patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher, 'release_identity', return_value={'lock_sha256':'fixture'}), patch.object(launcher, 'wait_for_health', side_effect=launcher.LaunchError('failed')), patch.object(launcher, 'browser_open') as browser, patch.object(launcher, 'stop_child') as stop:
            spawn.return_value.pid = 123
            with self.assertRaises(launcher.LaunchError):
                launcher.start(tmp)
            browser.assert_not_called()
            stop.assert_called_once_with(spawn.return_value)
            self.assertFalse((Path(tmp) / '.trustretina/server.json').exists())

    def test_credential_file_permissions_are_checked_without_reading_contents(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': tmp}):
            path = Path(tmp) / 'access_token'
            path.write_text('test-secret')
            path.chmod(0o644)
            with self.assertRaises(launcher.LaunchError):
                launcher.check_credential_permissions()
            path.chmod(0o600)
            launcher.check_credential_permissions()

    def test_verified_bundle_setup_does_not_prompt_or_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'artifacts/inference-v8'
            bundle.mkdir(parents=True)
            with patch.object(launcher, 'verify_bundle'), patch.object(launcher, 'prepare_environment') as prepare, patch.object(launcher.getpass, 'getpass') as prompt, patch.dict(os.environ, {}, clear=True):
                launcher.setup_artifacts(tmp)
                prepare.assert_not_called()
                prompt.assert_not_called()

    def test_hidden_token_is_child_environment_only_not_arguments_or_logs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch.object(launcher, 'prepare_environment', return_value=Path(sys.executable)), patch.object(launcher.sys.stdin, 'isatty', return_value=True), patch.object(launcher.getpass, 'getpass', return_value='test-private-token'), patch.object(launcher, 'verify_bundle'):
            def run(root, args, phase, *, env):
                self.assertEqual(env['KAGGLE_API_TOKEN'], 'test-private-token')
                self.assertNotIn('test-private-token', ' '.join(args))
            with patch.object(launcher, 'run_checked', side_effect=run):
                launcher.setup_artifacts(tmp)
            logs = (Path(tmp) / '.trustretina/launcher.jsonl').read_text()
            self.assertNotIn('test-private-token', logs)
            self.assertNotIn('KAGGLE_API_TOKEN', os.environ)

    def test_health_must_succeed_before_browser_opens(self):
        order = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'verify_bundle'), patch.object(launcher, 'existing_server', return_value=None), patch.object(launcher, 'prepare_environment', return_value=Path(sys.executable)), patch.object(launcher.subprocess, 'Popen') as spawn, patch.object(launcher, 'release_identity', return_value={'lock_sha256':'fixture'}), patch.object(launcher, 'wait_for_health', side_effect=lambda child, port: order.append('health')), patch.object(launcher, 'browser_open', side_effect=lambda port: order.append('browser')), patch.object(launcher, 'stop_child'):
            spawn.return_value.pid = 123
            spawn.return_value.wait.return_value = 0
            launcher.start(tmp)
        self.assertEqual(order, ['health', 'browser'])

    def test_changed_release_cannot_reuse_even_healthy_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = launcher.secure_folder(Path(tmp) / '.trustretina')
            (folder / 'server.json').write_text(json.dumps({'pid':123, 'port':8501, 'lock_sha256':'old'}))
            with patch.object(launcher, 'owned_process', return_value=True), patch.object(launcher, 'release_identity', return_value={'lock_sha256':'new'}), patch.object(launcher, 'health_ok', return_value=True):
                with self.assertRaisesRegex(launcher.LaunchError, 'changed'):
                    launcher.existing_server(tmp, 8501)

    def test_relative_bundle_path_is_relative_to_checkout(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TRUSTRETINA_BUNDLE':'private-bundle'}):
            self.assertEqual(launcher.bundle_location(tmp), (Path(tmp) / 'private-bundle').resolve())

    def test_real_child_output_is_not_logged_and_failure_keeps_stage_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher.run_checked(tmp, [sys.executable, '-c', "print('test-secret-upload-bytes')"], 'Portable subprocess check')
            with self.assertRaisesRegex(launcher.LaunchError, 'exit 3'):
                launcher.run_checked(tmp, [sys.executable, '-c', "import sys;print('test-secret-token');sys.exit(3)"], 'Portable failure check')
            text = (Path(tmp) / '.trustretina/launcher.jsonl').read_text()
            self.assertNotIn('test-secret', text)
            records = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(records[-1]['phase'], 'Portable failure check')
            self.assertEqual(records[-1]['code'], 3)

    def test_framework_python_argv0_does_not_prevent_process_identification(self):
        record = {'pid': 123, 'root': str(ROOT), 'python': '/project/.venv-mac/bin/python', 'port': 8501}
        command = f'/Library/Frameworks/Python.framework/Python.app/Python -m streamlit run {ROOT / "streamlit_app.py"} --server.address 127.0.0.1 --server.port 8501 --server.headless true'
        with patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0, stdout=command)):
            self.assertTrue(launcher.owned_process(record, ROOT))
