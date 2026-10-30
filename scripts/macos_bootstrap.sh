#!/bin/bash
# Compatible with Apple's Bash 3.2. Never echo credentials or the environment.
set -u
trustretina_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$trustretina_root" || exit 1
umask 077
trustretina_python=""
for trustretina_candidate in "${TRUSTRETINA_PYTHON:-}" "$trustretina_root/.venv-mac/bin/python" "$trustretina_root/.venv-artifacts-mac/bin/python" /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/opt/python@3.11/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 "$(command -v python3.12 || true)" "$(command -v python3.11 || true)" "$(command -v python3 || true)"; do
    if [ -n "$trustretina_candidate" ] && [ -x "$trustretina_candidate" ] && "$trustretina_candidate" -c 'import platform,sys;sys.exit(0 if platform.system()=="Darwin" and platform.machine()=="arm64" and sys.version_info[:2] in ((3,11),(3,12)) else 1)' >/dev/null 2>&1; then
        trustretina_python="$trustretina_candidate"
        break
    fi
done
if [ -z "$trustretina_python" ]; then
    echo 'TrustRetina needs native Apple-silicon Python 3.11 or 3.12 and macOS 13+.'
    echo 'Install the Python 3.12 macOS universal2 installer from https://www.python.org/downloads/macos/ and retry.'
    echo 'Intel/Rosetta and newer Python versions are not supported by this locked launcher.'
    trustretina_code=1
else
    "$trustretina_python" "$trustretina_root/scripts/macos_launcher.py" "$@"
    trustretina_code=$?
fi
if [ -t 0 ]; then
    printf '\nPress Return to close this Terminal window. '
    read -r trustretina_unused || true
fi
exit "$trustretina_code"
