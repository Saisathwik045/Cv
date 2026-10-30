#!/bin/bash
# Double-click on an Apple-silicon Mac after one-time bundle setup.
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/scripts/macos_bootstrap.sh" "$@"
