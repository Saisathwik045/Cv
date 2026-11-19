#!/bin/bash
# Run locally on a supported Mac. Recover only the original frozen bundle, then start.
set -eu
umask 077
export TRUSTRETINA_BUNDLE="${TRUSTRETINA_BUNDLE:-$HOME/Library/Application Support/TrustRetina/inference-v8}"
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/scripts/macos_bootstrap.sh" --setup-and-start "$@"
