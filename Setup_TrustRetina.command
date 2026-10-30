#!/bin/bash
# One-time hidden-token recovery of the original pinned private model bundle.
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/scripts/macos_bootstrap.sh" --setup-artifacts "$@"
