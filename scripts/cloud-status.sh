#!/usr/bin/env bash
# cloud-status — reconcile local lease records against each provider's live (cloud-lease-owned)
# instances; surface STRAY/ORPHAN leaks. Add --reap to destroy them. Truth = provider API.
#
# SAFE BY CONSTRUCTION: it only ever lists/reaps instances whose label starts with "cloud-lease-",
# so any UNRELATED nodes on the same account (e.g. a production video-streaming headend) are never
# shown and never reaped.
#
#   scripts/cloud-status.sh            # list
#   scripts/cloud-status.sh --reap     # destroy STRAY/ORPHAN
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m cloud_lease.cli status "$@"
