#!/usr/bin/env bash
# Ship a committed tree to a GPU machine as an immutable checkout and print how to run it.
#   scripts/sync_checkout.sh <herdr-machine-label> [<git-ref>]   (default ref: HEAD)
# The tree lands in $DAZZLE_WORKSPACE/checkouts/<sha12> on the machine; uncommitted changes are
# not shipped, so every run is attributable to a commit (docs/agents/compute.md).
set -euo pipefail
label=${1:?usage: sync_checkout.sh <herdr-machine-label> [<git-ref>]}
ref=${2:-HEAD}
sha=$(git rev-parse --verify "$ref^{commit}")
short=${sha:0:12}
target=$(herdr machine list --json | python3 -c "import json,sys; m=[x['target'] for x in json.load(sys.stdin) if x['label']==sys.argv[1]]; print(m[0] if m else '')" "$label")
[ -n "$target" ] || { echo "no herdr machine labelled '$label' (herdr machine list)" >&2; exit 1; }
ws=${DAZZLE_REMOTE_WORKSPACE:-dazzle-v2}
dest="$ws/checkouts/$short"
git archive --format=tar "$sha" | ssh "$target" "mkdir -p '$dest' && tar -x -C '$dest'"
echo "$label:~/$dest  (commit $sha)"
echo "run: . ~/$ws/env.sh && cd ~/$dest && export DAZZLE_GIT_SHA=$sha PYTHONPATH=src && python -m dazzle.evaluation.run --help"
