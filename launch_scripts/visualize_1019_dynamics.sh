#!/usr/bin/env bash
# Current YAML/USD -> RSS MuJoCo -> existing interactive tracker repair page.
set -euo pipefail
RSS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCALETRACK_ROOT="${SCALETRACK_ROOT:-$(dirname -- "$RSS_ROOT")/ScaleTrack}"
DATA_ROOT="${DATA_ROOT:-$SCALETRACK_ROOT/data/1019test}"
RSS_PYTHON="${RSS_PYTHON:-$RSS_ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$(dirname -- "$RSS_ROOT")/supermimic_vibed/logs/refine_eef_audit/ckpt_010000.pt}"
METADATA="${METADATA:-$RSS_ROOT/outputs/humanoid-real2sim/2026-10-24/refined_tracker_1019/runs/ckpt_010000/metadata.json}"
if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: launch_scripts/visualize_1019_dynamics.sh [PACKAGE] [repair options]
       launch_scripts/visualize_1019_dynamics.sh --list
Default package: brace-dining-table-wipes-kinematic-v1
Examples:
  launch_scripts/visualize_1019_dynamics.sh brush-sweep-kinematic-v1 --port 8200
  launch_scripts/visualize_1019_dynamics.sh brace-dining-table-wipes-kinematic-v1 --check-scene
The page starts paused. Ctrl-C closes it and its inference worker.
--check-scene builds/audits without starting inference or a visualization server.
Environment: SCALETRACK_ROOT, DATA_ROOT, RSS_PYTHON, CHECKPOINT, METADATA.
Install USD dependencies once: uv sync --extra usd
Outputs and the effective-property audit are saved in outputs/humanoid-real2sim.
HELP
    exit 0
fi
if [[ "${1:-}" == --list ]]; then
    for manifest in "$DATA_ROOT"/*/motions.yaml; do
        [[ -f "$manifest" ]] && basename -- "$(dirname -- "$manifest")"
    done
    exit 0
fi
PACKAGE=brace-dining-table-wipes-kinematic-v1
if [[ $# -gt 0 && "$1" != --* ]]; then
    PACKAGE="$1"
    shift
fi
if [[ "$PACKAGE" != /* ]]; then PACKAGE="$DATA_ROOT/$PACKAGE"; fi
[[ -f "$PACKAGE/motions.yaml" ]] || { echo "Missing package: $PACKAGE" >&2; exit 2; }
cd -- "$RSS_ROOT"
if ! "$RSS_PYTHON" -c 'import pxr, newton_usd_schemas' 2>/dev/null; then
    echo 'Missing USD importer dependencies. Run: uv sync --extra usd' >&2
    exit 2
fi
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/robosimstudio_humanoid_real2sim_warp}"
RUN_OUT="$("$RSS_PYTHON" -c 'from robosimstudio.humanoid_real2sim.paths import run_dir; print(run_dir("package_dynamics_review"))')"
echo "Run: $RUN_OUT"
exec "$RSS_PYTHON" scripts/humanoid-real2sim/run.py \
    --package "$PACKAGE" --scaletrack-root "$SCALETRACK_ROOT" \
    --checkpoint "$CHECKPOINT" --metadata "$METADATA" \
    --out "$RUN_OUT" --port 8200 "$@"
