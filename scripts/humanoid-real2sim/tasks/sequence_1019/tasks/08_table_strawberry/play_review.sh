#!/usr/bin/env bash
set -euo pipefail
candidate=$(realpath "${1:?candidate directory required}")
port=${2:-8104}
motion=${3:-motion.npz}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
run=$(dirname "$(dirname "$(dirname "$(dirname "$candidate")")")")
if [[ "$candidate" == */final/* ]]; then run=$(dirname "$(dirname "$candidate")"); fi
export RSS_SEQUENCE_RUN="$run"
export RSS_08_GEOMETRY_CONFIG="$candidate/geometry_config.json"
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion "$candidate/$motion" --motion-order "$candidate/motion_order.json" \
  --human-motion "$run/analysis/08_table_strawberry/human_motion_world.npz" \
  --human-reference "$candidate/human.npz" --source-sync "$candidate/source_sync.npz" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/08_table_strawberry/scene.py \
  --tracks-manifest "$candidate/tracks.json" --source-offset-s 90 --port "$port" \
  --dark-mode --camera-eye-offset -2.4 -1.2 1.5 --camera-look-offset 0 -0.3 -0.15
