#!/usr/bin/env bash
set -euo pipefail

mode=${1:-thumb}
case "$mode" in
  thumb) candidate=hoi_v3_thumb_contact; port=8380 ;;
  palm) candidate=hoi_v3_palm_push; port=8390 ;;
  *) echo 'usage: play_hand_variant.sh [thumb|palm]' >&2; exit 2 ;;
esac

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
base=outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot
task=$base/tasks/07_armchair

exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion "$task/$candidate/motion.npz" \
  --motion-order "$base/retarget/ground_support_27_end/sidecars/motion_order.json" \
  --human-motion "$task/$candidate/human_motion_world.npz" \
  --human-reference "$task/human.npz" \
  --source-sync "$task/source_sync.npz" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py \
  --object-track "$task/$candidate/object_estimate.npz" \
  --object-name grey_armchair --source-offset-s 75 --port "$port" \
  --camera-eye-offset -1.49 0.87 0.68 \
  --camera-look-offset -1.08 0.17 0.33
