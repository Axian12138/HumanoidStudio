#!/usr/bin/env bash
set -euo pipefail

mode=${1:-thumb}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
run_root=outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/runs
run=${2:-$(find "$run_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)}
base=outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot

case "$mode" in
  thumb)
    clip=chair_thumb; offset=75; port=8480
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_edge)
    clip=chair_thumb_edge; offset=75; port=8510
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_surface)
    clip=chair_thumb_surface; offset=75; port=8530
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_strip)
    clip=chair_thumb_strip; offset=75; port=8550
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_backface)
    clip=chair_thumb_backface; offset=75; port=8560
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_backface_flip)
    clip=chair_thumb_backface_flip; offset=75; port=8570
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  thumb_backface_flip_open)
    clip=chair_thumb_backface_flip_open; offset=75; port=8580
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  palm)
    clip=chair_palm; offset=75; port=8490
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  palm_edge)
    clip=chair_palm_edge; offset=75; port=8520
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  palm_surface)
    clip=chair_palm_surface; offset=75; port=8540
    eye=(-1.49 0.87 0.68); look=(-1.08 0.17 0.33) ;;
  table)
    clip=table_support; offset=83.5; port=8500
    eye=(-0.8 0.6 0.45); look=(-0.2 -0.8 -0.1) ;;
  *) echo 'usage: play_split_feedback.sh [thumb|thumb_edge|thumb_surface|thumb_strip|thumb_backface|thumb_backface_flip|thumb_backface_flip_open|palm|palm_edge|palm_surface|table] [run-directory]' >&2; exit 2 ;;
esac

test -f "$run/$clip/motion.npz"
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion "$run/$clip/motion.npz" \
  --motion-order "$base/retarget/ground_support_27_end/sidecars/motion_order.json" \
  --human-motion "$run/$clip/human_motion_world.npz" \
  --human-reference "$run/$clip/human.npz" \
  --source-sync "$run/$clip/source_sync.npz" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py \
  --object-track "$run/$clip/object_estimate.npz" \
  --object-name grey_armchair --source-offset-s "$offset" --port "$port" \
  --camera-eye-offset "${eye[@]}" --camera-look-offset "${look[@]}"
