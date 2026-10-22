#!/usr/bin/env bash
set -euo pipefail

candidate=${1:?usage: play_hoi.sh /path/to/hoi_candidate [port]}
port=${2:-8409}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
task=$(.venv/bin/python -c 'from robosimstudio.humanoid_real2sim.paths import example_dir; print(example_dir("scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/scene.py"))')
base=$(dirname "$(dirname "$task")")
candidate=$(realpath "$candidate")
export RSS_BASKET_VISUAL_MESH="$candidate/basket_visual.obj"

exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
  --motion "$candidate/motion.npz" \
  --motion-order "$base/retarget/ground_support_27_end/sidecars/motion_order.json" \
  --human-motion "$candidate/human_motion_world.npz" \
  --human-reference "$task/human.npz" \
  --source-sync "$task/source_sync.npz" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/scene.py \
  --object-track "$candidate/object_estimate.npz" \
  --object-name lined_rect_basket --source-offset-s 98 --port "$port" --follow-root --dark-mode \
  --camera-eye-offset 1.2 -0.8 0.9 \
  --camera-look-offset 0.35 0.0 0.1
