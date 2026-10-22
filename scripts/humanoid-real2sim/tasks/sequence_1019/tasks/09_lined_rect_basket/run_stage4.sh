#!/usr/bin/env bash
set -euo pipefail

candidate=${1:?usage: run_stage4.sh /path/to/hoi_candidate [port]}
port=${2:-8530}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
candidate=$(realpath "$candidate")
workspace=$(dirname "$repo")
run=$(.venv/bin/python -c 'from robosimstudio.humanoid_real2sim.paths import run_dir; print(run_dir("sequence_1019/09_lined_rect_basket"))')
export RSS_BASKET_STAGE4_CANDIDATE="$candidate"

exec .venv/bin/python scripts/humanoid-real2sim/run.py \
  --motion "$candidate/motion.npz" \
  --motion-order "$workspace/ScaleTrack/retarget3/assets/robot/motion_order.json" \
  --metadata "$workspace/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt_metadata.json" \
  --checkpoint "$workspace/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt.pt" \
  --scalebridge-python "$workspace/ScaleBridge/.venv/bin/python" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/scene_stage4.py \
  --reference-object-track "$candidate/object_estimate.npz" \
  --reference-object-mesh "$candidate/basket_visual.obj" \
  --reference-object-outline box \
  --start-frame 0 --stop-frame 550 \
  --port "$port" --out "$run/stage4_$(basename "$candidate")" \
  --show-reference
