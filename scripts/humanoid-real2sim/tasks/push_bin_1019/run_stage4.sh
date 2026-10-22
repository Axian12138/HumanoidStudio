#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
WORKSPACE=$(dirname "$ROOT")
TASK="$ROOT/outputs/humanoid-real2sim/2026-10-20/push_bin_1019/curated_snapshot"
RUN_DIR=$("$ROOT/.venv/bin/python" -c "from robosimstudio.humanoid_real2sim.paths import run_dir; print(run_dir('push_bin_1019'))")

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/humanoid-real2sim/run.py" \
  --motion "$TASK/retarget/hoi_candidate_scene.npz" \
  --motion-order "$WORKSPACE/ScaleTrack/retarget3/assets/robot/motion_order.json" \
  --metadata "$WORKSPACE/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt_metadata.json" \
  --checkpoint "$WORKSPACE/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt.pt" \
  --scalebridge-python "$WORKSPACE/ScaleBridge/.venv/bin/python" \
  --scene-file "$ROOT/scripts/humanoid-real2sim/tasks/push_bin_1019/scene_mu_0p10_com_0p03_contactfix.py" \
  --show-reference \
  --start-frame 0 --stop-frame 474 \
  --port 8180 \
  --out "$RUN_DIR/stage4_deployed_tracker_editor" \
  "$@"
