#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: run_stage4.sh index|thumb [run-directory] [port]}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
cd "$repo"
workspace=$(dirname "$repo")
run_root=outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_stage4/runs
run=${2:-$(find "$run_root" -mindepth 1 -maxdepth 1 -type d -name '*+0800' | sort | tail -1)}
test -n "$run"
run=$(realpath "$run")
assets="$run/assets_v3"
test -f "$assets/collision_manifest.json"

case "$variant" in
  index)
    candidate="$repo/outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_split_feedback/runs/17-54-03.563689+0800/chair_palm_surface"
    port=${3:-8600} ;;
  thumb)
    candidate="$repo/outputs/humanoid-real2sim/2026-10-20/sequence_1019__armchair_finger_entry/runs/19-20-24.732142+0800/chair_thumb_backface_flip_open"
    port=${3:-8640} ;;
  *) echo 'variant must be index or thumb' >&2; exit 2 ;;
esac
test -f "$candidate/motion.npz"
test -f "$candidate/object_estimate.npz"
export RSS_ARMCHAIR_STAGE4_CANDIDATE="$candidate"
export RSS_ARMCHAIR_STAGE4_ASSETS="$assets"

exec .venv/bin/python scripts/humanoid-real2sim/run.py \
  --motion "$candidate/motion.npz" \
  --motion-order "$repo/outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json" \
  --metadata "$workspace/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt_metadata.json" \
  --checkpoint "$workspace/ScaleBridge/scalebridge/data/model/bfm_transformer_dataset_L_8192_64_64/model_22200_tensorrt.pt" \
  --scalebridge-python "$workspace/ScaleBridge/.venv/bin/python" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene_stage4.py \
  --reference-object-track "$candidate/object_estimate.npz" \
  --reference-object-mesh "$repo/outputs/humanoid-real2sim/2026-10-20/sequence_1019/curated_snapshot/tasks/07_armchair/grey_armchair_visual.obj" \
  --reference-object-outline box \
  --start-frame 0 --stop-frame 475 \
  --hand-open-frame 194 --hand-close-frame 300 \
  --port "$port" --out "$run/${variant}_stage4" \
  --show-reference
