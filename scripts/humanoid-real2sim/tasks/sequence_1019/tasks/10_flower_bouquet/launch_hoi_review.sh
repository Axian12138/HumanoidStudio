#!/usr/bin/env bash
set -euo pipefail
run="$(realpath "${1:?run directory required}")"
motion="$(realpath "${2:?motion NPZ required}")"
assets="$(realpath "${3:?collision and visual assets directory required}")"
tracks="$(realpath "${4:?tracks manifest required}")"
port="${5:-8140}"
read -r -a camera_eye <<< "${REVIEW_CAMERA_EYE:-1.2 1.2 0.9}"
read -r -a camera_look <<< "${REVIEW_CAMERA_LOOK:--0.3 0.0 0.15}"
export RSS_SEQUENCE_RUN="$run"
export RSS_FLOWER_ASSETS="$assets"
export WARP_CACHE_PATH=/tmp/flower_warp_cache
unset RSS_FLOWER_INITIAL_TRACKS RSS_FLOWER_INITIAL_FRAME RSS_FLOWER_LIVE
exec .venv/bin/python scripts/humanoid-real2sim/tasks/sequence_1019/tasks/10_flower_bouquet/review_clock.py \
  --motion "$motion" \
  --motion-order "$run/../../curated_snapshot/retarget/ground_support_27_end/sidecars/motion_order.json" \
  --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/10_flower_bouquet/scene_registered.py \
  --human-motion "$run/retarget/body_prior/10_flower_bouquet/human_motion_world.npz" \
  --human-reference "$run/retarget/body_prior/10_flower_bouquet/human.npz" \
  --source-sync "$run/retarget/body_prior/10_flower_bouquet/source_sync.npz" \
  --source-offset-s 108 \
  --tracks-manifest "$tracks" \
  --follow-root \
  --camera-eye-offset "${camera_eye[@]}" \
  --camera-look-offset "${camera_look[@]}" \
  --port "$port"
