#!/bin/bash
set -eu
cd /home/galbot/workspace/robosimstudio
run=$1
version=${2:-v6}
export WARP_CACHE_PATH=/tmp/warp_drawer_1019
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
 --motion "$run/retarget/$version/motion.npz" \
 --motion-order "$run/retarget/$version/motion_order.json" \
 --human-motion "$run/retarget/$version/human_motion_world.npz" \
 --human-reference "$run/retarget/$version/human.npz" \
 --source-sync "$run/retarget/$version/source_sync.npz" --source-offset-s 83.5 \
 --object-track "$run/retarget/$version/chair_track.npz" --object-name grey_armchair \
 --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/07_armchair/scene.py \
 --port 8100 --follow-root --camera-eye-offset -1.0 0.5 0.7 --camera-look-offset 0.05 -0.3 0.12
