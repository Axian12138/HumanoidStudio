#!/bin/bash
set -eu
cd /home/galbot/workspace/robosimstudio
run=$1
version=${2:-v2}
export RSS_SEQUENCE_SCENE_CONFIG="$run/scene/config.json"
export WARP_CACHE_PATH=/tmp/warp_drawer_1019
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py \
 --motion "$run/retarget/$version/motion.npz" \
 --motion-order "$run/retarget/$version/motion_order.json" \
 --human-motion "$run/retarget/$version/human_motion_world.npz" \
 --source-sync "$run/retarget/$version/source_sync.npz" \
 --tracks-manifest "$run/retarget/$version/scene_tracks/tracks.json" \
 --source-offset-s 49.5 --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/unified_scene.py \
 --port 8100 --follow-root --camera-eye-offset -1.1 0.8 0.55 --camera-look-offset -0.22 -0.28 0.1
