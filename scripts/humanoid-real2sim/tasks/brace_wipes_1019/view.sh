#!/bin/bash
set -eu
cd /home/galbot/workspace/robosimstudio
export RSS_WIPES_RUN=$1
version=${2:-v1}
export WARP_CACHE_PATH=/tmp/warp_drawer_1019
p="$RSS_WIPES_RUN/retarget/$version"
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py --motion "$p/motion.npz" --motion-order "$p/motion_order.json" --human-motion "$p/human_motion_world.npz" --human-reference "$p/human.npz" --source-sync "$p/source_sync.npz" --tracks-manifest "$p/tracks.json" --source-offset-s 83.5 --scene-file scripts/humanoid-real2sim/tasks/brace_wipes_1019/scene.py --port 8100 --follow-root --camera-eye-offset 1.0 0.3 1.0 --camera-look-offset .1 -.3 .05
