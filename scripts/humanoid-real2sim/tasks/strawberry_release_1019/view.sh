#!/bin/bash
set -eu
cd /home/galbot/workspace/robosimstudio
candidate=$1
export RSS_SEQUENCE_RUN=/home/galbot/workspace/robosimstudio/outputs/humanoid-real2sim/2026-10-20/sequence_1019/runs/21-08-52.042738+0800
export RSS_08_GEOMETRY_CONFIG="$candidate/geometry_config.json"
export WARP_CACHE_PATH=/tmp/warp_drawer_1019
exec .venv/bin/python scripts/humanoid-real2sim/kinematic_viewer.py --motion "$candidate/motion.npz" --motion-order "$candidate/motion_order.json" --human-motion "$RSS_SEQUENCE_RUN/analysis/08_table_strawberry/human_motion_world.npz" --human-reference "$candidate/human.npz" --source-sync "$candidate/source_sync.npz" --scene-file scripts/humanoid-real2sim/tasks/sequence_1019/tasks/08_table_strawberry/scene.py --tracks-manifest "$candidate/tracks.json" --source-offset-s 90 --port 8100 --dark-mode --follow-root --camera-eye-offset -0.9 -0.9 0.9 --camera-look-offset 0 -0.1 0.2
