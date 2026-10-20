# RSS kick-basket stage 4 diagnosis

Input: `/home/galbot/workspace/ScaleTrack/data/1012test/kick-basket-v1/motion.npz`.
The 150-frame source is left unchanged. The source contact plan places the
left foot against `round_laundry_basket` at frames 67–105.

The tracker is the exact raw ScaleTrack checkpoint named by
`ScaleTrack/logs/debug/20261016/late_clips/play_tracking_only.sh`:
`bfm_transformer_dataset_L_8192_32_48/model_22200.pt`. The worker loads its
actor and task embedder with pure Torch and runs MuJoCo/RSS physics; it does
not launch Isaac Sim. `tracker_metadata.json` records the ordered 29 body
joints, nominal joint pose, action scales, gains, and source checkpoint.

`scene.py` is a local contact scene using the G1 Dex3 MJCF, the void-cell floor,
and the basket mesh from `round_laundry_basket_v2.blend`. `export_assets.py`
exports the 1035 visible meshes and the 65 Blender `Hull` collision meshes
with Z up. Each hull is a separate MuJoCo collision shape; importing their
combined OBJ as one shape incorrectly seals the basket cavity. The hull count
matches the supplied USD collision count. The room
USD is not loaded, and basket mass 0.8 kg and friction 0.8 are provisional;
this scene is not a calibrated living-room twin.

To reproduce the unedited RSS/ScaleBridge BFM baseline from the RSS root:

```bash
WARP_CACHE_PATH=/tmp/robosimstudio_kick_basket_warp \
  .venv/bin/python scripts/humanoid-real2sim/tasks/kick_basket/baseline.py \
  --motion /home/galbot/workspace/ScaleTrack/data/1012test/kick-basket-v1/motion.npz \
  --out outputs/humanoid-real2sim/kick_basket/correct_tracker_65_hulls_gpu
```

This needs a CUDA-visible GPU. The baseline writes actual RSS joint/body state,
tracker actions, basket pose, source-frame mapping, and input hashes. Run
`analyze.py --motion ... --rollout ... --motion-order
/home/galbot/workspace/ScaleTrack/retarget3/assets/robot/motion_order.json`
for the numeric comparison.

The corrected 150-frame GPU MuJoCo/RSS baseline remains upright: mean pelvis
position error is 0.034 m. The left ankle's mean position error in the
contact window is 0.139 m, and the basket moves 0.014 m versus 0.212 m in the
reference. The supplied Isaac Sim playback rollout with this same checkpoint
also remains upright and has 0.128 m left-ankle contact error and 0.019 m
basket displacement. RSS versus Isaac mean body-position differences are
0.008 m at the pelvis and 0.021 m at the left ankle. The remaining kick error
is therefore present in the reference tracker playback too. It exceeds the
stage-4 editor's 5 cm local edit budget; no candidate motion was exported. See
`outputs/humanoid-real2sim/kick_basket/correct_tracker_65_hulls_gpu/analysis.json`.
One bounded trial took over the left ankle at source frame 60, commanded
movement toward the basket for 15 physics frames, and attempted recovery.
The basket moved only 0.003 m; recovery was still unconverged after 260
simulation frames. See `outputs/humanoid-real2sim/kick_basket/probe_left_minus_x/probe.json`.

The interactive editor for this exact checkpoint and local scene is launched
with:

```bash
.venv/bin/python scripts/humanoid-real2sim/run.py \
  --motion /home/galbot/workspace/ScaleTrack/data/1012test/kick-basket-v1/motion.npz \
  --metadata examples/humanoid-real2sim/kick_basket/tracker_metadata.json \
  --checkpoint /home/galbot/workspace/ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/bfm_transformer_dataset_L_8192_32_48/model_22200.pt \
  --scene-file scripts/humanoid-real2sim/tasks/kick_basket/scene.py \
  --port 8091 --out outputs/humanoid-real2sim/kick_basket/editor_correct_65_hulls
```

The repair page opens at `http://localhost:8092/` when `--port 8091` is used
(`--port` is the embedded 3D Viser server; the control page uses the next
port). It starts paused at source frame zero. The single timeline only seeks
cached physical states. Its orange dots jump to repair starts. One button
plays, pauses, or commits a staged repair. The left and right hands have
independent one-dimensional open/close sliders.

The earlier `zup_baseline` used a different `L_8192_64_64` deployment model,
fed Newton world angular velocity to a local-frame policy input, and started
the mid-clip motion with zero joint/root velocity. Its fall does not describe
the checkpoint selected by the reference play script.

Earlier isolation tests with the mismatched deployment model in the same RSS scene show that holding the first reference
joint pose without gravity compensation lets the pelvis fall 1.511 m in three
seconds. Setting `RSS_KICK_DIAGNOSTIC_GRAVCOMP=1` holds that static pose to
0.18 mm, but a moving reference still diverges, and BFM diverges further.
Gravity compensation is therefore only a diagnostic branch, not a repaired
take. See `outputs/humanoid-real2sim/kick_basket/stability_diagnosis.json`
and `gravcomp_baseline/analysis.json`. The current BFM/MJCF/RSS control contract
needs calibration or a physically feasible retarget before local editing.
Scaling all position gains by four still lets the held first pose fall 1.396 m,
so a simple gain increase does not resolve it.

## Stage 5 basket physics candidates

The original local scene remains at 0.8 kg and sliding friction 0.8. Its
visual mesh spans z=0–0.43 m and has a geometric center of mass near 0.222 m.
`scene_stage5.py` is a separate, provisional candidate: mass 1.6 kg, friction
0.35, and local center of mass z=0.12 m. The contact mesh, initial pose, and
robot tracker are unchanged. The center of mass is applied to the built model
and propagated to the MuJoCo solver. These numbers are tuning hypotheses, not
measurements of the physical laundry basket.

The current 150-frame rollout gave 0.0234 m basket XY displacement, versus
0.0142 m in the original and 0.2118 m in the reference. Pelvis mean position
error was 0.0317 m and the robot stayed upright. In a separate 3-second
free-rest simulation the candidate had zero horizontal drift and less than
0.000004° maximum tilt; it settled 1 mm vertically from its stated initial
height. `scene_stage5_low_friction.py` tests friction 0.15 with the same mass
and center of mass; displacement was 0.0235 m. Lowering friction
further did not close the contact gap. The left ankle's contact-window mean
reference error is still about 0.136 m, so this Stage 5 candidate needs a
Stage 4 contact correction before it can be accepted as a repaired take.

The candidate is open in the interaction page at
`http://localhost:8102/` (3D viewer port 8101). To reproduce its unedited
rollout, use the baseline command above with
`--scene-file scripts/humanoid-real2sim/tasks/kick_basket/scene_stage5.py`
and a separate `--out` folder. Results and input hashes are in
`outputs/humanoid-real2sim/kick_basket/stage5_candidate_01/`;
the friction diagnostic is in `stage5_candidate_02/`.

`scene_stage5_v2.py` keeps mass 1.6 kg, lowers friction to 0.30 and the local
center of mass to 0.08 m, as requested. Its 3-second free-rest check has no
horizontal drift or tip. The unchanged 150-frame motion moves the basket
0.0228 m; the robot remains upright, with 0.136 m mean left-ankle reference
error in the contact window. This version is open at `http://localhost:8104/`.
Its rollout, analysis and rest check are in `stage5_candidate_v2/`.

### Saved repair and recovery check

The user-saved v2 edit is in `editor_stage5_candidate_v2/motion_repair/`.
Its 266 simulated frames contain reference playback through output frame 57,
then a left-ankle takeover through output frame 265. No recovery segment was
saved, so it is a physical rollout rather than an exported motion candidate.
Source time reaches its last frame (149) at output frame 149 and stays there
for another 116 simulated steps. The basket moves about 0.156 m by the final
output step, mostly after source time has stopped. Its final position remains
about 0.178 m from the source basket's final position.

`check_recovery.py` loads the saved robot/object state into a separate MuJoCo
instance and tries the actual 25-step interpolation controller, then lets it
settle under the same completion rule (maximum selected-body position error
0.05 m, angle 0.25 rad, joint error 0.2 rad). From output frame 265, a match to
reference frame 77 still had 0.112 m maximum body error after 75 steps.
Reference frame 117 was better but did not complete after 200 steps; its final
maximum body error was 0.073 m. The detailed trajectories are in
`outputs/humanoid-real2sim/kick_basket/recovery_checks/`. The live editor
shows frame 117 as a reference ghost without changing the saved cache.

### Recovery editor

The updated page at `http://localhost:8106/` starts from the user's saved
266-slot physical cache. **Enter Recovery** hides the five-link edit controls
and pauses at the current cached physical state. Its single timeline selects
the reference endpoint and moves a translucent full-robot preview; the default
endpoint is takeover start plus takeover elapsed frames, clamped to source
frame 149. A duration slider sets 0–10 s in 50 Hz steps. **Play** tracks a
full 43-joint and floating-root interpolation through BFM/RSS physics, then
continues the original reference. **Return to Repair** restores the entry
slot and its original contiguous cache, retaining the trial as a separate
branch. The original saved rollout remains
untouched.

The review export has four explicit segments: source reference before repair,
achieved full-body qpos during repair, the full-body interpolation reference,
and source reference after recovery. `candidate_mapping.npz` records the
phase/source mapping, while `rollout.npz` retains physical achieved states.
A 0-second duration makes the interpolation segment empty and switches to the
selected source frame on the next physics step. A separate 1-second trial from
the saved final state to source frame 117 produced 58 + 208 + 50 + 32 = 348
output frames and reached source frame 149. The candidate is marked
`candidate_unreviewed`: the qpos seam into recovery was about 0.001 rad and
the seam back to source about 0.017 rad, but the achieved robot still diverged
from the source after tracking. This is a workflow test, not an accepted repair.
