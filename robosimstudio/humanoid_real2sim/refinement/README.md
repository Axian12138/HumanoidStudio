# Offline dynamics refinement (experimental)

This is an opt-in sampling optimizer, separate from the Stage 4 tracker and
repair UI. It does not change accepted reference motions, the tracker, scene
files, or ScaleTrack packages. The first integration is the lined rectangular
basket from sequence 1019.

Run from the repository with a CUDA-capable environment:

```bash
PYTHONPATH=. .venv/bin/python \
  scripts/humanoid-real2sim/tasks/sequence_1019/tasks/09_lined_rect_basket/refine_dynamics.py \
  --candidate outputs/humanoid-real2sim/2026-10-20/sequence_1019__09_lined_rect_basket/runs/20-11-03.205681+0800/hoi_v21 \
  --project-initial --bfm-warmstart reference-arms --render-review \
  --samples 64 --iterations 4 --sigma 0.08 \
  --arm-noise-scale 2 --leg-noise-scale 0.25
```

The command automatically projects the initial overlap, generates a frozen-BFM
control seed, optimizes, replays, exports, and renders the review. It does not
train or modify BFM. Every invocation creates a date/task/run output directory. `scene.mjb` contains
the RSS-compiled model, `reference.npz` the reference and initial velocity,
`baseline.npz` the direct joint-target execution, and `optimized.npz` the
executed states and bounded position targets. `review.json` reports errors and
control replay discrepancy; `progress.jsonl` records optimization wall time.
The exported `candidate_motion.npz` contains actual executed robot motion;
`object_executed.npz` contains actual free-body motion, not the input estimate.
These candidates are never automatically accepted for training.

The algorithm uses a one-second horizon, commits 0.1 seconds at a time, samples
smooth joint-target perturbations with annealed variance, and applies an
exponential cost-weighted update. It retains the best *evaluated* candidate,
including the incumbent, rather than executing an untested weighted mean.
The cost tracks root position/orientation, robot joints, object pose, velocity,
and control smoothness. Dynamics, actuator torque limits, and collisions are
provided by the compiled RSS model. No root actuators, object welds, applied
forces, kinematic object updates, or reference resets exist in a rollout.

Candidate prediction uses batched MuJoCo Warp on GPU. The selected prefix is
executed continuously by CPU MuJoCo using the same compiled model; an independent
CPU control replay checks reproducibility. CPU/GPU contact solver differences
remain an approximation in candidate ranking. A GPU-only prototype showed
nonzero control replay discrepancy and was not accepted. GPU overflow bitmasks
are recorded, including line-search exhaustion, rather than hidden by log spam.
Only the offline model's line-search budget is raised to 50; source Stage 4
scene settings are untouched. The source scene disables robot self-collision,
so this experiment cannot certify self-collision safety.

This is a **SPIDER-inspired annealed MPPI baseline**, not a complete SPIDER
implementation: virtual contact guidance, its curriculum, robustification,
and the paper's exact objective are not implemented. It is not DynaRetarget
SBTO and does not implement CEM, expanding windows, or cached-prefix skipping.
The paper describes a 1.2-second horizon in its appendix; window size is a
configuration choice, not the defining difference between these methods.

Sources:
- [SPIDER paper](https://arxiv.org/html/2511.09484v1), sections 2.2–2.4 and A.2.
- [DynaRetarget paper](https://arxiv.org/html/2602.06827v1).
- [Official SBTO implementation](https://github.com/Atarilab/sbto).

An expanding horizon does not inherently require CPU execution. Variable-length
rollouts can use multiple graph captures or a fixed maximum horizon with masks;
whether that is worthwhile requires measurements. CPU/GPU implementation and
hardware differences preclude attributing wall-time differences to MPPI versus
CEM alone.

For an overlapping initial pose, `--project-initial` applies a bounded left-arm
collision projection, saving both the unmodified reference and the new initial
state. This is not an equilibrium solve. For this basket the original initial
middle finger intersects the dining-table proxy by 33.8 mm; a tested projection
reduced this below 1 micrometre with a maximum 0.114 rad joint adjustment.

`seed_dynamics.py RUN [--reference-arms]` creates an optional frozen-BFM control
warm start against the run's CPU model. It performs no policy training. Pass its
NPZ to `refine_dynamics.py --warmstart FILE --project-initial`; the entry point
checks initial pose, finite values, shape, and actuator names/order. The arm
and leg noise multipliers are explicit CLI options, recorded in the manifest.

`review_dynamics.py RUN` exports the executed robot/object motion, reconstructs
contact diagnostics with saved controls, writes conservative automatic gates,
and renders a three-panel video and an HTML review page. Serve only that run's
directory for review. The video uses collision geometry and labels the
kinematic reference separately from physical execution. Contact samples are at
50 Hz and do not bound all substep peaks. Even passing the automatic gates does
not imply human acceptance, robustness, or readiness for RL training.

The current objective additionally tracks torso, foot and wrist poses, wrists
relative to the free object, and penalizes robot-object contacts outside the
forearm/wrist/hand links. These are reference/contact costs, not applied virtual
forces. They were added after the initial object-dominated objective produced
low object position error by carrying against the upper body and changing the
human action. The review gates therefore also cover torso/object orientation,
intended bilateral contact coverage, unintended contact, and loaded-foot slip.
Do not select a candidate by object RMSE alone.
