# For the agent working in this repository

RoboSimStudio is a Newton-native simulation toolkit. Packages:

- `robosimstudio/`: the core API; builds and steps scenes.
- `robosimstudio_studio/`: a browser page over any scene.
- `robosimstudio_bench/`: tasks and their scoring.
- `robosimstudio_policy_agent/`: policies as graphs of robot tools.
- `robosimstudio_real2sim_agent/`: the live page that rebuilds a `scene.py`
  on every save, and the `./ctl` command line onto it.

## Read first

1. [`README.md`](README.md): what is here and how to run it.
2. [`reference-knowledge/00_getting_started/README.md`](reference-knowledge/00_getting_started/README.md)
   (`Scene`, `Sim`, `Ref`, the constructors) and each example folder's README. Poses
   are in the cell's work-surface frame: `at=(0, 0, 0.2)` is 200 mm over the
   middle of the bench, whatever bench (frames and units:
   [`reference-knowledge/live/REFERENCE.md`](reference-knowledge/live/REFERENCE.md#frames-and-units)).
3. `reference-knowledge/`: numbered scripts, shortest first. `reference-knowledge/01_cloth/cloth.py`,
   `reference-knowledge/02_soft/soft.py` and `reference-knowledge/03_rope/rod/` are the shipped cloth,
   soft-body and rope implementations; read them to see how each works.

## Checks

- `uv run python scripts/testrun.py --tier fast`: no GPU, gates every commit.
- `uv run --extra gap python scripts/testrun.py --tier medium`: needs the GPU;
  builds and steps scenes.
- `uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check`.
- `reference-knowledge/live/run/` is not committed. A commit message is a lowercase
  `area:` prefix and one declarative sentence
  (`studio: the room, and lights that match the render`).

## Real to sim: a real bench, scene or clip, rebuilt on the live page

When asked to rebuild a real scene from a photo or video, match a reference,
twin a real cell, or build a scene interactively, use the live page in
[`reference-knowledge/live/`](reference-knowledge/live/README.md) instead of writing a script and
running it blind: edit `scene.py`, and the page rebuilds it on every save.
From `reference-knowledge/live/`:

```bash
./ctl status                          # a page is up (names, positions, last error), or it exits 2
./ctl up                              # if not: start one; prints the URL (--share for a public one)
./ctl ref frames <clip> --at 0:start,4.5:lid_open,end:end   # stills named by their second, a contact sheet, ref/timeline.json
./ctl snapshot --camera <cam> --beside ref/<still>          # the render beside the still, full size
./ctl compare --camera <cam> --key case=red --key socket=@1240,410 --note "<why>"   # size, offset, overlap per thing, and a look sheet
./ctl check --camera <cam> --key ...  # the gate: PASS/FAIL with the number
./ctl eval "t.play()" --record take1 --camera <cam>         # film a routine; it is compared to the clip when it ends
./ctl play tidy --record take1        # the page's Run button: scene.py binds routine or routines={name: fn}
./ctl log                             # every reading so far, with the scene code it measured
./ctl catalog robots cells            # the names rss.robot() and Scene(cell=) accept
./ctl eval "sim['<name>'].position"   # python against the live scene
```

A snapshot is path traced and draws the room, the mat and the textures; the
page draws the bench and what you added.

The method and what counts as a finished twin are in
[`.claude/skills/real-to-sim/SKILL.md`](.claude/skills/real-to-sim/SKILL.md)
(the `real-to-sim` skill for Claude Code). The `scene.py` contract, every
command and every reading are in
[`reference-knowledge/live/REFERENCE.md`](reference-knowledge/live/REFERENCE.md).

## Feedback memory and humanoid real to sim

For every task in this repository, record user reported failures, omissions,
and new requirements in
[`reference-knowledge/humanoid-real2sim/feedback/`](reference-knowledge/humanoid-real2sim/feedback/README.md).
Search by natural-language task, action, and symptom before solving. Read the
matching knowledge's conditions, observations, decision rationale, tradeoffs,
validation boundaries, and useful references; load old implementation artifacts
only when needed. Record feedback immediately as error, new requirement, or new
problem. Fix actual errors; give new requirements observable acceptance criteria.
For each case decide whether the main skill, shared tools, and reference knowledge
need updates. Use `repository` for work outside the humanoid pipeline.
For humanoid work, use
[`codex_skills/rss-scaletrack-real2sim/SKILL.md`](codex_skills/rss-scaletrack-real2sim/SKILL.md)
as the router. Keep take-specific approval in its stage ledger. Main skills hold
harness rules; reusable interaction knowledge belongs in reference-knowledge.

Reusable retarget, tracker, and visualization code belongs under
[`robosimstudio/humanoid_real2sim/`](robosimstudio/humanoid_real2sim/README.md), independent of
the skill. Put task programs and launchers under
[`scripts/humanoid-real2sim/`](scripts/humanoid-real2sim/README.md).
Keep full runs in `outputs/humanoid-real2sim/<date>/<task>/runs/<run-id>/`.
Organize [`reference-knowledge/humanoid-real2sim/`](reference-knowledge/humanoid-real2sim/README.md)
by reusable interactions and failure mechanisms. Knowledge is primarily prose:
conditions, evidence-based decision summaries, alternatives and tradeoffs,
transfer limits, and helpful references. Do not turn internal reasoning traces,
version logs, task-specific numeric recipes, or code listings into the main text.
Text-only entries with meaningful references can be indexed without code,
motion or artifacts. References explain why they are useful; distinguish local
experience, hypotheses, scoped user approval and independent validation.

Existing selected `artifacts/` and `results/` remain optional provenance and keep
their hashes and acceptance scope. New entries need not copy binary data; full
experiments remain in outputs. If selectively retained, pending/failed evidence
goes in artifacts and scoped accepted outcomes in results, with Git LFS for
binaries. The current contract is
[`KNOWLEDGE.md`](reference-knowledge/humanoid-real2sim/KNOWLEDGE.md).
Search with `python scripts/humanoid-real2sim/feedback.py search --query '<task action symptom>'`.
The sole directory is `reference-knowledge/`; do not recreate an `examples/` alias.
