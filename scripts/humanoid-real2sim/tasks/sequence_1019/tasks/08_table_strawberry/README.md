# Table strawberry into the lined rectangular basket

`prepare_candidate.py --run <existing sequence run> --version <new version>`
creates an immutable 90–99 s candidate at 50 Hz. It reads the existing continuous
body prior, the accepted 07 tabletop-brace overlap before 91 s, and the archived
09 basket-carry overlap from 98 s. It never overwrites these inputs.

The second strawberry is a separate instance from the floor strawberry. The
source videos show right-hand pickup around 92.2–92.6 s, transport around the
dining table, release around 96.7–97.0 s and withdrawal before the bilateral
basket carry. The generated object trajectory is a kinematic estimate. Its
held segment uses one fixed palm-relative transform. The short release drop
is a ballistic estimate with a mesh-derived support height; it is not a
physics observation.

The script produces the seven-field robot motion, source sync, original human
reference, naming order, object tracks, a full 09 strawberry successor track,
and a numerical review. `tracks.json` is for the existing shared kinematic
viewer's `--tracks-manifest` argument. `scene.py` implements that viewer's
`build_sim(asset, motion)` callback and is also a live-page scene.

The task-specific initial scene uses five separate basket collision meshes;
it does not replace them with a convex hull across the opening. The hidden
table support is a provisional planar proxy at 0.754 m. Its extent and camera
registration remain subject to scene review. `RSS_SEQUENCE_RUN` selects the
run containing `scene/08_table_strawberry/basket.obj`.

No candidate is automatically accepted. The review deliberately preserves
failed collision frames, unsupported camera calibration, incomplete swept
geometry checks and missing agent playback review. Initial free-rest success
does not establish successful manipulation, and the inherited 09 contact
preload must be distinguished from new 08 rim collisions.

`audit_body.py <candidate> --bidirectional` independently checks all 47 enabled
robot collision shapes against the original room and the accepted 07 terminal
chair, with source-contact foot intervals and exact capsule floor minima.
`--motion-file body_prior.npz --output-name independent_body_prior_audit` runs
the same examination without the interaction correction. The collision check
uses both directions of edge/triangle intersections; open scene surfaces and
tessellated primitives limit it to geometric evidence, not dynamics.

The v13 whole-body check exposed left-hand tabletop penetration at
97.93–98.03 s, including the original 09 initial pose. `repair_left_connection.py`
creates a separate candidate with bounded left-arm correction.
`smooth_connection.py` broadens that correction and smooths the connection:
v15 reduces left-arm peak acceleration from 196.09 to 80.94 rad/s² and has no
room/chair intersections in the 100 Hz check. Its candidate-only correction
affects 97.4–98.20 s; integration must retain these frames before returning to
the original 09 trajectory at 98.22 s. Accepted source files remain unchanged.
No hardware acceleration limit was identified, so the report gives measured
kinematic peaks without claiming hardware acceleration acceptance.

The later v20 package supersedes v15 for integration. It uses the common
`shared_level_room_floor_v3.json` transform, whose SHA256 is
`2e935c654abc4aac4a1f324919e8aa0217a1c9c5c8c4e4df7709618d1118cd11`.
The tracked basket and accepted chair remain in their original world frame.
The initial strawberry keeps its original dimensions; its pose follows fixed
marker-camera projections, an independent 91.5 s holdout, full-mesh table
support and a native RSS rest trial. `geometry_active.json` selects this
shared geometry for the live scene. Source cameras use image Header timing.

v17 rebuilds the grasp for that geometry; v18 and v19 repair and smooth the
left-arm connection. The v19 whole-body audit has zero room/chair edge
intersection records at 100 Hz. Its unchanged robot, geometry and basket
hashes justify reuse in v20. The independent body prior has 4,153 hand/wrist
intersection records and is archived separately, rather than being declared
interaction-ready.

`apply_release_settling.py` creates v20 by replacing the postrelease object
segment with a measured native RSS static-basket settling trace. It maps the
relative pose onto the inherited basket trajectory and writes a consistent
full-09 successor track. This is a hybrid kinematic estimate; it does not
establish moving-basket or robot manipulation dynamics. The failed 5.29 mm
terminal rest and successful 0.16 mm terminal rest remain in the evidence.
Initial rest drift was 3.02 mm. The native same-camera holdout render and
source comparison are retained in the final review package.

Use v20 through 98.20 s and original09 q/pose from 98.22 s. Compose the full
timeline before recomputing velocity fields, since independently sliced
central differences have endpoint discrepancies. The original07/09 files
remain unchanged. Stage3 interactive visual acceptance and remaining
calibration uncertainty are explicit review items; the final directory
does not imply acceptance.
