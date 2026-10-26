---
name: real-to-sim
description: Rebuild a real bench, scene or reference clip in simulation on the live page. Use when asked to real-to-sim a photo or video, twin a real cell, match a reference picture, or build a scene interactively and watch it appear. The page rebuilds reference-knowledge/live/scene.py on every save; ./ctl measures the scene against the photograph, gates it, and films what the robot does.
---

# Real to sim on the live page

`reference-knowledge/live/scene.py` runs on every save, and the `rss.Scene` it binds is
built and swapped into the open page without a reload. `./ctl` (run from
`reference-knowledge/live/`) is the command line onto that process. Do not run the scene
yourself: edit the file and read what the page and its readings show.

Every command, flag, output file and reading is in
[`reference-knowledge/live/REFERENCE.md`](../../../reference-knowledge/live/REFERENCE.md), with the
frames and units. The API is in `reference-knowledge/00_getting_started/README.md` and
each example folder's README; the names are in `./ctl catalog`. Read those
rather than guessing a constructor or a name.

## The tools

- `./ctl status` / `./ctl up`: a page is up, or start one and tell the user
  its URL. A failed rebuild leaves the previous scene up with the traceback
  in `last_error`.
- `./ctl ref frames <clip> --at 0:start,<t>:<event>,end:end`: stills named by
  their second, a contact sheet, and `ref/timeline.json` (the moments you
  chose, which `compare --take` uses). View them with your image tool;
  `./ctl ref show ref/<still>` puts one in the page's sidebar.
- `./ctl snapshot --camera <name> --beside ref/<still>`: the render beside
  the still, full size. Open it after every save, before looking at numbers.
- `./ctl compare --camera <name> --key <thing>=<how> ... --note "<why>"`: the
  pair (`cmp_NNN_beside.png`); per keyed thing the size ratio, pixel offset
  and overlap; and a look sheet (`cmp_NNN_look.png`): photo outlines in
  yellow, render outlines in cyan, each thing up close, and both pictures
  with the things painted out. Open the look sheet every time and list, per
  thing, what the photo has that the render lacks (a lid seam, latch tabs, a
  label, a chamfer, a colour, a texture) and what stands behind it (table
  top, floor, wall, furniture); that list is your next edit. Keys:
  `case=red`, `socket=@1240,410` or `case=red@980,640`.
- `./ctl check --camera <name> --key ...`: PASS/FAIL with its number for
  builds, finite, camera aspect, keyed things present, worst overlap, and
  drift when left alone. Advisory only; explain any FAIL you accept, with the
  number.
- `./ctl eval "<generator>" --record <take> --camera <name>` films a routine.
  When it ends it is compared to the clip's timeline with your last compare
  keys (or run `./ctl compare --take <take> --key ...`); the strip
  (`cmp_NNN_strip.png`, clip anchors above, take frames below) and a verdict
  (`2 of 3 did what the clip did; lid stayed (clip moved)`) go on the board.
  Answer for the clip and for the take: the first state, the last state,
  what happened between, what touched what. If the verdict names something
  the clip moved that stayed put, fix the scene before the routine.
- `./ctl log`: every reading so far, with the hash of the scene code it
  measured.
- `./ctl show <png> --label "<what>"` posts a picture to the page's `Live`
  sidebar folder for the person watching (the same label replaces it;
  `--clear` removes all). `compare` posts its pair and look sheet, and a take
  its strip, by themselves. Post what you are judging as you go: a still, a
  close-up, the snapshot.
- Bind `routine` (sim -> generator, one yield per control step) or
  `routines = {"tidy": ..., "open lid": ...}` in `scene.py` to get a **Run**
  button per routine and a **Reset**; `./ctl play [name] [--record <take>]`
  presses them, and `status` lists them under `routines`. A routine that
  declares a `session` parameter is handed the page's, so it can hold the
  arm panels off while it drives a gripper.
- `eval`, `run`, `pause`, `resume`, `reset`, `catalog`: see REFERENCE.md.
  `eval` shares `scene.py`'s namespace until the next save; `run N` is
  refused while a routine plays; `reset` stops a playing routine and
  finishes its take. While a routine plays, the page's own stepping is held
  whatever Pause says.

## What a finished twin is

- The first frame agrees: each thing the clip shows is in the scene at its
  size and place, through a camera where the clip's was, and to your own eye
  on the pair and the look sheet it looks like the clip: the things in their
  detail (a case with its latches, seam and label, not a red box) and what
  stands behind them (the table top, floor, wall and furniture the clip
  shows), with their colours and textures.
- The gate passes, or the report says which line fails and why that is
  right for this scene.
- The take moved what the clip moved, by about as much; the before and the
  after match the clip's first and last frames.
- Where the reference cannot settle a number, the report says which number
  and what was assumed.

There is no prescribed order: what to build first, when to compare, and how
many saves are up to you. Report readings, not impressions: the `./ctl log`
lines, the gate, and the take comparison.

Two cases are not done. A first frame that matches the numbers but not the
look (the look sheet's list still standing), and a routine retried take
after take while the verdict keeps naming something the clip moved and the
take did not. In both, the next save fixes the scene (the mechanism, the
texture or the room), before the next take.

## Rules of the page

- Poses are in the cell's work-surface frame, as everywhere in the core API;
  `snapshot --eye/--look-at` are world metres.
- `rss.view`, `rss.draw` and `rss.viewer` are no-ops while the file runs, so
  a script from `reference-knowledge/` pastes in unchanged.
- A snapshot is path traced and draws the scenery (a room, a mat, a texture);
  the page draws the bench and what you added.
- A rebuild of the starter scene takes about 6 to 7 s, about 4.8 s of it
  CoACD on the YAM fingertips; the first build in a process takes about 10 s.
  `eval` blocks the page for as long as it runs unless it yields. A save also
  re-imports the helper modules beside `scene.py`.
- `run/` holds every snapshot, comparison, take and the log, and is not
  committed; `scene.py` is your result, so leave it in the state you are
  reporting.
- What the user drags on the page is what your next reading sees; if the
  scene looks moved, `./ctl reset` before judging it.
