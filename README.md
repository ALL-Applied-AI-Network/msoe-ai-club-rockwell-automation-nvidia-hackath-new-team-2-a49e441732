# Her Majesty's Magical Severed Homunculus Arm

Rockwell Automation + NVIDIA Physical AI Hackathon (Fall 2026), **Dataset 3: Grasping and Sorting** on a FANUC ER-4iA.

**Goal:** a cell that handles unfamiliar rigid parts without reprogramming. It looks closely with depth, nudges things to
find out what is one object, grasps by 3D shape, checks its own work, and leaves the meaning ("those are jacks, line them up")
to a vision-language model that never drives the arm. See [docs/architecture.md](docs/architecture.md) (three-phase design, tools, reading list) and [docs/plan.md](docs/plan.md).

> Work in progress. This README becomes the project write-up (problem, approach, results, how to run, credits) before the finals.

## Layout

| Path | What |
|---|---|
| `cell/` | SIDC's client package for the real cell, with our policy. `policy.py` is the policy the cell runs. |
| `cell/lab/` | The lab: `simcell.py` (MuJoCo sim cell), `catalog.py` (object kinds), `trial.py` (run trials), `perception.py` (the robot's own segmentation, scored against truth), `session.py` + `viz.py` (one Rerun layout for live and replayed runs), `recorder.py`/`replay.py` (every observation saved and replayable), `vlm.py` (vision-language model), `policies/scan.py` (look-before-touching sweep, also runs on the real cell). |
| `docs/architecture.md` | The three-phase architecture (whiteboard + diagram), internal tools, simulators, reading list. |
| `docs/plan.md` | Approach, build order, open questions. |
| `docs/pitch-notes.md` | Raw material for the business case and video: plain-language story, extensions, limits. |
| `docs/event-guide.md` | The organizers' original README: dates, links, submission requirements. |

## Run

Install [uv](https://docs.astral.sh/uv/), then:

```bash
cd cell
uv sync
uv run run.py --mock                    # check the policy against a local stand-in for the cell (records to runs/)
uv run python -m lab.trial list         # simulated trials (lab/trials/*.yaml)
uv run python -m lab.trial run cube_tile     # watch the robot shuffle a flush tile apart and test each piece
uv run python -m lab.trial run mixed_pile --print   # same on a dumped pile, narration printed too
uv run python -m lab.replay             # replay the newest run (real, mock or sim) into the viewer
```

Real-cell runs: see `cell/README.md` (submit, book a slot on hackathon.sidclab.org, `uv run run.py`).
Never commit `cell/.request.json`: it holds the secret for the team's submission.

## The lab (runs on a Mac)

A simulated cell for trying everything before using real robot time. It speaks the same language as the real cell,
so a policy can't tell the difference.

- **Order objects** in a small YAML file (`cell/lab/trials/`): kinds from `uv run python -m lab.trial kinds`
  (10 mm cubes, dice, screws, jacks, a ring, Lego, a toy soldier, dowels, a cup, the bin, ...), colors and counts.
- **Layout:** `spread` (set down apart) or `dump` (poured from a tube into a pile, with MuJoCo physics).
- **Cameras** are rendered through the real cell's calibration: table, side and wrist RGB, table and wrist depth.
- **The viewer** shows the true world and the robot's mind side by side, the raw camera inputs, what the VLM thinks,
  a table of what the robot has found, and a narration line per step. The robot's mind is built only from its cameras;
  in sim it's scored against the truth every step.
- **VLM (optional):** copy `cell/.env.example` to `cell/.env` and add a key for any OpenAI-compatible vision model
  (NVIDIA build.nvidia.com works). `cell/.env` is git-ignored.

### The explore loop (`cell/lab/policies/explore.py`)

Lay the pieces out like a jigsaw puzzle, one hypothesis at a time:
1. **Sweep**: close wrist-camera looks at the pile, fused depth, blobs.
2. **Hypothesis**: "blob #k is one object".
3. **Test**: grip part of it (analytic grip finder, `grasp.py`) and carry it to a clear staging spot, or rake through it
   if nothing can be gripped. A test has to be able to fail, so larger things are gripped near one end.
4. **Look again and compare** (`world.py`): came along whole with nothing left behind = one object;
   part stayed behind = several (each piece becomes a new hypothesis); nothing arrived = the grip slipped.
5. Repeat until every thing has been carried whole once.

Simulated results (seed 0, robot time):

| Trial | Objects | Result | Robot time |
|---|---|---|---|
| six_cubes (spread) | 6 | 6/6 confirmed | 147 s |
| six_cubes_pile (dumped) | 6 | 6/6 confirmed | 182 s |
| cube_row (flush row) | 6 | 6/6 confirmed | 220 s |
| cube_tile (flush 2×3) | 6 | 6/6 confirmed (2 slipped grips, then a rake) | 217 s |
| mixed_pile (cubes, Lego, screws, dowels) | 9 | 9/9 confirmed | 440 s |

Known limits: tangle-prone parts (jacks, rings, chains) are out of scope; the sim has the arm's lag but guesses friction
and grip force; two parts glued or wedged together would pass as one.

## Status

- **Baseline v0** (`cell/policy.py`): a blind scripted pick of one screw at known coordinates into the yellow bin, in tcp mode.
  Passes the mock. `GRASP_Z` and `JAW_OFFSET` are still unverified on the real arm.
- **Seeing it** (`cell/lab/`): every observation the policy gets is recorded (RGB, depth, joints, actions sent). Any run replays into one
  fixed Rerun layout: 3D view (arm, cameras, point clouds, objects, planned paths), the three camera images with object outlines and ids,
  depth images, a height map, and a narration log. Synthetic runs carry ground truth, so each perception step can be scored.

## Credits

- Client package, calibration and mock cell: [SIDCLab/PhysicalAI-Hackathon-2026](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026),
  Apache-2.0 (`cell/LICENSE`).
- Libraries: Open3D, trimesh, Rerun, NumPy, SciPy, OpenCV.
