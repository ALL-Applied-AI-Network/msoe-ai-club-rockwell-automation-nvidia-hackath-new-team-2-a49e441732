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
| `cell/lab/` | Perception, planning and visualisation. `viz.py` (one Rerun layout for everything), `recorder.py` (every observation saved to `cell/runs/`), `replay.py` (any run into the viewer), `synth.py` (a known scene rendered through the real camera calibration, saved like a real run), `fusion.py` (depth → height map). |
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
uv run python -m lab.synth              # synthetic scene through the real camera calibration -> runs/, opens the viewer
uv run python -m lab.replay             # replay the newest run (real, mock or synthetic) into the viewer
```

Real-cell runs: see `cell/README.md` (submit, book a slot on hackathon.sidclab.org, `uv run run.py`).
Never commit `cell/.request.json`: it holds the secret for the team's submission.

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
