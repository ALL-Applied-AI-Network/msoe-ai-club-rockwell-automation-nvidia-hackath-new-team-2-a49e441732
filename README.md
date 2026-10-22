# Her Majesty's Magical Severed Homunculus Arm

Rockwell Automation + NVIDIA Physical AI Hackathon (Fall 2026), **Dataset 3: Grasping and Sorting** on a FANUC ER-4iA.

**Goal:** a cell that handles unfamiliar rigid parts without reprogramming. It looks closely with depth, nudges things to
find out what is one object, grasps by 3D shape, checks its own work, and leaves the meaning ("those are jacks, line them up")
to a vision-language model that never drives the arm. See [docs/plan.md](docs/plan.md).

> Work in progress. This README becomes the project write-up (problem, approach, results, how to run, credits) before the finals.

## Layout

| Path | What |
|---|---|
| `cell/` | SIDC's client package for the real cell, with our policy. `policy.py` is the policy the cell runs. |
| `cell/lab/` | Perception and planning experiments. `scene_viewer.py` renders a synthetic scene through the cell's real depth-camera calibration and logs it to [Rerun](https://rerun.io). |
| `docs/plan.md` | Approach, build order, open questions. |
| `docs/event-guide.md` | The organizers' original README: dates, links, submission requirements. |

## Run

Install [uv](https://docs.astral.sh/uv/), then:

```bash
cd cell
uv sync
uv run run.py --mock                    # check the policy against a local stand-in for the cell
uv run python -m lab.scene_viewer       # synthetic depth scene -> lab/out/scene.rrd
uv run rerun lab/out/scene.rrd          # inspect it in 3D
```

Real-cell runs: see `cell/README.md` (submit, book a slot on hackathon.sidclab.org, `uv run run.py`).
Never commit `cell/.request.json`: it holds the secret for the team's submission.

## Status

- **Baseline v0** (`cell/policy.py`): a blind scripted pick of one screw at known coordinates into the yellow bin, in tcp mode.
  Passes the mock. `GRASP_Z` and `JAW_OFFSET` are still unverified on the real arm.
- **Synthetic depth pipeline** (`cell/lab/scene_viewer.py`): overhead plus wrist-sweep depth, fused into a point cloud and a 1 mm height map.

## Credits

- Client package, calibration and mock cell: [SIDCLab/PhysicalAI-Hackathon-2026](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026),
  Apache-2.0 (`cell/LICENSE`).
- Libraries: Open3D, trimesh, Rerun, NumPy, SciPy, OpenCV.
