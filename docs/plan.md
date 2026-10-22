# Plan

## The task

Dataset 3 asks for a standalone FANUC ER-4iA to grasp and sort objects reliably using its cameras. The judges want it running
on the real arm, against a measured baseline, with a business case.

**Problem we pick:** high-mix, low-volume manufacturing. Hand-programming each new part costs a changeover every time.
A cell that can work out unfamiliar rigid parts by itself makes changeovers nearly free.

## What the cell gives us

- **Arm:** FANUC ER-4iA. Fingertip targets in UF0 (mm; table at z ≈ −335). Keep within about 550 mm of the base.
  The real arm lags its targets by about 0.8 s.
- **Gripper:** parallel, open or closed only. About 12 mm between the faces when open. Fingers are 76 × 12 × 6 mm with a hooked notch.
  (Which gripper it is, SMC or SCHUNK, is still to be confirmed.)
- **Cameras:**
  - table (overhead): RGB plus depth, 640×400.
  - wrist: RGB plus depth, 848×480.
  - side: RGB only.

  Depth arrives **live** in each observation but is **not** in the recorded datasets, so we save it ourselves.
- **Control loop:** one snapshot per `act()` call. `act` returns 1–100 targets, which play one every 0.1 s. There is no time limit
  on `act`, so look-think-act is fully supported.

## Approach

Two complementary systems, after Spelke's principles of object perception:

1. **Physical (no learning):** depth plus motion gives rigid objects, their 3D shape, where they can be gripped, and how they rest.
2. **Semantic (VLM):** names objects and decomposes the instruction into steps. It never commands motion.

Mechanisms:
- **Look closely.** The overhead depth is coarse: about 2–3 mm per pixel, and screws nearly vanish. Fusing wrist-camera views gives the detail
  (in the synthetic scene, height-map coverage went from 13% to 88%). With the fingers pointing down, the wrist depth camera sits
  about 100 mm in front of the fingertip.
- **Nudge to perceive.** Flush-tiled parts are one slab to a depth camera. Push slowly, compare before and after in 3D:
  whatever moves as one rigid body is one object (common fate). Pushing also clears room for the fingers.
- **Grasp from shape.** An analytic search finds opposing surfaces under 12 mm apart with both finger slots clear on the way in and out.
  Start top-down on a height map; go to full 3D later.
- **Placement.** Stable resting poses from the reconstructed mesh (uniform density assumed).
- **Close the loop.** Every action has an expected outcome. Look, compare, retry. The same goal-state loop runs at every level:
  "arrange these 10 parts" breaks into "move part 4 to B", which breaks into "push it free, grasp, verify, place".
- **Inspect everything.** Each stage logs to Rerun so a person can see what the system believes.

## Measured claim

Pick success on cluttered or awkward rigid parts: **grasp-only vs. scan, push, then grasp**, 10 rollouts each, against the blind baseline.
Every rollout goes in a trial log (code version, setup, outcome, interventions).

## Build order

1. Scan-and-log policy, run on the real cell, to get real RGB-D frames.
2. Top-down grasp search, visualised (finger footprints drawn as valid or blocked).
3. Push and motion segmentation.
4. Goal-state loop tying perception, grasp and push together.
5. VLM layer for naming and task decomposition.

## Open questions

- Gripper model and true finger opening.
- Whether custom objects and a "dump inside a taped rectangle" setup are allowed instead of listed coordinates.
- Depth quality on real parts (shiny screws, edges, shadows).
