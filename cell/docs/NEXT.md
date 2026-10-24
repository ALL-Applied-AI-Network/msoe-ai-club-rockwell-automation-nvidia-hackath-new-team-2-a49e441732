# Next: what the first real-cell runs taught us, and what to build (Oct 24)

The chat app reached the real cell (run `r20261024-1f2d05`, trials t00–t03): it connects, moves, waits properly, and
our kinematics match the real arm exactly (FK of the reported joints = the cell's tcp_pose, to 0.1 mm, straight-down
pose). But perception falls apart on real depth, so it never closed the jaws. Recordings: `cell/runs/2026102415*_chat/`
(narration + VLM close-ups; the raw images/depth were NOT recorded until commit f964914).

## What went wrong on the real cell

1. **Depth noise -> phantoms.** A table with six 10 mm cubes became 84–90 "things": 5 x 3 mm specks, some "40 mm
   tall". The VLM then labelled blank table as "white 10 mm cube, 95% sure" (it describes, it doesn't doubt).
2. **A phantom box.** Noise merged into one ~97,000 mm² blob around the cubes; box detection took it for a box, so
   every cube was "next to a box": "can't grip it and can't rake it". Real boxes were missed ("one box", "no boxes").
3. **Scans overwrite instead of accumulating** (Joel). `lab/fusion.py:carve_merge` drops every old point in any cell
   the new look covers. Nothing averages out; a side view's points are erased by the next top view; a thing's 3D
   model only gets built if one single look sees it whole and clean, which real noise never allows -> no grips.
4. **Only top-down looks** (Joel). The wrist camera looks along the gripper and every look keeps it pointing down;
   "from the side" only moves the camera sideways. Sides, seams between touching parts, heights: lost.
5. **The arm pre-check blocked valid moves** (fat finger capsules, IK stricter than the controller near J6's wrap).
   Fixed for now: on the real cell it only holds back a fingertip into the table; the rest is logged.
6. **The chat turned "keep going" into rules and restarted the job.** Fixed: a `resume` tool; history trimming fixed.
7. **Instructions are ignored as data.** "Six 10 mm cubes, two of each colour" changed nothing: it rediscovered it all.

## To build (in the simulator first), in this order

0. **Look first, then touch** (Joel: "just use Luna to look at one photo so it can say: there's one object, the blue
   cube, over there"). Today depth proposes ~90 blobs and the VLM is asked about every one: noise gets 90 questions.
   Instead: one glance by the director model at the whole table image (~8 s): what's here, roughly where (a rough box
   per object). That list gates everything: only depth blobs where the glance named something become things; the rest
   is noise (no question, no look, no test). Back it with cheap checks: a coloured part on the white table is easy to
   find by colour in the image; depth only has to agree something stands up there. With the scene card ("one blue
   cube") it's expect one, find one. There's an unused full-table `vlm.gist` from the first lab days to start from.
   Cap the close-up questions per look.

1. **TSDF fusion instead of `carve_merge`** (KinectFusion; Open3D's ScalableTSDFVolume is already installed). Every
   look adds weighted evidence to one 3D grid; noise averages out; free space seen through a phantom outvotes it;
   angled views keep their walls; shapes accumulate. Segmentation, grasping and models read the fused surface.
2. **A scene card up front** (the usual robotics shortcut, for a reliable demo): declared objects with sizes and counts
   (e.g. 6 x 10 mm cubes: 2 red, 2 yellow, 2 blue), the boxes' colours and positions (no detection needed), the work
   area. Perception rejects what matches no declared size (kills the phantoms); a thing that matches needs no test.
   Given as a file or said in the chat (the agent turns it into the card). Instructions become priors to validate.
3. **Angles in the view search** (Joel: "expand the search space"). Next-best-view also scores tilted looks (30 and
   45 deg toward the target, 8 directions, plus straight down); lab.armcheck drops the unreachable; most new
   information wins. Phantom test: seen from two angles?
   - Before sending tilted looks to the real arm: confirm the tilt convention with one read-only check of a tilted
     pose (FK of its joints vs the cell's tcp_pose; straight-down already matches).
4. **Real depth noise in Isaac.** Measure it from the saved real frames (`runs/<session>/obs/*.npz`, from the next
   session on): speckle, flying pixels at edges, holes, ripple, error vs distance; add the same to Isaac's depth so
   perception is tuned at home.
5. **VLM sanity:** allow "nothing there" (count 0); reject a name that contradicts the measured size (a 22 x 4 mm
   "10 mm cube").
6. **Strict box detection:** ~60 mm with a rim and a well, or just the scene card's boxes.
7. **Fix lab.armcheck's fingers** (test the fingertip itself, not a capsule 4 mm fatter) before it blocks anything again.

## Then (agreed with Joel)

- **Play with the simulator** (the chat app, Joel's hands).
- **Teach by demonstration**: Joel drags/jogs the arm in the sim; record the motion relative to the object handled;
  replay moves with the object (pouring a box out is the first candidate: see `lab/skills.py`, parked).
- **The rewrite**: a few physical skills (look, pick, place / place-on-top, rake, push, pour) + one planner composing
  them from the goal (sort, stack, ...), the scene card as its prior. Keep perception + world model, drop the
  explore-era hacks. Stacking falls out of place-on-top.

## Open questions for the real cell

- Jaw axis: fingers down at R = 0, do the jaws close left-right (our JAW_AXIS_DEG = 90, the twin's) or front-back?
  Never seen yet: it hasn't gripped anything long.
- The boxes' real size: outer, wall, height.
- The team letter for task_id (submitted without one; accepted).
