# V8: the cell console and its mind

Scan -> carve -> label -> plan -> grab -> verify, agent-driven, on the Isaac twin and the real
FANUC ER-4iA through one interface. Built 2026-10-31. This file is the map for whoever patches it
next, possibly mid-trial, possibly in a hurry.

## Run it

    cd cell && SAM_REAL=1 uv run python -m v8.app     # http://127.0.0.1:8801

App page on 8801, its empty-at-boot Rerun viewer on 9091 (gRPC 9877). Logs: stdout (the launcher
usually redirects to a file; flights, grabs, crashes with full tracebacks all land there). The
chat shows every tool verdict and every crash ("I CRASHED: ..."); the dot on the chat header is
server truth: green waiting, amber working, red crashed. SAM runs on the GPU box through an SSH
tunnel (lab/sam.py starts it); the brains go through OpenRouter (keys in cell/.env - never print
them; V8_MODEL overrides the mind, V8_LABEL_VLM the crop labeler).

## The files (all of V8 is these four)

- **app.py** - HTTP shell: connect sim/real, task menu (stages trials, Run sends the instruction
  as the user), hands panel (sim only), objects panel, chat, reset/clear (which ABORT the running
  job via `eyes.stop_now`). Staging clamps fixtures into reach and re-rolls object layouts until
  everything is born inside the envelope (editing positions after settle does nothing: Isaac
  imposes settled body poses).
- **agent.py** - the mind: one pydantic-ai agent, full conversation history, tools = narrate,
  look, scan, look_at, inspect, label, set_plan, grab_to, put_in. A tool-less commentator answers
  the user while the main loop is busy. The persona carries the working philosophy; change it
  last, not first.
- **eyes.py** - everything physical. Carve: three-valued silhouette evidence per shot (mask=keep,
  in-frame-non-mask=carve, out-of-frame/occluded=unknown), blobs as voxel volumes, gravity drops
  hovering debris (stacks survive), coverage ledger of what the wrist has framed. Aiming: ik_for
  (roll is a free parameter; fingertip floor, far-low rule, base keep-out all checked AT PROPOSAL),
  _vantages (top + 40 deg + grazing 22 deg, FOV-sized standoff, aims at the volume centre).
  Flying: the datarun's joint-space mover, armcheck + real-mesh self-collision + blob obstacles on
  every plan, arrival verified from observed joints, failed plans re-planned from HOME only when a
  from-home plan exists (computation first), home() is the un-cornering reflex. Hands: grab_to
  (slide-in first - descend into PROVEN-empty space, slide in; top pick fallback), put_in (release
  above the container, near-rim pull for far bins, yes/no glance), clearances measured to the
  hull's REAL footprint, world updated ONLY by verification verdicts (rigid translate on confirm).
- **view.py** - Rerun: camera pyramid with the photo on its plane, cast rays, blobs as marching-
  cubes meshes, reach rings drawn from the planners' own constants, flash for the objects panel.

## Invariants - a patch that violates these is worse than the bug it fixes

1. **No swallowed errors.** No `except: pass`, no silent fallbacks, no error handler that can
   itself throw. A crash names its cause in the chat and the log. (lab/sam.py deliberately does
   NOT fall back to Meta's API; SimCell.observe deliberately raises on bad IK.)
2. **Checks are the only authority.** The world model updates on what verification LOOKS found,
   never on what a motion intended. A MISS leaves the model unchanged.
3. **Every motion goes through the gates**: joint limits, jump guard, capsule+real-mesh self-
   collision (fitted to the arm's two real October faults - do not loosen those constants),
   fingertip floor, far-low rule (beyond ~485 mm out, fingertips stay >= ~90 mm up), blob
   obstacles, arrival check. Same gates sim and real.
4. **Sim and real share one code path.** RealCell/IsaacCell both speak observe/play(kind="joints").
   The real cell wants a picture in every observe (use eyes._joints()), runs at tuned speeds
   (150/50 mm/s, set on connect), and raises TrialOver when the trial ends.
5. **The eyes never name; the mind never computes geometry.** SAM locates, carving shapes, the
   label tool names, the LLM decides, motion planning places vantages and paths.

## When it breaks (the fast-patch loop)

- Read the chat first (verdicts + crashes), then the app log (flight prints, "first plan failed:
  ..." reasons, full tracebacks). The failure usually names itself.
- `uvx pyflakes v8/*.py` after every edit - two of today's crashes were refactor stragglers it
  catches statically.
- Offline geometry check (no cell, no API):
  `uv run python <scratch>/eyes_check.py` equivalent: carve a synthetic 30 mm cube through
  _carve + _vantages; expect hull ~32x32x<=42 within 1 mm of truth.
- Restart = kill the process, same command. Reset in the page aborts the running job and
  re-stages; it does NOT need a process restart for code that is imported lazily per run (the
  agent tools are not - agent/eyes import at app start, so CODE changes need a process restart).
- Known sharp edges: bins are one-way (solid boxes in collision - nothing can reach INTO them);
  no stacking/tight packing (clearance refuses deliberate contact); no orientation control on
  placement; objects wider than ~16 mm across their narrow side are terrain, not cargo; the far
  bin of the standard row sits at ~510 mm, at/over the envelope - the near-rim release handles it.

## Credits

Built with Claude Code (Claude Fable 5) pair-designing with Joel Roberts; the architecture
(dorsal/ventral split, three-valued carving, verification-only world updates, slide-first
manipulation) is Joel's design.
