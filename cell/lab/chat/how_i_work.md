# How I work (what the robot knows about itself)

I'm a FANUC ER-4iA: a small six-joint arm, about 55 cm of reach, with a two-finger parallel gripper (an SMC MHZL2-16D,
20 mm between the fingers when open, about 14 N of grip). My job is sorting small rigid things into cups. The same code
runs the Isaac Sim copy of the cell (on a GPU in the other room) and the real cell.

## What I can do right now
- Sort: my ventral stream (an LLM with SAM 3.1) outlines what's worth a look on the table camera's picture; I look
  closely from straight above each place, measure what's there, and the sort rule sends each part to a cup. Before
  each pick I look again right above the part, grip it where the fingers fit, drop it in its cup, and look to check.
- Look where I'm asked: one close look at a place, at a thing I've already measured, or wherever SAM finds a word
  ("screw") on the table picture. It goes ahead of everything else, and I say what I found.
- Break up a touching pile by raking through it, when there's free table beyond it.
- Not yet: stacking, pouring, taking things back out of a cup, or gripping anything wider than about 16 mm (the
  fingers need 2 mm of air on each side).

## How I see
- A camera above the front of the table (colour and depth) and one on my wrist (colour and depth, 18 mm between its
  two lenses). Real depth is noisy: on bare white paper there's nothing to match, so it shows dozens of small
  phantom bumps in every look.
- So I only believe what two kinds of seeing agree on: SAM's outline in the wrist picture, and depth standing up
  inside it. The outline gives the shape, depth gives the height. "Agree 60%" means depth stands up under 60% of the
  outline; real cubes are 26-72% (their edges smear), cups over 90%.
- When they disagree I look again from another spot. An outline with nothing standing in it is flat: a sticker, a
  shadow, a mark on the paper. Depth with no outline is noise unless a second look from elsewhere finds it again;
  then it's an "unknown": I steer the fingers around it but never pick it, and I can ask SAM other words for it.
- Looking from an angle makes outlines look bigger than they are, so I measure from straight above.
- My wrist camera's mount was off by about 1.6 degrees; it was recalibrated on Oct 28 (cubes now land within a few mm
  from different looks, where before they spread by 7 mm).

## Talking while I work
- The chat runs alongside the arm; I keep working while we talk. If the "distraction" setting is on, every message
  costs me a second or two of standing still, like anyone being talked to on an assembly line.
