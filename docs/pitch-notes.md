# Pitch notes

Raw material for the business case, the one-pager and the video. The final story gets rewritten once results exist.
The audience includes people who don't think about technology at all, so lead with what it does, not how.

## The one-liner

**Dump a pile of parts in front of the robot, tell it what you want in plain words, and it sorts them.**
No programming, no training on your parts, no fixtures. You brief it like a new temp: "sort these", "use the cups",
"by color", "same shape together".

## The core idea (plain language)

- Most robot vision is trained: show it thousands of pictures of *your* part, and it learns to spot *that* part.
  New part, new training. That's the changeover cost.
- We don't train anything. The robot measures the scene in 3D with its depth cameras, the way you'd feel your way around
  a dark room. It works out "this is one solid thing, this is another" by looking closely and nudging. A general AI model
  that already knows what most things look like then says what each thing is.
- Two jobs, kept separate, like the brain: one part works out **where things are and how to grab them**; the other works out
  **what they are**. (Neuroscience: the dorsal and ventral streams.)

## Why it's foundational

- "Know every object on the table in 3D, without knowing in advance what they are" sits upstream of many other products:
  kitting, bin picking, inspection, inventory, packing, returns processing, lab automation.
- Depth is badly underused. The cell's depth cameras weren't even in the default feed. Most approaches work from flat images
  and pay for it with training data.
- **Extensions we don't build, but that the architecture enables:**
  - **Match objects to SKUs.** Compare each scanned 3D shape against a facility's CAD library and label it with the real part
    number while sorting. (Not built: no inventory list or CAD here.)
  - **Any scale.** Nothing in the method depends on object size. A bigger gripper or a bigger arm handles bigger things,
    and bigger objects are actually *easier* for depth cameras. Our 12 mm gripper is the limit of this cell, not of the idea.
  - Quality checks: compare the scanned shape with the expected shape and flag damage.

## It explains itself, and takes edits

- Given no direction ("here's a pile, sort it"), it makes up a sorting rule from what it sees, **and says why**:
  "I grouped by shape, because there were several of each shape and the colors were mixed."
- Ask "what are the groups?" and it answers in plain words. Say "I like it, but put the red ones separately" and it re-plans.
- Nothing irreversible (dropping parts into cups) happens until the rule is settled. That's the operator approval step:
  people stay in charge and can see the reasoning, which builds trust.
- If it still gets it wrong: "dump the cup out and try again." The robot can do that itself, which makes a memorable demo.

## Not just picking

- Things too big for the gripper, or flat, or round, can still be **pushed**: into piles, toward an edge, or rolled.
  A marker rolls; a toy car rolls along its wheels. The robot can learn which by pushing a little and watching.
- The whole loop is **look first, grip when you can, nudge when you can't** (in research terms, active and interactive perception).

## How success is judged

- Pass/fail is the honest human question: **given what the person asked for, did it sort them?**
  - No rule given: it states its own rule first. If the result follows that rule and looks sorted, pass.
    If most things end up roughly where they started, or in a heap, fail.
  - Specific rule given ("by color"): any obvious violation is a fail.
- Our own logs also count per-object correctness, attempts and recoveries, so there are numbers behind the pass rate.

## Honest limits (say them, they build trust)

- Two weeks, small arm, 12 mm gripper: the demos are deliberately simple tasks, but with almost no instructions and
  objects it was never shown.
- It struggles with: many objects, tiny or shiny objects, things it can't see well, soft or floppy things (rope, cloth: out of scope).
- The pitch: **look how general this already is on simple tasks; that's the base idea validated.** The improvements from here
  are engineering, not research.

## Claims to be careful with

- Say "most approaches train a detector", not "nobody uses depth".
- "Works at any scale" means the *method* is scale-free. Big arms bring payload, safety and cost questions we didn't test.
- Mention FANUC / NVIDIA / Rockwell tools only for what we actually used.
