"""The ventral stream: an LLM that looks at the table camera's picture, asks SAM 3.1 to outline what it recognises, reads
where its outlines and the depth stream's 3D segments agree, and acts on the disagreements: it names things, asks the
dorsal stream to break up a segment it thinks is several things (or to look closer), and sets the sort rule.

It thinks when there's news (a look finished, a request done, the person said something), not in a spin: between
two pieces of news the table hasn't changed, so a new thought would see the same picture.

It never gives a coordinate, a grip or a path: requests name a thing ("break up #4, keep the pieces close") and the
dorsal stream turns that into a pick or a rake, or refuses and says why.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time

import cv2
import numpy as np
import rerun as rr

from lab import agree, sam, viz, vlm
from lab.streams import Board, Rule
from lab.things import _inside


COLOURS = ("red", "blue", "yellow", "green", "orange", "purple", "white", "black", "gray", "pink", "brown")
HUES = ("red", "blue", "yellow", "green", "orange", "purple", "pink")


def _in_box(t, box) -> bool:
    return _inside(t.xy, box, margin_mm=3)

MODEL = os.getenv("VENTRAL_MODEL") or os.getenv("CHAT_MODEL") or "openrouter/openai/gpt-6-luna"
DORSAL_NEWS = ("started", "map_progress", "map_done", "skill_done", "request_done", "request_refused", "idle", "done",
               "user_rule", "user_fact", "gave_up")

SYSTEM = """You are the ventral stream of a robot at a sorting table: the part that recognises what things are. A
separate dorsal stream maps the table in 3D from depth and moves the arm. You share a board with it.

The dorsal stream finds "things": touching solid evidence above the table, one 3D segment each, with an id (#4),
size, height, a colour measured from the camera, and whether it's a part or a box (wider than 40 mm). It can't tell
what a thing is, and touching things merge into one segment. You can see the picture, and you have SAM 3.1: give it
a few words ("lego", "screw", "cube") and it outlines every match. Outlines are compared with the 3D segments:
  agree: one outline on one segment          split: several outlines on one segment (it's several things)
  merged: one outline over two segments       no_mask: a segment nothing outlined
  no_segment: an outline with no 3D segment under it (too flat for depth so far, or not there)
If SAM won't outline a part you can plainly see as one object after two or three prompts, vouch for it.
Segments with no outline that you can't see as objects in the picture (flat, pale, on the table) are table texture:
paper seams and wrinkles. Ignore them; don't ask to look closer at them.
SAM is literal about words: try a few short, plain, common words, singular; one prompt per kind of object. "lego"
finds bricks where "lego brick" may not. Re-run segment after things move (old outlines go stale).

Your job, in order of value:
1. Know what's on the table: segment, name each thing (name), and get the streams to agree.
2. Where they disagree, ask the dorsal stream to fix it: request break_up for a split segment (say how many things you
   see in it, and "keep the pieces close" unless they need room); look_closer for no_segment or no_mask. Requests skip
   the dorsal stream's own queue. The dorsal stream decides how (a pick for two or three, a rake for a pile).
3. Set the sort rule as soon as you know enough to say where each kind of thing goes (set_sort_rule). Start general
   ("blocks together, bolts and pins together") and refine it as you learn; the dorsal stream only moves things that
   have agreed segments (one 3D segment, one outline, from your latest segment call) and a destination in the rule:
   after things move, call segment again so the moved and new pieces can agree. Destinations: a box by its id from the board's box
   list ("box #7"), or "table" (a tidy group on the table, next to the rule's other table groups).
   THE BOXES DECIDE THE NUMBER OF GROUPS: make exactly as many groups as there are boxes (fewer only if there are
   fewer parts than boxes), one group per box, and give every part on the table a box. More kinds than boxes: merge
   the most alike kinds (say which and why). Fewer kinds than boxes: split a kind by colour or size. Decide the
   groups once, for the whole table; what's already in a box is that box's group: send more of the same kind there. "table" only when
   there are no boxes, or the person asks for it. The person's words decide what the groups are (colour, kind, a
   named exception); their rules win. Ids change when things move: put only current part ids in the rule.
4. Then wait for news (wait). Don't re-check what hasn't changed.

Colours on the board are measured by the dorsal stream from one camera (a small part can read "white" or the wrong
colour): check them in the picture, and when colour matters, segment with colour words ("red cube", "blue cube") so
each outline carries its colour, then name the parts by what you see. Never send a part to a cup of another colour
just because its board colour is odd.
Be brief: one or two sentences of thought per turn, then tools. Never invent ids; use the ones on the board."""

TOOLS = [
    {"type": "function", "function": {"name": "segment", "description": "Ask SAM 3.1 to outline every match of each prompt in the table camera's latest picture, then compare the outlines with the 3D segments. Returns the agreement, per thing.",
     "parameters": {"type": "object", "properties": {"prompts": {"type": "array", "items": {"type": "string"}, "description": "short plain words, one per kind of object, e.g. [\"lego\", \"screw\", \"cube\"]"}},
                    "required": ["prompts"]}}},
    {"type": "function", "function": {"name": "name", "description": "Say what a thing is (sticks to its id).",
     "parameters": {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string", "description": "e.g. 'red cube', 'screw', 'yellow lego 1x2'"}},
                    "required": ["id", "name"]}}},
    {"type": "function", "function": {"name": "vouch", "description": "When SAM won't outline a thing you can plainly see in the picture as ONE object (and the board has it as one confirmed part): say so, and the dorsal stream treats it as agreed.",
     "parameters": {"type": "object", "properties": {"id": {"type": "integer"}, "why": {"type": "string"}}, "required": ["id", "why"]}}},
    {"type": "function", "function": {"name": "request", "description": "Ask the dorsal stream to do something about one thing; it goes ahead of its own work.",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["break_up", "look_closer", "put_away"]},
         "id": {"type": "integer", "description": "the thing's id (leave out when giving mask)"},
         "mask": {"type": "string", "description": "look_closer only: an outline with no 3D segment, e.g. 'm3'"},
         "expect": {"type": "integer", "description": "break_up: how many things you think are in it"},
         "note": {"type": "string", "description": "e.g. 'keep the pieces close'"},
         "to": {"type": "string", "description": "put_away: 'box #2' or 'table'"}},
                    "required": ["action"]}}},
    {"type": "function", "function": {"name": "set_sort_rule", "description": "Where each group of things goes. Replaces the rule; call again to refine it.",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "description": "the rule in words"},
         "groups": {"type": "array", "items": {"type": "object", "properties": {
             "name": {"type": "string"}, "ids": {"type": "array", "items": {"type": "integer"}},
             "to": {"type": "string", "description": "'box #N' or 'table'"}}, "required": ["name", "ids", "to"]}}},
                    "required": ["text", "groups"]}}},
    {"type": "function", "function": {"name": "wait", "description": "Done thinking for now: sleep until the dorsal stream has news (a look, a request done, a pick done, the person).",
     "parameters": {"type": "object", "properties": {"why": {"type": "string"}}, "required": ["why"]}}},
]


FOCUS = """
THIS JOB WORKS DIFFERENTLY: the dorsal stream does NOT map the whole table. You point it.
1. First, call segment on the table picture with one short prompt per kind of thing you see, always including the
   containers ("cube", "cup", ...). Every outline becomes a region; the arm takes a close look only there.
2. In each close look, SAM outlines things in the wrist picture and depth is pooled inside each outline: the board then
   lists MEASURED objects (id, size, height, colour, the prompt that found them). A cup or box is anything over 40 mm.
   Measured objects are already one thing each: no agreement step, no break-ups.
3. If something you can see has no measured object, request look_closer at its outline (mask "mN") or near an object.
4. Set the rule over measured part ids and measured cup ids ("box #N"). The arm looks again right before each pick.
If the person told you what's on the table, segment with their words first (and name parts by them): they know.
Every cup is measured from the table camera's depth at the start. For a plain colour sort the arm sets the rule itself
(source "dorsal") for the colours it can tell apart; leave that rule alone unless it is wrong for what you see, and
give the other parts (red and orange especially: the cameras can't tell those apart) their cups."""


def say(text: str, level: str = "INFO") -> None:
    viz.sync_time()
    rr.log("ventral", rr.TextLog(text, level=level))
    viz._log("ventral", text, level)


class Ventral(threading.Thread):
    def __init__(self, board: Board, cams: dict, instruction: str = "", model: str = MODEL, max_turns: int = 60):
        super().__init__(daemon=True, name="ventral")
        self.board, self.cams, self.instruction, self.max_turns = board, cams, instruction, max_turns
        self.client, self.model = vlm._client(vlm.model_id(model), 90)
        self.history: list[dict] = []
        self.table_px = None
        self.turns = 0
        self.rec = None  # rerun recording to log into (the thread needs it passed in)

    def run(self) -> None:
        if self.rec is not None:
            rr.set_thread_local_data_recording(self.rec)
        seq = 0
        while not self.board.stopped and self.turns < self.max_turns:
            news = self.board.wait(seq, DORSAL_NEWS, timeout=120)
            if self.board.stopped:
                break
            if not news:
                continue
            seq = news[-1][0]
            if any(k == "done" for _, _, k, _ in news):
                break
            try:
                self._turn(news)
            except Exception as e:  # a bad answer or a network hiccup must not end the stream
                say(f"(thinking failed: {type(e).__name__}: {e})", "WARN")
                time.sleep(3)
        say("(ventral stream stopped)", "DEBUG")

    # ------------------------------------------------------------ one turn: news in, tools, until it waits

    def _turn(self, news) -> None:
        self.turns += 1
        b = self.board
        lines = [f"- {k}: {t}" for _, _, k, t in news[-8:]]
        state = self._state()
        content = [{"type": "text", "text": "News:\n" + "\n".join(lines) + "\n\n" + state}]
        img = self._picture()
        if img is not None:
            content.append({"type": "image_url", "image_url": {"url": img}})
        self._strip_old_images()
        self.history.append({"role": "user", "content": content})
        for _ in range(8):
            sys_text = SYSTEM + (FOCUS if b.mode == "focus" else "")
            msgs = [{"role": "system", "content": sys_text + (f"\n\nThe person asked: \"{self.instruction}\"" if self.instruction else "")}] + self.history[-24:]
            while msgs[1]["role"] == "tool":  # never start mid tool call
                msgs.pop(1)
            t_ask = time.time()
            r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=TOOLS, max_tokens=3000,
                                                    extra_body={"reasoning": {"effort": "low"}})
            m = r.choices[0].message
            calls = m.tool_calls or []
            from lab import record
            first, self._logged_system = not getattr(self, "_logged_system", False), True
            record.write("ventral_llm.jsonl", {"system": msgs[0]["content"] if first else "(as before)",
                                               "last": record.text_only(msgs[-2:]), "answer": m.content or "",
                                               "seconds": round(time.time() - t_ask, 1),
                                               "tools": [{"name": c.function.name, "args": c.function.arguments} for c in calls]})
            self.history.append({"role": "assistant", "content": m.content or "",
                                 **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
            if m.content and m.content.strip():
                say(m.content.strip())
            if not calls:
                return
            waiting = False
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    out = self._do(c.function.name, args)
                except Exception as e:
                    out = f"error: {type(e).__name__}: {e}"
                    say(f"({c.function.name} failed: {out})", "WARN")
                if c.function.name == "wait":
                    waiting = True
                self.history.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if waiting:
                return
        b.post("note", "ventral: ran out of steps this turn")

    def _do(self, name: str, a: dict) -> str:
        b = self.board
        if name == "segment":
            return self._segment([p for p in a.get("prompts", []) if p][:6])
        if name == "name":
            b.name(int(a["id"]), a["name"])
            say(f"#{a['id']} is a {a['name']}.", "DEBUG")
            return "named"
        if name == "vouch":
            tid = int(a["id"])
            with b.cond:
                if tid not in b.known_ids:
                    return f"not vouched: #{tid} isn't a confirmed part on the board"
                b.vouched.add(tid)
            say(f"#{tid}: SAM won't outline it, but it's plainly one thing ({a.get('why', '')}).")
            b.post("vouch", f"#{tid}")
            return "vouched: the dorsal stream may move it"
        if name == "request" and a.get("action") == "look_closer":
            with b.cond:
                n = sum(r.action == "look_closer" for r in b.requests)
            if n >= 4:
                return ("not queued: 4 close looks already this job. Unexplained segments with no outline are likely "
                        "table texture (paper seams, wrinkles): ignore them and sort the parts you can see.")
        if name == "request":
            if a.get("mask") and str(a["mask"]).lstrip("m").isdigit():
                j = int(str(a["mask"]).lstrip("m"))
                with b.cond:
                    xy = b.regions[j]["xy"] if b.mode == "focus" else b.masks[j]["xy"]
                r = b.request(a["action"], None, a.get("note", ""), xy=xy)
                say(f"→ dorsal: {a['action'].replace('_', ' ')} where outline {a['mask']} is")
                return f"queued as request {r.id}, ahead of the dorsal stream's own work"
            try:
                expect = int(a["expect"]) if a.get("expect") not in (None, "") else None
            except (TypeError, ValueError):
                expect = None
            r = b.request(a["action"], int(a["id"]), a.get("note", ""), expect, a.get("to"))
            say(f"→ dorsal: {a['action'].replace('_', ' ')} #{a['id']}" + (f", {a['note']}" if a.get("note") else "")
                + (f" (I see {a['expect']})" if a.get("expect") else ""))
            return f"queued as request {r.id}, ahead of the dorsal stream's own work"
        if name == "set_sort_rule":
            groups = [{"name": g["name"], "ids": [int(i) for i in g.get("ids", [])], "to": g["to"].strip()} for g in a.get("groups", [])]
            with b.cond:
                boxes = {t.id for t in b.things.values() if t.kind == "box"}
                parts = {t.id for t in b.things.values() if t.kind != "box"}
            bad = [g["to"] for g in groups if g["to"].lower() != "table"
                   and not (g["to"].lower().startswith("box #") and g["to"][5:].strip().isdigit() and int(g["to"][5:]) in boxes)]
            if bad:
                return (f"not set: {', '.join(bad)} isn't a destination. Boxes on the board: "
                        + (", ".join(f"box #{i}" for i in sorted(boxes)) or "none") + "; or 'table'.")
            unknown = sorted({i for g in groups for i in g["ids"]} - parts)
            for g in groups:  # ids change as the map updates: keep the rule for the parts that still exist
                g["ids"] = [i for i in g["ids"] if i in parts]
            used = {g["to"].lower() for g in groups if g["to"].lower() != "table"}
            with b.cond:
                used |= {f"box #{i}" for i, c in b.contents.items() if c}  # a box with things in it is a group already
            with b.cond:
                on_table = {t.id for t in b.things.values() if t.kind != "box" and t.id in b.known_ids
                            and not any(_in_box(t, x) for x in b.things.values() if x.kind == "box")}
                user = list(b.user_rules)
            missing = sorted(on_table - {i for g in groups for i in g["ids"]})
            problems = []
            # a partial rule is fine while parts are still unknown; a complete one must use every box
            warn = ""
            if boxes and not missing and len(used) < min(len(boxes), len(on_table)) and not user:
                # a nudge, not a refusal: forcing a split made it carve one kind in two by size to fill a box
                warn = (f" Note: {len(boxes)} boxes and the rule fills {len(used)}. If there's a natural way to make "
                        f"{len(boxes)} groups (a second attribute, or a kind you merged), use it; if not, say why.")
            if boxes and any(g["to"].lower() == "table" for g in groups) and not user:
                problems.append("there are boxes: every group goes in one")
            with b.cond:
                box_colour = {f"box #{t.id}": t.color for t in b.things.values() if t.kind == "box"}
            for g in groups:  # "red cubes" -> the blue cup is a slip of the id, not a choice
                said = [c for c in COLOURS if c in g["name"].lower().split()]
                cup = box_colour.get(g["to"].lower())
                # only for coloured cups, and only when a cup of the group's colour exists: plain white, gray or
                # black boxes say nothing about what goes in them
                if (len(said) == 1 and cup in HUES and cup != said[0] and said[0] in box_colour.values()
                        and not user):
                    problems.append(f"group '{g['name']}' goes to {g['to']}, which is the {cup} cup: check the box ids")
            with b.cond:
                cup_hues = {t.color for t in b.things.values() if t.kind == "box" and t.color in HUES}
                part_hues = {t.color for t in b.things.values() if t.kind != "box" and t.color in HUES}
            if (len(cup_hues) >= 2 and len(used) < len(boxes) and len(part_hues) >= 2 and part_hues <= cup_hues
                    and not user):  # seed 2: "all cubes in the only cup" with three cups matching three cube colours
                problems.append(f"the cups are {', '.join(sorted(cup_hues))} and the parts are {', '.join(sorted(part_hues))}: "
                                f"every cup is on the board above; use each one")
            if problems:
                return "not set: " + "; ".join(problems)
            if missing:
                note = (f" Parts with no destination yet: {', '.join('#' + str(i) for i in missing)} "
                        f"(fine while they're unidentified or disagreeing; add them when you know them).")
            else:
                note = ""
            b.set_rule(Rule(a["text"], groups))
            say(f"Sort rule: {a['text']} — " + "; ".join(f"{g['name']} {['#' + str(i) for i in g['ids']]} → {g['to']}" for g in groups))
            gone = (f" Dropped ids no longer on the board: {', '.join('#' + str(i) for i in unknown)}." if unknown else "")
            return "rule set." + gone + note + warn
        if name == "wait":
            say(f"(waiting: {a.get('why', '')})", "DEBUG")
            return "ok"
        return f"no such tool {name}"

    def _segment(self, prompts: list[str]) -> str:
        b = self.board
        with b.cond:
            img, arm, things = b.image, b.arm, list(b.things.values())
        if img is None:
            return "no picture yet"
        if not sam.configured():
            return "SAM isn't configured (no SAM3_API_KEY)"
        if b.real and os.getenv("SAM_REAL") != "1":
            return "SAM is off on the real cell (its pictures don't go to Meta unless the team turns it on): name and vouch by eye"
        cam = self.cams["table"]
        wh = (img.shape[1], img.shape[0])
        if self.table_px is None or self.table_px_wh != wh:
            self.table_px, self.table_px_wh = agree.TablePixels(cam, wh), wh
        if b.mode == "focus":
            return self._glance(img, prompts)
        crop, scale = agree.focus(things, cam, wh)
        masks, failed = [], []
        try:
            masks = sam.segment_many(img, list(prompts), crop=crop, scale=scale)
        except Exception as e:
            failed.append(f"{', '.join(prompts)}: {e}")
        ag = agree.compare(things, masks, cam, wh, self.table_px, arm=arm)
        with b.cond:
            b.masks, b.agreement = ag.masks, ag
        viz.sync_time()
        agree.draw(ag, things, cam)
        counts = ag.case_counts()
        say(f"SAM ({', '.join(prompts)}): {len(ag.masks)} outlines; " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()), "DEBUG")
        return (f"{len(ag.masks)} outlines from {', '.join(prompts)}" + (f" (failed: {'; '.join(failed)})" if failed else "")
                + "\n" + ag.text({t.id: t for t in things}))

    def _glance(self, img, prompts: list[str]) -> str:
        """Focus mode: outline on the table picture -> regions on the table for the arm's close looks."""
        from lab import focus
        b = self.board
        x0y0 = self._work_crop(img)
        masks, failed = [], []
        try:
            masks = sam.segment_many(img, list(prompts), crop=x0y0, scale=1.5)
        except Exception as e:
            failed.append(f"{', '.join(prompts)}: {e}")
        regs = focus.regions(masks, self.table_px)
        with b.cond:
            b.masks, b.prompts = masks, list(prompts)
            b.regions = regs if b.regions is None else b.regions + regs
        self._draw_table_masks(masks, img)
        say(f"SAM on the table picture ({', '.join(prompts)}): {len(regs)} outlines to look at closely.", "DEBUG")
        return (f"{len(regs)} outlines" + (f" (failed: {'; '.join(failed)})" if failed else "") + ":\n"
                + "\n".join(f"m{j} '{r['prompt']}' near ({r['xy'][0] * 1000:.0f}, {r['xy'][1] * 1000:.0f}) mm"
                             for j, r in enumerate(regs)))

    def _work_crop(self, img):
        from calibration.geometry import SURFACE_Z, project
        cam = self.cams["table"]
        W, H = img.shape[1], img.shape[0]
        c = np.array([[x, y, SURFACE_Z] for x in (0.15, 0.65) for y in (-0.45, 0.45)])
        uv = project(cam, c) * [W / cam["size"][0], H / cam["size"][1]]
        x0, y0 = np.clip(uv.min(0) - 30, 0, None).astype(int)
        x1, y1 = np.minimum(uv.max(0) + 30, [W, H]).astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    def _draw_table_masks(self, masks, img) -> None:
        f = viz.view_scale(img.shape[1])
        w, h = int(round(img.shape[1] * f)), int(round(img.shape[0] * f))
        seg = np.zeros((h, w), np.uint8)
        for j, m in enumerate(masks[:250]):
            seg[cv2.resize(m["mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)] = j + 1
        viz.sync_time()
        rr.log("cams/table/rgb/sam", rr.AnnotationContext([rr.AnnotationInfo(id=0, label="", color=(0, 0, 0, 0))]
                                                          + [rr.AnnotationInfo(id=j + 1, label=f"m{j} {m['prompt']}", color=(90, 140, 255, 150))
                                                             for j, m in enumerate(masks[:250])]))
        rr.log("cams/table/rgb/sam", rr.SegmentationImage(seg, opacity=0.45))

    # ------------------------------------------------------------ what it's shown

    def _state(self) -> str:
        b = self.board
        with b.cond:
            things = list(b.things.values())
            ag = b.agreement
            rule, user, queue = b.rule, list(b.user_rules), b.queue_text()
            known = set(b.known_ids)
        cases = {r.seg: r.case for r in ag.rows} if ag else {}
        rows = []
        boxes = [t for t in things if t.kind == "box"]
        things = [t for t in things if t.kind != "box"]
        for t in sorted(things, key=lambda t: t.id):
            rows.append(f"#{t.id} {t.kind}{'' if t.id in known else ' (maybe: not confirmed yet)'}, {t.color}, "
                        f"{t.size_mm[0]:.0f} x {t.size_mm[1]:.0f} mm, {t.top_mm:.0f} mm tall, at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) mm"
                        + (f", named '{t.label}'" if getattr(t, "label", None) else "")
                        + ("" if b.mode == "focus" else (f", streams: {cases[t.id]}" if t.id in cases else ", streams: not compared since it last changed")))
        with b.cond:
            contents = {i: list(c) for i, c in b.contents.items()}
        out = ("Boxes (the only boxes; use exactly these ids): "
               + (", ".join(f"box #{t.id} ({t.color}, at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) mm)"
                            + (f" already holds: {', '.join(contents[t.id])}" if contents.get(t.id) else " empty")
                            for t in boxes) or self._no_boxes())
               + "\nParts on the board:\n" + ("\n".join(rows) or "(none yet: still mapping)"))
        with b.cond:
            unknown = list(b.unknown)
        if unknown:
            out += ("\nOnly depth sees these (no SAM outline agrees, so they aren't parts and won't be picked; the arm "
                    "steers around them). If the picture shows what they are, segment with a word that fits:\n"
                    + "\n".join(unknown))
        out += f"\nSort rule ({rule.source if rule else ''}): {rule.text if rule else 'none yet'}"
        if user:
            out += "\nThe person's rules (these win): " + "; ".join(user)
        with b.cond:
            facts = [f["text"] for f in b.facts]
        if facts:
            out += "\nWhat the person told you is on the table: " + "; ".join(facts)
        out += f"\nRequests in the dorsal queue: {queue}"
        out += "\nThe picture: the table camera, cropped to the things (x away from the robot, y to its left)."
        return out

    def _no_boxes(self) -> str:
        b = self.board
        with b.cond:
            surveyed = b.mode == "focus" and any(k == "map_done" for _, _, k, _ in b.events)
        if not surveyed:
            return "none found yet"
        return ("NONE: the close looks measured no cup or box. If you see one in the picture, request look_closer at "
                "its outline (mask 'mN'); otherwise there is none for the arm, so set the rule with 'table' piles now")

    def _picture(self) -> str | None:
        b = self.board
        with b.cond:
            img, things = b.image, list(b.things.values())
        if img is None:
            return None
        wh = (img.shape[1], img.shape[0])
        (x0, y0, x1, y1), _ = agree.focus(things, self.cams["table"], wh, margin_m=0.06, target_px=1024)
        crop = img[y0:y1, x0:x1]
        s = min(4.0, 1024 / max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def _strip_old_images(self) -> None:
        """Only the newest picture stays in the conversation: old ones show a table that has changed."""
        for m in self.history:
            if m["role"] == "user" and isinstance(m["content"], list):
                m["content"] = [c for c in m["content"] if c.get("type") != "image_url"]
