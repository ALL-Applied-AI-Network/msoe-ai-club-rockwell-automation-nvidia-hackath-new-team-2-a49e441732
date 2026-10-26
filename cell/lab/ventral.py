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

MODEL = os.getenv("VENTRAL_MODEL") or os.getenv("CHAT_MODEL") or "openrouter/openai/gpt-6-luna"
DORSAL_NEWS = ("started", "map_progress", "map_done", "skill_done", "request_done", "request_refused", "idle", "done",
               "user_rule", "gave_up")

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
   list ("box #7"), or "table" (a tidy group on the table, next to the rule's other table groups). When there are
   boxes, use them: one kind per box; "table" only for kinds left over when the boxes run out. Follow what the person
   asked; their rules win. Ids change when things move: put only current part ids in the rule.
4. Then wait for news (wait). Don't re-check what hasn't changed.

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


def say(text: str, level: str = "INFO") -> None:
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
            msgs = [{"role": "system", "content": SYSTEM + (f"\n\nThe person asked: \"{self.instruction}\"" if self.instruction else "")}] + self.history[-24:]
            while msgs[1]["role"] == "tool":  # never start mid tool call
                msgs.pop(1)
            r = self.client.chat.completions.create(model=self.model, messages=msgs, tools=TOOLS, max_tokens=3000,
                                                    extra_body={"reasoning": {"effort": "low"}})
            m = r.choices[0].message
            calls = m.tool_calls or []
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
        if name == "request":
            if a.get("mask"):
                j = int(str(a["mask"]).lstrip("m"))
                with b.cond:
                    xy = b.masks[j]["xy"]
                r = b.request(a["action"], None, a.get("note", ""), xy=xy)
                say(f"→ dorsal: {a['action'].replace('_', ' ')} where outline {a['mask']} is")
                return f"queued as request {r.id}, ahead of the dorsal stream's own work"
            r = b.request(a["action"], int(a["id"]), a.get("note", ""), a.get("expect"), a.get("to"))
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
            if unknown:
                return f"not set: no part with id {', '.join('#' + str(i) for i in unknown)} on the board (boxes can't be sorted)"
            b.set_rule(Rule(a["text"], groups))
            say(f"Sort rule: {a['text']} — " + "; ".join(f"{g['name']} {['#' + str(i) for i in g['ids']]} → {g['to']}" for g in groups))
            return "rule set"
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
        crop, scale = agree.focus(things, cam, wh)
        masks, failed = [], []
        for p in prompts:
            try:
                masks += sam.segment(img, p, crop=crop, scale=scale)
            except Exception as e:
                failed.append(f"{p}: {e}")
        ag = agree.compare(things, masks, cam, wh, self.table_px, arm=arm)
        with b.cond:
            b.masks, b.agreement = ag.masks, ag
        agree.draw(ag, things, cam)
        counts = ag.case_counts()
        say(f"SAM ({', '.join(prompts)}): {len(ag.masks)} outlines; " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()), "DEBUG")
        return (f"{len(ag.masks)} outlines from {', '.join(prompts)}" + (f" (failed: {'; '.join(failed)})" if failed else "")
                + "\n" + ag.text({t.id: t for t in things}))

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
                        + (f", streams: {cases[t.id]}" if t.id in cases else ", streams: not compared since it last changed"))
        out = ("Boxes (the only boxes; use exactly these ids): "
               + (", ".join(f"box #{t.id} ({t.color}, at ({t.xy[0] * 1000:.0f}, {t.xy[1] * 1000:.0f}) mm)" for t in boxes) or "none found yet")
               + "\nParts on the board:\n" + ("\n".join(rows) or "(none yet: still mapping)"))
        out += f"\nSort rule: {rule.text if rule else 'none yet'}"
        if user:
            out += "\nThe person's rules (these win): " + "; ".join(user)
        out += f"\nRequests in the dorsal queue: {queue}"
        out += "\nThe picture: the table camera, cropped to the things (x away from the robot, y to its left)."
        return out

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
