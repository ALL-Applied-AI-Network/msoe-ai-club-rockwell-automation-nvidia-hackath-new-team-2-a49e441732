"""Sort by colour: map the table (lab.minds.mapper), read the things off the map (lab.things), then for each part pick it
(lab.grasp on the map's surface: unseen ground counts as blocked, so the fingers never come down where nothing has
been seen), drop it in the box of its colour, and look again to check the spot is empty.

    uv run python -m lab.trial run sort_cubes --mind sort

The order of work per part:
    1. grips from the map around it (jaws square to a boxy part's faces, both finger slots clear down to grip height)
    2. the move checked against the arm model (wrist flips, the table, the arm against itself) before it's sent
    3. pick, carry above the box walls, drop just over the rim
    4. forget what the map knew at the grip spot (we moved what was there), look at it straight down, and re-read:
       anything still there -> try again (at most twice), else next part
Parts already inside a box are left there. A part with no box of its colour stays on the table (said, not hidden).
"""
from __future__ import annotations

import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z
from calibration.kinematics import tcp_uf0, tool_ik, tool_pose, tool_r, wpr_of
from lab import armcheck, grasp, primitives as P, viz
from lab.minds.mapper import MapperMind
from lab.things import Things, _inside
from lab.views import _cam_in_tool, _look_at

LOOK_M = 0.20          # wrist camera this far above a spot to check it
TRIES = 2              # attempts per part
DROP_ABOVE_RIM_MM = 4.0
STILL_VOXELS = 8       # after a pick and one fresh look: this many voxels with any solid evidence at the spot = still there
FRAMES = 4            # table-camera depth frames averaged to spot changes (one is too noisy for a 10 mm part at ~1 m)
CHANGE_MM = 6.0       # averaged depth this much closer than before: something new stands there
CHANGE_PX = 10        # ... over at least this many pixels
GRIP_SHARE = 0.45     # the hooked fingertips close at most this far up the part (its measured top face): below the middle


class SorterMind(MapperMind):
    def reset(self) -> None:
        super().reset()
        self.things = Things()
        self.table_rgb = None
        self.report: list[str] = []
        self.picks = {"tried": 0, "verified": 0, "failed": 0, "unchecked": 0, "no_grip": 0, "no_box": 0}

    def _loop(self, obs):
        self.table_rgb = obs["images"]["table"]  # before the arm wanders into the table camera's view
        self.q_home = np.asarray(obs["joints"][:6], float)
        baseline = [obs["depth"]["table"]]
        for _ in range(FRAMES - 1):              # a few frames of the untouched table, to spot changes at the end
            obs = yield self._still(obs, 2)
            baseline.append(obs["depth"]["table"])
        yield from super()._loop(obs)            # map the table first
        obs = self.obs
        found = self.things.update(self.map, self.table_rgb, self.cams)
        self.boxes = {t.color: t for t in found if t.kind == "box"}
        self._draw_things(found)
        viz.narrate(f"Found {sum(t.kind == 'part' for t in found)} parts and {len(self.boxes)} boxes: "
                    + "; ".join(t.describe() for t in found))
        for tid in self._todo(found):
            obs = yield from self._put_away(tid, obs)
        for round_ in range(2):                  # anything new on the table that shouldn't be (dropped, knocked)?
            obs, spots = yield from self._changes(obs, baseline)
            if not spots:
                self._say("The table matches what I expect: nothing left out.")
                break
            self._say(f"Something changed on the table where I didn't put anything: {len(spots)} spot(s); looking.")
            for xy in spots:
                self.map.forget(xy, 0.02)
                look = self._look_rows(xy, obs)
                if look is None:
                    continue
                obs = yield look
                self._fuse_wrist(obs)
                obs = yield self._still(obs, 3)
                self._fuse_wrist(obs)
            found = self.things.update(self.map, self.table_rgb, self.cams)
            self._draw_things(found)
            for tid in self._todo(found):
                obs = yield from self._put_away(tid, obs)
        self.obs = obs

    def _todo(self, found) -> list[int]:
        """Parts on the table (not already in a box), nearest the boxes first."""
        parts = [t for t in found if t.kind == "part" and not any(_inside(t.xy, b, margin_mm=3) for b in self.boxes.values())]
        return [t.id for t in sorted(parts, key=lambda t: t.xy[1])]

    def _put_away(self, tid, obs):
        """Pick thing tid, drop it in the box of its colour, check the spot; try again if it's still there."""
        for attempt in range(TRIES):
            t = next((x for x in self.things.known if x.id == tid), None)
            if t is None:
                break  # gone
            box = self.boxes.get(t.color)
            if box is None:
                self._say(f"{t.name}: no {t.color} box, leaving it on the table.")
                self.picks["no_box"] += 1
                break
            approach, rows, g = self._pick_rows(t, box, obs)
            if rows is None:
                self._say(f"{t.name}: no clear grip from what I've seen; leaving it for now.")
                self.picks["no_grip"] += 1
                break
            self.picks["tried"] += 1
            viz.narrate(f"Pick {t.name} (try {attempt + 1}): grip at ({g.x * 1000:.0f}, {g.y * 1000:.0f}) mm, "
                        f"{g.height_mm:.1f} mm up, jaws at {g.yaw:.0f} deg; into the {box.color} box.")
            obs = yield approach
            obs = yield rows
            # we moved what was at the grip spot (and put something in the box): what we knew there is stale
            self.map.forget((g.x, g.y), 0.012)
            look = self._look_rows((g.x, g.y), obs)
            if look is None:
                self.picks["unchecked"] += 1
                self._say(f"{t.name}: picked, but I can't get a look at the spot to check.")
                break
            obs = yield look
            self._fuse_wrist(obs)
            if self._evidence_at((g.x, g.y)) < STILL_VOXELS:
                self.picks["verified"] += 1
                self._say(f"{t.name}: gone from the table, into the {box.color} box.")
                self._draw_things(self.things.update(self.map, self.table_rgb, self.cams))
                break
            self.picks["failed"] += 1
            self._say(f"{t.name}: still there after the pick (try {attempt + 1}); looking again to re-measure it.")
            obs = yield self._still(obs, 3)  # one fresh look only leans solid: a second makes it a thing again
            self._fuse_wrist(obs)
            found = self.things.update(self.map, self.table_rgb, self.cams)
            self._draw_things(found)
            still = [x for x in found if x.kind == "part" and np.linalg.norm(x.xy - [g.x, g.y]) < 0.012]
            if not still:
                break
            tid = still[0].id
        return obs

    def _changes(self, obs, baseline):
        """Park the arm where it was for the first pictures (so it hides the same pixels), average a few table-camera
        depth frames, and find where something now stands that wasn't there before, outside the boxes."""
        from calibration.geometry import depth_to_base
        from scipy import ndimage
        q_now = np.asarray(obs["joints"][:6], float)
        T_now, T_home = tool_pose(q_now), tool_pose(self.q_home)
        park = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), self.q_home, T_home[:3, 3], wpr_of(T_home[:3, :3]))
        if park is not None:
            obs = yield park
        frames = [obs["depth"]["table"]]
        for _ in range(FRAMES - 1):
            obs = yield self._still(obs, 2)
            frames.append(obs["depth"]["table"])
        mean = lambda fs: np.where(np.all(np.stack(fs) > 0, 0), np.stack(fs).astype(float).mean(0), 0)
        before, now = mean(baseline), mean(frames)
        closer = (before > 0) & (now > 0) & (before - now > CHANGE_MM)
        closer = ndimage.binary_opening(closer, iterations=1)
        lab, n = ndimage.label(closer)
        spots = []
        cam = self.cams["table_depth"]
        for c in range(1, n + 1):
            m = lab == c
            if m.sum() < CHANGE_PX:
                continue
            d = np.where(m, now, 0).astype(np.uint16)
            pts = depth_to_base(cam, d)
            xy = np.median(pts[:, :2], axis=0)
            if (np.percentile(pts[:, 2], 90) - SURFACE_Z) * 1000 < 4 or any(_inside(xy, b, margin_mm=6) for b in self.boxes.values()):
                continue
            if not P.reachable(*xy, margin_mm=10):
                continue  # out of the arm's reach: nothing to do about it
            spots.append(xy)
        viz.narrate(f"Table camera, before vs now: {n} changed patches, {len(spots)} outside the boxes standing up.", level="DEBUG")
        return obs, spots

    def _still(self, obs, n: int):
        """Keep still for n rows (a fresh picture), in joint space: holding a tcp pose would snap the tool upright."""
        return P.JointRows([[*np.asarray(obs["joints"][:6], float).tolist(), 0.0]] * n)

    # ------------------------------------------------------------ moves

    def _pick_rows(self, t, box, obs):
        """The best grip on t whose whole move passes the arm check: (a joint-space move to straight above the grip,
        fingers pointing down; the pick from there: down, close, lift, carry high, drop in the box; the grip).
        The first part is in joint space because the arm may be anywhere (mapping ends on a tilted look): snapping
        to fingers-down in one row is a wrist flip."""
        scene, target = self._surface(t)
        # the planner judges "the middle" from all the part's points, which noise makes taller; the top face doesn't lie
        grips, _ = grasp.find(scene, target, max_height_mm=GRIP_SHARE * t.top_mm)
        top = box.top_mm
        q_now = np.asarray(obs["joints"][:6], float)
        T_now = tool_pose(q_now)
        for g in grips:
            safe = max(P.SAFE_MM, top + 15.0 + g.height_mm)  # the part hangs g.height_mm below the fingertips
            over = np.array([g.x, g.y, SURFACE_Z + safe / 1000])
            wpr = np.array([-180.0, 0.0, tool_r(g.yaw)])
            try:
                q_over = tool_ik(over, wpr, q_now)
            except ValueError:
                continue
            approach = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q_over, over, wpr, height_mm=safe, settle=0)
            if approach is None or not armcheck.check_joints(np.asarray(approach)[:, :6], 0.0).ok:
                continue
            rows, _ = P.pick_and_place(tcp_uf0(q_over), (g.x, g.y), g.height_mm, g.yaw, tuple(box.xy),
                                       drop_mm=top + DROP_ABOVE_RIM_MM, safe_mm=safe)
            rep = armcheck.check(rows, q_over)
            if rep.ok:
                self._draw_grip(g)
                return approach, rows, g
            viz.narrate(f"(grip at {g.yaw:.0f} deg skipped: {rep.problem})", level="DEBUG")
        return None, None, None

    def _evidence_at(self, xy, radius: float = 0.008) -> int:
        """Voxels above the table near xy with any solid evidence (one fresh look can't confirm, but it does hit)."""
        m = self.map
        c = ((np.asarray(xy) - m.origin[:2]) / m.v).astype(int)
        r = int(np.ceil(radius / m.v))
        sl = (slice(max(0, c[0] - r), c[0] + r + 1), slice(max(0, c[1] - r), c[1] + r + 1))
        return int(((m.L[sl] > 0.3) & m.clear_of_table()[sl]).sum())

    def _surface(self, t):
        """Points for the grip search: everything seen around t (the table where it was found, solid evidence above
        it), each voxel split into four so the 1 mm grid has no false holes; and t's own points."""
        m = self.map
        near = lambda p: p[np.max(np.abs(p[:, :2] - t.xy), axis=1) < grasp.CROP]
        table = near(m.table_points(step=1, sure=False))
        above = near(np.vstack([m.solid_points(), m.leaning_points()]))
        q = m.v / 4
        split = lambda p: np.vstack([p + [dx, dy, 0] for dx in (-q, q) for dy in (-q, q)])
        return split(np.vstack([table, above])), split(t.pts)

    def _look_rows(self, spot, obs):
        """A joint-space move that puts the wrist camera straight above spot, LOOK_M up, path checked; None if none."""
        cam = self.cams["wrist_depth"]
        T_ct = _cam_in_tool(cam)
        q_now = np.asarray(obs["joints"][:6], float)
        T_now = tool_pose(q_now)
        target = np.array([spot[0], spot[1], SURFACE_Z])
        for roll in np.radians([0, 90, 180, 270]):
            Tt = _look_at(target, np.array([0.0, 0.0, -1.0]), LOOK_M, roll) @ np.linalg.inv(T_ct)
            tip, wpr = Tt[:3, 3], wpr_of(Tt[:3, :3])
            if not P.reachable(*tip[:2], margin_mm=20):
                continue
            try:
                q = tool_ik(tip, wpr, q_now)
            except ValueError:
                continue
            rows = P.joint_move(q_now, T_now[:3, 3], wpr_of(T_now[:3, :3]), q, tip, wpr)
            if rows is not None and armcheck.check_joints(np.asarray(rows)[:, :6], 0.0).ok:
                return rows
        return None

    # ------------------------------------------------------------ showing and telling

    def _say(self, line: str) -> None:
        self.report.append(line)
        viz.narrate(line)

    def _draw_things(self, found) -> None:
        rr.log("mind/things", rr.Clear(recursive=True))
        for t in found:
            h = max(t.top_mm, 2.0) / 1000
            rr.log(f"mind/things/{t.id}", rr.Boxes3D(centers=[[*t.xy, SURFACE_Z + h / 2]],
                                                     half_sizes=[[t.size_mm[0] / 2000, t.size_mm[1] / 2000, h / 2]],
                                                     rotation_axis_angles=[rr.RotationAxisAngle([0, 0, 1], degrees=t.angle)],
                                                     colors=[t.rgb], labels=[t.name]))

    def _draw_grip(self, g) -> None:
        a = np.radians(g.yaw)
        d = np.array([np.cos(a), np.sin(a), 0]) * grasp.GAP / 2000
        c = np.array([g.x, g.y, SURFACE_Z + g.height_mm / 1000])
        rr.log("mind/plan/grip", rr.LineStrips3D([[c - d, c + d]], colors=[[60, 220, 90]], radii=0.0015))

    def status(self) -> str:
        p = self.picks
        return (f"Sorting: {p['verified']} parts put away (checked), {p['failed']} picks failed, {p['unchecked']} unchecked, "
                f"{p['no_grip']} with no clear grip, {p['no_box']} with no box of their colour.")
