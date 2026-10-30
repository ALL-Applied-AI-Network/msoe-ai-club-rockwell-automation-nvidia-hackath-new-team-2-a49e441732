"""The opening scan: how high it's safe to fly, then sweep the whole table, fast.

    pics = [table_depth(client), ...]            # table-camera depth pictures, the arm somewhere else in each
    tallest_mm, info = tallest(pics)              # the tallest thing on the table
    tip = safe_tip_mm(tallest_mm)                 # fingertips this high: CLEAR_MM over it, never under MIN_TIP_MM
    rows = sweep(q, tip, speed)                   # joint rows: lines both ways over REGION, hops between them
    lowest_tip_mm(rows)                           # the check: the lowest the fingertips go on the way

The table camera looks straight down and measures depth, so the tallest thing is a height read off it directly (its
~8 mm noise is covered by the clearance); nothing has to be picked out of a picture. The arm is in its view: its own
pixels are blanked (its shape at those joints), and two pictures with the arm in different places cover for each other. Lines go both ways (an edge along
the motion has no parallax, so both directions are needed), LINE_GAP apart: the wrist camera sees about +-240 x +-135 mm
of table from ~280 mm up, so neighbouring lines overlap.
"""
from __future__ import annotations

import numpy as np

from calibration.geometry import SURFACE_Z, UF0_Z_MM, depth_to_base, load
from calibration.kinematics import DOWN, tool_ik
from lab import primitives as P

REGION = ((0.26, 0.50), (-0.30, 0.08))  # x, y (m): where things are put (the sim's layouts)
CLEAR_MM = 30.0
MIN_TIP_MM = 76.0  # (at 60 a one-cube table gave an empty grid, Oct 30: not understood yet)
FALLBACK_TIP_MM = 120.0  # when the one picture can't be trusted (the arm misread as a thing)
MAX_TALL_MM = 200.0  # taller than this is the arm misread as a thing (nothing on this table is)
LINE_GAP = 0.10
HOP = 150.0  # mm/s between lines


def table_depth(client) -> tuple[np.ndarray, np.ndarray]:
    """One table-camera depth picture (mm, 0 = none), with the arm's own pixels blanked out, and the joints then."""
    o = client.observe(["table"], (64, 36), "jpeg", depth=["table"])
    return blank_arm(o["depth"]["table"], o["joints"]), np.asarray(o["joints"], float)


def blank_arm(depth, joints) -> np.ndarray:
    """A table-camera depth picture with the arm's own pixels (its shape at these joints) set to 0."""
    d = np.asarray(depth).copy()
    pts, uv = depth_to_base(load()["table_depth"], d, with_pixels=True)
    arm = near_arm(pts, joints)
    d[uv[arm, 1], uv[arm, 0]] = 0
    return d


def tallest(pictures, region=REGION) -> tuple[float, dict]:
    """The tallest thing over the region (mm above the table) from table-camera depth pictures taken with the arm in
    different places (table_depth): per pixel the farthest reading wins (the arm, wherever it is in one picture, is
    above what's under it; the things on the table are the same in every picture), then the 99.7th percentile of the
    heights over the region (a few noisy pixels don't count)."""
    d = np.max(np.stack([p[0] for p in pictures]), 0)
    pts = depth_to_base(load()["table_depth"], d, stride=2)
    (x0, x1), (y0, y1) = region
    m = (pts[:, 0] > x0 - 0.05) & (pts[:, 0] < x1 + 0.05) & (pts[:, 1] > y0 - 0.05) & (pts[:, 1] < y1 + 0.05)
    h = (pts[m, 2] - SURFACE_Z) * 1000
    h = h[h < MAX_TALL_MM]  # higher than anything put on this table: the arm (its model misses the motor housings)
    top = float(np.percentile(h, 99.7)) if len(h) else 0.0
    return top, {"pixels": int(m.sum()), "above_5mm": int((h > 5).sum()), "pictures": len(pictures)}


def near_arm(pts: np.ndarray, joints, margin: float = 0.05) -> np.ndarray:
    """Which points lie within `margin` of the arm or gripper at these joints: its capsule model (lab/armcheck.py),
    sampled every centimetre along each link (the rendered arm is a little fatter than the capsules)."""
    from scipy.spatial import cKDTree
    from lab.armcheck import shapes
    near = np.zeros(len(pts), bool)
    tree = cKDTree(pts)
    for _, sp, r, _ in shapes(np.asarray(joints, float)[:6], float(np.asarray(joints)[6]) if len(joints) > 6 else 0.0):
        sp = np.atleast_2d(sp)
        dense = [sp[:1]] + [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.01) + 1))[1:]
                            for a, b in zip(sp[:-1], sp[1:])]
        for c in np.concatenate(dense):
            near[tree.query_ball_point(c, r + margin)] = True
    return near


def safe_tip_mm(tallest_mm: float) -> float:
    return max(MIN_TIP_MM, tallest_mm + CLEAR_MM)


def lines(region=REGION, gap: float = LINE_GAP) -> list:
    """Lines along x at every `gap` of y, then along y at every `gap` of x, each the other way from the last."""
    (x0, x1), (y0, y1) = region
    out = []
    for i, y in enumerate(np.linspace(y0, y1, max(2, int(round((y1 - y0) / gap)) + 1))):
        out.append(((x0, y), (x1, y)) if i % 2 == 0 else ((x1, y), (x0, y)))
    for i, x in enumerate(np.linspace(x0, x1, max(2, int(round((x1 - x0) / gap)) + 1))):
        out.append(((x, y1), (x, y0)) if i % 2 == 0 else ((x, y0), (x, y1)))
    return out


def _tip_m(p):
    return np.array([p[0], p[1], p[2] - UF0_Z_MM]) / 1000


def cam_offset(q, T_fc) -> np.ndarray:
    """Where the wrist camera is relative to the fingertips, in x and y (m), with the fingers straight down."""
    from calibration.geometry import flange_pose
    from calibration.kinematics import tcp_of
    return (flange_pose(q) @ T_fc)[:2, 3] - tcp_of(q)[0][:2]


def for_camera(segs, offset) -> list:
    """Lines for the camera turned into lines for the fingertips: the camera rides `offset` ahead of them."""
    o = np.asarray(offset)
    return [(tuple(np.asarray(a) - o), tuple(np.asarray(b) - o)) for a, b in segs]


def sweep(q, tip_mm: float, speed: float, segs=None, start=None, say=print) -> list:
    """Joint rows along `segs` (fingertip lines; default: lines()) at fingertip height tip_mm, from joints q at UF0
    point `start` (default: the first line's start): hop to each line's start at HOP mm/s, glide along it at `speed`.
    Each row is solved on the same wrist branch as the one before; a leg that can't be is skipped."""
    segs = lines() if segs is None else segs
    p = P.uf0(*segs[0][0], tip_mm) if start is None else start
    rows, r = [], float(DOWN[2])
    for a, b in segs:
        for goal, v in ((P.uf0(*a, tip_mm), HOP), (P.uf0(*b, tip_mm), speed)):
            seg = []
            try:
                for row in P.line(p, goal, r, r, 0, v):
                    q = tool_ik(_tip_m(row[:3]), row[3:6], q, near=True)
                    seg.append([*q.tolist(), 0.0])
            except ValueError:
                say(f"skipped a leg to ({goal[0]:.0f}, {goal[1]:.0f}) mm: out of reach on this wrist branch")
                break
            rows += seg
            p = goal
    return rows + [rows[-1]] * 10


def lowest_tip_mm(rows) -> float:
    """The lowest the fingertips go along joint rows (mm above the table)."""
    tcp = np.array(P.tcp_rows(np.asarray(rows)[:, :7]))
    return float((tcp[:, 2] - P.TABLE_UF0).min())


def feed(c, rows, queued, chunk: int = 40, keep: int = 80, each=None) -> None:
    """Post joint rows a chunk at a time as the cell plays them: never more than `keep` queued (the real cell's queue
    is small: a whole scan at once was refused, "action queue is full"). queued(): how many it has queued now."""
    import time
    i = 0
    while i < len(rows):
        q = queued()
        if each:
            each(q)
        if q + chunk > keep:
            time.sleep(0.2)
            continue
        try:
            c.act(rows[i:i + chunk], P.DT)
            i += chunk
        except RuntimeError as e:
            if "full" not in str(e):
                raise
            time.sleep(0.5)


REACH = ((0.22, 0.53), (-0.40, 0.18))  # the most the scan region may grow to (x, y m): what the arm reaches down-looking


def region_with(depth, joints, region=REGION, min_mm: float = 8.0, grow: float = 0.04) -> tuple:
    """REGION grown to cover everything the table camera sees standing up (min_mm..MAX_TALL_MM over the table, the
    arm blanked), within REACH: a bin left at the table's edge gets scanned too (one wasn't, Oct 30)."""
    d = blank_arm(depth, joints)
    pts = depth_to_base(load()["table_depth"], d, stride=2)
    h = (pts[:, 2] - SURFACE_Z) * 1000
    (rx0, rx1), (ry0, ry1) = REACH
    m = (h > min_mm) & (h < MAX_TALL_MM) & (pts[:, 0] > rx0) & (pts[:, 0] < rx1) & (pts[:, 1] > ry0) & (pts[:, 1] < ry1)
    if m.sum() < 30:
        return region
    xy = pts[m, :2]
    lo, hi = np.percentile(xy, 2, 0) - grow, np.percentile(xy, 98, 0) + grow
    (x0, x1), (y0, y1) = region
    return ((max(rx0, min(x0, lo[0])), min(rx1, max(x1, hi[0]))), (max(ry0, min(y0, lo[1])), min(ry1, max(y1, hi[1]))))


def angled(q, tip_mm: float, region, T_fc, tilt: float = 25.0, speed: float = 75.0, say=print) -> list:
    """Joint rows for angled passes after the top-down sweep: along each long edge of the region, the wrist tilted
    `tilt` deg so the camera looks in across the table (sides of things, heights by a second angle). Each pass is
    solved on the same wrist branch; one that can't be is skipped."""
    from calibration.geometry import flange_pose
    from calibration.kinematics import tool_ik as ik
    (x0, x1), (y0, y1) = region
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    h = tip_mm + 30.0
    passes = [((x0, y1), (x1, y1)), ((x1, y0), (x0, y0)), ((x0, y0), (x0, y1)), ((x1, y1), (x1, y0))]  # each edge
    rows, p_now, done = [], None, 0
    for a, b in passes:
        if done == 2:
            break
        a, b = np.asarray(a), np.asarray(b)
        mid = (a + b) / 2
        best = None
        for dw, dp in ((tilt, 0), (-tilt, 0), (0, tilt), (0, -tilt)):  # which tilt looks toward the middle
            wpr = np.array([-180.0 + dw, dp, 0.0])
            try:
                qq = ik(_tip_m(P.uf0(*mid, h)), wpr, q)
            except ValueError:
                continue
            ax = (flange_pose(qq) @ T_fc)[:3, 2]  # how much of the camera's sideways look points at the middle
            score = float(ax[:2] @ (ctr - mid)) / ((np.linalg.norm(ctr - mid) + 1e-9) * (np.linalg.norm(ax[:2]) + 1e-9))
            if best is None or score > best[0]:
                best = (score, wpr)
        if best is None or best[0] < 0.8:  # not looking in across the table
            continue
        wpr = best[1]
        seg = []
        try:  # tilt on the way to the pass's start, then along it
            start = P.uf0(*a, h)
            p0 = np.asarray(P.tcp_rows(np.asarray([[*q, 0.0]]))[0][:3]) if p_now is None else p_now
            w0 = np.array(DOWN) if p_now is None else wpr
            n = max(1, int(np.ceil(np.linalg.norm(start - p0) / (HOP * P.DT))), int(np.ceil(tilt / 3)))
            for f in np.linspace(0, 1, n + 1)[1:]:
                q = ik(_tip_m(p0 + (start - p0) * f), w0 + (wpr - w0) * f, q, near=True)
                seg.append([*q.tolist(), 0.0])
            end = P.uf0(*b, h)
            n = max(1, int(np.ceil(np.linalg.norm(end - start) / (speed * P.DT))))
            for f in np.linspace(0, 1, n + 1)[1:]:
                q = ik(_tip_m(start + (end - start) * f), wpr, q, near=True)
                seg.append([*q.tolist(), 0.0])
        except ValueError:
            say(f"the angled pass at y {mid[1] * 1000:.0f} mm is out of reach on this wrist branch: skipped")
            continue
        rows += seg
        p_now = end
        done += 1
    return rows


ORBIT_R = 0.25      # m from the hypothesis' centre to the camera
ORBIT_EL = 62.0     # deg above the table
ORBIT_SPAN = 120.0  # deg of azimuth, centred on the side facing the robot's base (where the arm reaches best)


def hypotheses(depth, joints, region, min_mm: float = 8.0, most: int = 3) -> list:
    """Things standing up in the table camera's picture (the arm blanked): [(centre xy (m), top mm)], biggest first."""
    import cv2
    d = blank_arm(depth, joints)
    pts = depth_to_base(load()["table_depth"], d, stride=1)
    h = (pts[:, 2] - SURFACE_Z) * 1000
    (x0, x1), (y0, y1) = region
    m = (h > min_mm) & (h < MAX_TALL_MM) & (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    if m.sum() < 10:
        return []
    cell = 0.005
    ij = np.floor((pts[m, :2] - (x0, y0)) / cell).astype(int)
    shape = (int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1)
    img = np.zeros(shape, np.uint8)
    img[ij[:, 0], ij[:, 1]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(img)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 3:
            continue
        sel = lab[ij[:, 0], ij[:, 1]] == k
        c = np.array([x0, y0]) + (cen[k][::-1] + 0.5) * cell
        out.append((c, float(np.percentile(h[m][sel], 90)) if sel.any() else min_mm, int(st[k, cv2.CC_STAT_AREA])))
    out.sort(key=lambda t: -t[2])
    return [(c, top) for c, top, _ in out[:most]]


def _look_rot(R0, z_to):
    """R0 turned the least way so its z axis points along z_to (the camera keeps its natural roll)."""
    z0 = R0[:, 2]
    z1 = z_to / np.linalg.norm(z_to)
    v, cth = np.cross(z0, z1), float(z0 @ z1)
    if np.linalg.norm(v) < 1e-9:
        return R0
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K / (1 + cth)) @ R0


def orbit(q, centre, top_mm: float, T_fc, min_tip_mm: float, speed: float = 75.0, say=print) -> list:
    """Joint rows flying the wrist camera along an arc of a sphere ORBIT_R round the hypothesis (centre xy, at half
    its height), always looking at it: the 10 Hz video sees it from many angles (its height, its sides). Points
    where the fingertips would come below min_tip_mm, or out of reach on this wrist branch, end the arc there."""
    from calibration.geometry import flange_pose
    from calibration.kinematics import tool_ik as ik, tool_pose, wpr_of
    o = np.array([centre[0], centre[1], SURFACE_Z + top_mm / 2000])
    T_t = tool_pose(q)
    T_tc = np.linalg.inv(T_t) @ flange_pose(q) @ T_fc  # tool (fingertips) -> camera: fixed
    R0 = T_t[:3, :3] @ T_tc[:3, :3]  # the camera's turn when the fingers point straight down
    az0 = np.degrees(np.arctan2(-o[1], -o[0]))  # facing the robot's base
    el = np.radians(ORBIT_EL)
    poses = []
    for az in np.linspace(az0 - ORBIT_SPAN / 2, az0 + ORBIT_SPAN / 2, 25):
        a = np.radians(az)
        c = o + ORBIT_R * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)])
        Rc = _look_rot(R0, o - c)
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = Rc, c
        Tt = Tc @ np.linalg.inv(T_tc)
        poses.append((Tt[:3, 3], wpr_of(Tt[:3, :3])))
    rows, p_prev = [], None
    for tip, wpr in poses:
        if (tip[2] - SURFACE_Z) * 1000 < min_tip_mm:
            continue
        steps = 1 if p_prev is None else max(1, int(np.ceil(np.linalg.norm(tip - p_prev[0]) / (speed / 1000 * P.DT))),
                                             int(np.ceil(np.abs(((wpr - p_prev[1]) + 180) % 360 - 180).max() / 1.0)))
        # (at most 1 deg of turn per 0.1 s row: the camera's view changes slowly enough for the depth matching)
        try:
            for f in np.linspace(0, 1, steps + 1)[1:]:
                tp = tip if p_prev is None else p_prev[0] + (tip - p_prev[0]) * f
                wp = wpr if p_prev is None else p_prev[1] + (((wpr - p_prev[1]) + 180) % 360 - 180) * f
                q = ik(tp, wp, q, near=p_prev is not None)
                rows.append([*q.tolist(), 0.0])
        except ValueError:
            if rows:
                break
            continue
        p_prev = (tip, wpr)
    return rows
