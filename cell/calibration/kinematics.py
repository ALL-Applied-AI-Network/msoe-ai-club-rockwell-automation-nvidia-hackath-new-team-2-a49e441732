"""The ER-4iA's tool pose in the controller's terms (x, y, z, w, p, r), and IK for any orientation (numpy + scipy).

    R = wpr_matrix(w, p, r)          # the tool frame's rotation in the base frame (UTOOL1: z = the approach direction)
    tip, wpr = tcp_of(joints)        # forward: joints (deg) -> fingertip (base, m) and w, p, r (deg)
    q = tool_ik(tip, (w, p, r), q0)  # inverse: the joints that put the fingertip there, turned that way

FANUC's w, p, r are fixed-axis rotations about x, then y, then z: R = Rz(r) Ry(p) Rx(w). w, p, r = -180, 0, R points
the fingers straight down and turns them by R (every pose the policies used until tilting). The tool frame relates to
the twin's flange frame (J6_link: tool along +x) by a constant rotation, FLANGE_FROM_TOOL, fixed by that straight-down
case: the flange's x is the tool's z (the approach), its y is the tool's -y, its z is the tool's x.

The convention is the controller's documented one, checked against the straight-down poses; a tilted pose read off the
pendant (w, p, r with its joints) is the check still to do before sending tilted poses to the real cell.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from calibration.geometry import UF0_Z_MM, flange_pose

TCP_OFFSET = 0.293  # fingertip, metres along the flange's +x (tool) axis
JOINT_LIMITS = [(-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360)]
HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286]
FLANGE_FROM_TOOL = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])  # columns: flange x, y, z in tool axes
DOWN = (-180.0, 0.0, 0.0)
# Which way the jaws close, measured from the tool's x axis about its z axis. The twin's gripper (scene_config
# FINGER_SPREAD_AXIS: the flange's y = the tool's -y) closes along the tool's y: 90. So at r = 0, fingers down, the
# jaws close left-right (base y), not front-back. UNVERIFIED on the real cell: look at the fingers at R = 0.
JAW_AXIS_DEG = 90.0


def tool_r(jaw_deg: float) -> float:
    """The tool's r (deg) that makes the jaws close along jaw_deg (base frame, fingers down)."""
    return (jaw_deg - JAW_AXIS_DEG + 180.0) % 360.0 - 180.0


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wpr_matrix(w: float, p: float, r: float) -> np.ndarray:
    """The tool frame's rotation in the base frame for w, p, r in degrees."""
    w, p, r = np.radians([w, p, r])
    return _rz(r) @ _ry(p) @ _rx(w)


def wpr_of(R: np.ndarray) -> np.ndarray:
    """w, p, r (deg) of a tool rotation; w is reported as -180 (not +180) when the fingers point straight down."""
    p = -np.arcsin(np.clip(R[2, 0], -1, 1))
    w = np.arctan2(R[2, 1], R[2, 2])
    r = np.arctan2(R[1, 0], R[0, 0])
    out = np.degrees([w, p, r])
    if out[0] > 179.99:
        out[0] -= 360.0
    return out


def tcp_of(joints) -> tuple[np.ndarray, np.ndarray]:
    """Joints (deg; a 7th value, the gripper, is ignored) -> (fingertip in the base frame, m; w, p, r in deg)."""
    T = flange_pose(joints)
    return T[:3, 3] + T[:3, 0] * TCP_OFFSET, wpr_of(T[:3, :3] @ FLANGE_FROM_TOOL.T)


def tool_pose(joints) -> np.ndarray:
    """The tool frame's 4x4 pose in the base frame (origin at the fingertip, z along the approach) for joints in deg."""
    F = flange_pose(joints)
    T = np.eye(4)
    T[:3, :3] = F[:3, :3] @ FLANGE_FROM_TOOL.T
    T[:3, 3] = F[:3, 3] + F[:3, 0] * TCP_OFFSET
    return T


def tcp_uf0(joints) -> list[float]:
    """The controller's tcp_pose: x, y, z mm in UF0, w, p, r deg."""
    tip, wpr = tcp_of(joints)
    return [tip[0] * 1000, tip[1] * 1000, tip[2] * 1000 + UF0_Z_MM, *wpr.tolist()]


def tool_ik(tip, wpr=DOWN, q0=HOME, tol_mm: float = 2.0, near: bool = False) -> np.ndarray:
    """Joints (6, deg) that put the fingertip at tip (base frame, m) with the tool turned to w, p, r (deg).
    near: start only from q0 (the last pose of a motion), never from elsewhere: the answer is then on the same
    branch as q0, or there is none (another branch would make the arm whip round)."""
    Rf = wpr_matrix(*wpr) @ FLANGE_FROM_TOOL  # the flange rotation that gives this tool rotation
    tip = np.asarray(tip, float)

    def err(q):
        T = flange_pose(q)
        return np.concatenate([(T[:3, 3] + T[:3, 0] * TCP_OFFSET - tip) * 10, T[:3, 0] - Rf[:, 0], 0.5 * (T[:3, 1] - Rf[:, 1])])

    lo, hi = np.array(JOINT_LIMITS, float).T
    best = None
    for start in (_nearby(q0) if near else _starts(q0)):  # the last pose first (smooth motion); then other wrist branches, for tilted poses
        sol = least_squares(err, np.clip(np.asarray(start, float)[:6], lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
        if sol.cost < 1e-8:
            break
    best.x = _nearest_turn(best.x, q0)
    T = flange_pose(best.x)
    miss = np.linalg.norm(T[:3, 3] + T[:3, 0] * TCP_OFFSET - tip)
    turn = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ Rf) - 1) / 2, -1, 1)))
    if miss > tol_mm / 1000 or turn > 2.0:
        raise ValueError(f"no IK solution for {np.round(tip * 1000)} mm, w p r {np.round(wpr, 1)} "
                         f"(missed by {miss * 1000:.1f} mm, {turn:.1f} deg)")
    return best.x


def _nearest_turn(q, q0) -> np.ndarray:
    """J4 and J6 can turn more than a full circle: +180 and -180 are the same pose. Take the one nearest q0, so a motion
    never asks the wrist to spin all the way round (the controller refuses that, rightly)."""
    q = np.array(q, float)
    lo, hi = np.array(JOINT_LIMITS, float).T
    q0 = np.asarray(q0, float)[:6]
    for j in (3, 5):
        alt = q[j] + 360.0 * np.round((q0[j] - q[j]) / 360.0)
        if lo[j] <= alt <= hi[j]:
            q[j] = alt
    return q


def _starts(q0):
    """Starting guesses: q0, home, home with the wrist (J4, J6) turned each way, then a few fixed random ones."""
    yield q0
    yield HOME
    for j4 in (-90.0, 90.0, 180.0, -180.0):
        for j6 in (0.0, 90.0, -90.0):
            q = np.array(HOME, float)
            q[3], q[5] = j4, j6
            yield q
    rng = np.random.default_rng(0)
    lo, hi = np.array(JOINT_LIMITS, float).T
    for _ in range(8):
        yield rng.uniform(lo * 0.8, hi * 0.8)


def _nearby(q0):
    """q0, then small nudges of the wrist (J4, J5, J6): still the same branch, but off a local minimum near a wrist
    singularity."""
    yield q0
    for dq in ((0, 0, 0, 12, 0, -12), (0, 0, 0, -12, 0, 12), (0, 0, 0, 0, 8, 0), (0, 0, 0, 0, -8, 0),
               (0, 0, 0, 20, 5, -20), (0, 0, 0, -20, -5, 20)):
        yield np.asarray(q0, float)[:6] + dq


def hand_matrix(wpr) -> np.ndarray:
    """The MuJoCo cell's hand frame (fingers along -z, jaws closing along its x) for a tool rotation:
    R_tool Rx(180) Rz(-JAW_AXIS), so the jaws close where the real (twin's) jaws do."""
    return wpr_matrix(*wpr) @ _rx(np.pi) @ _rz(-np.radians(JAW_AXIS_DEG))


def tool_from_hand(R_hand: np.ndarray) -> np.ndarray:
    return R_hand @ _rz(np.radians(JAW_AXIS_DEG)) @ _rx(np.pi)
