"""Carve a recorded data run offline: real stills + real joints, no cell needed.

    uv run python -m lab.carve_check runs/20261030-131742_datarun_real

Reads every stop (NNN.npz joints + NNN_wrist.jpg) plus the first stop's table and side shots,
asks SAM for the outlines, carves the visual hull, and writes next to the run:
    carve_map.jpg     top-down height map with each object's name, size and height
    carve_views.jpg   every view with its masks, for checking what SAM actually said
    carve.json        the objects: name, x, y, footprint, height
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, load, pixel_to_base
from lab import sam
from lab.carve import GRID_X, GRID_Y, View, carve, name_objects, objects

WORDS = ["box", "cube", "pen", "screw", "bolt"]


def _on_table(mask: np.ndarray, cam: dict, joints) -> bool:
    """Everything we care about sits on the table: a mask whose bottom edge doesn't reach the work
    surface inside the workspace (the robot's own body, the gantry, the far wall) is dropped."""
    vs, us = np.nonzero(mask)
    if not len(vs):
        return False
    i = int(np.argmax(vs))
    w, h = cam["size"]
    mh, mw = mask.shape
    p = pixel_to_base(cam, (us[i] * w / mw, vs[i] * h / mh), joints=joints)
    return GRID_X[0] - 0.05 < p[0] < GRID_X[1] + 0.05 and GRID_Y[0] - 0.05 < p[1] < GRID_Y[1] + 0.05


def _mask_union(found: list[dict], shape) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    union = np.zeros(shape, bool)
    per_word: dict[str, np.ndarray] = {}
    for d in found:
        union |= d["mask"]
        w = d["prompt"]
        per_word[w] = per_word.get(w, np.zeros(shape, bool)) | d["mask"]
    return union, per_word


def main(run_dir: str, words: list[str] = WORDS) -> None:
    run = Path(run_dir)
    stops = sorted((run / "stops").glob("*.npz"))
    cams = load()
    views: list[View] = []
    panels = []
    naming = None  # (per_word, view) from the most downward-looking shot
    best_down = -1.0
    for npz in stops:
        d = np.load(npz, allow_pickle=True)
        joints = np.asarray(d["joints"], float)
        shots = [("wrist", npz.with_name(npz.stem + "_wrist.jpg"), joints)]
        if npz == stops[0]:  # the fixed cameras see the same scene from every stop: once is enough
            shots += [("table", npz.with_name(npz.stem + "_table.jpg"), None),
                      ("side", npz.with_name(npz.stem + "_side.jpg"), None)]
        for cam_name, jpg, jnts in shots:
            if not jpg.exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
            t0 = time.time()
            found = sam.segment_many(img, words, scale=1.0)
            found = [d for d in found if _on_table(d["mask"], cams[cam_name], jnts)]
            union, per_word = _mask_union(found, img.shape[:2])
            view = View(mask=union, cam=cams[cam_name], joints=jnts)
            views.append(view)
            print(f"{npz.stem} {cam_name}: {len(found)} masks in {time.time() - t0:.1f}s "
                  f"({', '.join(sorted(set(f['prompt'] for f in found)))})", flush=True)
            down = -np.linalg.inv(view.T_base_cam())[2, 2]  # how much the camera looks straight down
            if per_word and down > best_down:
                best_down, naming = down, (per_word, view)
            over = img.copy()
            over[union] = (0.5 * over[union] + [127, 0, 0]).astype(np.uint8)
            panels.append(cv2.resize(over, (480, 480 * img.shape[0] // img.shape[1])))

    t0 = time.time()
    solid, axes = carve(views)
    obs = objects(solid, axes)
    if naming:
        name_objects(obs, axes, *naming)
    print(f"carved {len(views)} views in {time.time() - t0:.1f}s -> {len(obs)} objects", flush=True)
    for ob in obs:
        print(f"  {ob.get('name', '?'):8s} at ({ob['x_mm']:6.1f}, {ob['y_mm']:6.1f})  "
              f"footprint {ob['size_mm'][0]:.0f} x {ob['size_mm'][1]:.0f} mm  top {ob['top_mm']:.0f} mm")

    xs, ys, zs = axes
    top = np.where(solid.any(2), zs[np.argmax(solid[:, :, ::-1], 2) * -1 + solid.shape[2] - 1], np.nan)
    hmm = (top - SURFACE_Z) * 1000.0
    img = np.full(hmm.shape + (3,), 245, np.uint8)
    ok = ~np.isnan(hmm)
    if ok.any():
        n = np.clip(hmm[ok] / 60.0, 0, 1)
        img[ok] = (np.stack([255 * n, 80 + 100 * (1 - n), 255 * (1 - n)], -1)).astype(np.uint8)
    scale = 6
    img = cv2.resize(np.transpose(img, (1, 0, 2)), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    vox = float(xs[1] - xs[0])
    for ob in obs:
        u = int((ob["x_mm"] / 1000.0 - xs[0]) / vox * scale)
        v = int((ob["y_mm"] / 1000.0 - ys[0]) / vox * scale)
        txt = f"{ob.get('name', '?')} {ob['size_mm'][0]:.0f}x{ob['size_mm'][1]:.0f} h{ob['top_mm']:.0f}"
        cv2.putText(img, txt, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 3, (0, 0, 0), -1)
    cv2.imwrite(str(run / "carve_map.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if panels:
        cols = 4
        ph = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, ph - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
        rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
        pw = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, pw - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        cv2.imwrite(str(run / "carve_views.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))

    (run / "carve.json").write_text(json.dumps(
        [{k: v for k, v in ob.items() if k != "ixyz"} for ob in obs], indent=1))
    print(f"wrote {run / 'carve_map.jpg'}, carve_views.jpg, carve.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else WORDS)
