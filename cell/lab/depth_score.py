"""Score every way we measure depth against the real sizes, from one depth test (lab/ladder.py, ladder00).

    uv run python -m lab.depth_score runs/<time>_ladder/ladder00_depth_test          (uses real.json there)
    from code: score(run_dir, {"2": {"height": 15, "width": 15}, ...}) -> {"rows": [...], "summary": {...}}

real.json: the found object's number -> its real height and width (mm), typed in on the page. Methods:
    sweep map     the scan's evidence grid (sweep + orbits): the object's top, and the smaller side of its outline
    close look    the still wrist pictures over it: its width (the top face's smaller side)
    wrist depth   the wrist depth camera, still, ~350 mm up, median of 5: the top of what's within 12 mm of its centre
    table depth   the table depth camera, the same five moments: likewise
Each gives measured, real and the error (mm).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, load


def _top_near(pts: np.ndarray, centre_m, r_mm: float = 12.0) -> float | None:
    d = np.hypot(pts[:, 0] - centre_m[0], pts[:, 1] - centre_m[1]) * 1000
    h = (pts[d < r_mm, 2] - SURFACE_Z) * 1000
    h = h[(h > -10) & (h < 200)]
    return float(np.percentile(h, 90)) if len(h) >= 5 else None


def score(run: Path, real: dict) -> dict:
    run = Path(run)
    res = json.loads((run / "result.json").read_text())
    cams = load()
    rows = []
    for o in res["objects"]:
        r = real.get(str(o["id"])) or {}
        rh, rw = r.get("height"), r.get("width")
        centre = np.array(o.get("look_centre_mm") or o["depth_centre_mm"]) / 1000

        def row(method, what, measured, truth):
            rows.append({"id": o["id"], "method": method, "what": what,
                         "measured_mm": None if measured is None else round(float(measured), 1),
                         "real_mm": truth, "error_mm": None if (measured is None or truth in (None, "")) else
                         round(float(measured) - float(truth), 1)})
        row("sweep map", "height", o.get("depth_top_mm"), rh)
        row("sweep map", "width", min(o["depth_size_mm"]) if o.get("depth_size_mm") else None, rw)
        row("close look", "width", o.get("look_width_mm"), rw)
        st = run / f"still_{o['id']}.npz"
        if st.exists():
            z = np.load(st)
            wd = np.median(np.where(z["wrist_depth"] > 0, z["wrist_depth"], np.nan), 0)
            wd = np.nan_to_num(wd, nan=0).astype(np.uint16)
            pts = depth_to_base(cams["wrist_depth"], wd, joints=z["joints"][0]) if "wrist_depth" in cams else None
            row("wrist depth", "height", _top_near(pts, centre) if pts is not None and len(pts) else None, rh)
            td = np.median(np.where(z["table_depth"] > 0, z["table_depth"], np.nan), 0)
            td = np.nan_to_num(td, nan=0).astype(np.uint16)
            pts = depth_to_base(cams["table_depth"], td)
            row("table depth", "height", _top_near(pts, centre) if len(pts) else None, rh)
    summary = {}
    for m in ("sweep map", "close look", "wrist depth", "table depth"):
        for w in ("height", "width"):
            e = [abs(x["error_mm"]) for x in rows if x["method"] == m and x["what"] == w and x["error_mm"] is not None]
            if e:
                summary[f"{m} {w}"] = {"median_abs_mm": round(float(np.median(e)), 1), "worst_mm": round(max(e), 1),
                                       "n": len(e)}
    out = {"rows": rows, "summary": summary}
    (run / "score.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run = Path(sys.argv[1])
    real = json.loads((run / "real.json").read_text()) if (run / "real.json").exists() else {}
    print(json.dumps(score(run, real)["summary"], indent=1))
