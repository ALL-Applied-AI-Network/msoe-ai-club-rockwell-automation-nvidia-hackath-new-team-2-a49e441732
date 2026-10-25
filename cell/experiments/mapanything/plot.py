"""The picture: sim points coloured by error, raw depth vs MapAnything, plus error histograms -> result.png

    uv run --with matplotlib python -m experiments.mapanything.plot
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.geometry import SURFACE_Z
from experiments.mapanything.compare import HERE, _dist, _scene
from lab.trial import build, load_trial

c = np.load(HERE / "sim.npz")
scene = _scene(build(load_trial("sort_mixed"), 0).object_meshes())
K = c["K"][0]


def raw():
    out = []
    for i in range(len(c["img"])):
        z = c["depth"][i]
        v, u = np.nonzero(z > 0)
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * z[v, u], (v - K[1, 2]) / K[1, 1] * z[v, u], z[v, u]], -1)
        out.append(pc @ c["pose"][i][:3, :3].T + c["pose"][i][:3, 3])
    return np.concatenate(out)


def ma(m, rescale):
    """The model's depth per frame, placed with our camera poses; rescale = its scale fixed from our depth."""
    r = np.load(HERE / f"sim_{m}.npz")
    out = []
    for i in range(len(c["pose"])):
        P, Q = r["poses"][i], c["pose"][i]
        pc = (r["pts"][i][r["mask"][i]] - P[:3, 3]) @ P[:3, :3]
        if rescale:
            d = c["depth"][i]
            pc *= np.median(d[d > 0]) / np.median(pc[:, 2])
        out.append(pc @ Q[:3, :3].T + Q[:3, 3])
    return np.concatenate(out)


sets = [("Raw depth\n(one frame at a time)", raw()),
        ("MapAnything: images + arm poses", ma("images_poses", False)),
        ("MapAnything: images + poses + depth\n(best case: scale fixed from our depth)", ma("images_poses_depth", True))]
fig = plt.figure(figsize=(16, 9.5), facecolor="#111317")
rng = np.random.default_rng(0)
errs = []
for k, (name, p) in enumerate(sets):
    p = p[(p[:, 2] > SURFACE_Z - 0.02) & (p[:, 2] < SURFACE_Z + 0.06)]
    d, tab = _dist(scene, p)
    errs.append((name, d[~tab]))
    keep = rng.choice(len(p), min(len(p), 60000), replace=False)
    ax = fig.add_subplot(2, 3, k + 1, projection="3d", facecolor="#111317")
    sc = ax.scatter(p[keep, 0], p[keep, 1], p[keep, 2], c=np.clip(d[keep] * 1000, 0, 6), cmap="RdYlGn_r", vmin=0, vmax=6, s=0.3)
    ax.view_init(elev=28, azim=-150)
    ax.set_title(name, color="w", fontsize=11)
    ax.set_xlim(p[:, 0].min(), p[:, 0].max()); ax.set_ylim(p[:, 1].min(), p[:, 1].max())
    ax.set_zlim(SURFACE_Z - 0.02, SURFACE_Z + 0.06)
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), 0.08)); ax.set_axis_off()
cb = fig.colorbar(sc, ax=fig.axes, shrink=0.35, location="right", pad=0.02)
cb.set_label("distance to the true surface (mm)", color="w")
cb.ax.yaxis.set_tick_params(color="w"); plt.setp(cb.ax.get_yticklabels(), color="w")
ax = fig.add_subplot(2, 1, 2, facecolor="#171a20")
for (name, e), col in zip(errs, ["#7ee2a0", "#f2cc60", "#9cc4ff"]):
    ax.hist(e * 1000, bins=np.linspace(0, 15, 76), histtype="step", lw=2, color=col, density=True,
            label=f"{name.splitlines()[0]}: median {np.median(e) * 1000:.1f} mm, worse than 5 mm: {np.mean(e > 0.005) * 100:.0f}%")
ax.set_xlabel("error on the objects (mm)", color="w"); ax.set_ylabel("share of points", color="w"); ax.tick_params(colors="w")
for s in ax.spines.values():
    s.set_color("#2a2f38")
ax.legend(facecolor="#171a20", edgecolor="#2a2f38", labelcolor="w", fontsize=11)
ax.set_title("Wrist camera sweeping 10 cm past the objects, 0.23 m away (sim, noisy depth). Further left = better", color="w")
plt.savefig(HERE / "result.png", dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("wrote", HERE / "result.png")
