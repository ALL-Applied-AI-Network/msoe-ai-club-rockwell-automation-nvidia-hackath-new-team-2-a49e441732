import glob, cv2, numpy as np, gc
from lab.look import Shot, measure
import sys
run = (sys.argv[1] if len(sys.argv) > 1 else "runs/20261030-131742_datarun_real") + "/stops"
words = ["cup", "cube", "pen", "bolt"]
allseen = []
for f in sorted(glob.glob(run + "/*.npz")):
    z = np.load(f); k = f.split("/")[-1][:3]
    col = cv2.cvtColor(cv2.imread(f"{run}/{k}_wrist.jpg"), cv2.COLOR_BGR2RGB)
    allseen += measure(Shot(col, z["depth_wrist"], z["joints"], k), words)
    del col, z; gc.collect()
print(f"{len(allseen)} masks over 8 shots")
# one object per cluster of masks (any word) whose centres are within 12 mm: the word most often given, the measurements' medians
used = np.zeros(len(allseen), bool); objs = []
C = np.array([s.centre for s in allseen])
for i in np.argsort([-s.points for s in allseen]):
    if used[i]: continue
    near = np.nonzero((~used) & (np.linalg.norm(C - C[i], axis=1) < 0.012))[0]
    used[near] = True
    g = [allseen[j] for j in near]
    words_ = [s.word for s in g]; w = max(set(words_), key=words_.count)
    tops = [s.top_mm for s in g if s.top_mm is not None and s.points >= 30]
    objs.append((w, np.mean([s.centre for s in g], 0), np.median([s.size_mm[0] for s in g]), np.median([s.size_mm[1] for s in g]),
                 (np.median(tops) if tops else None), len(g), sum(s.real for s in g)))
for w, c, a, b, top, n, nr in sorted(objs, key=lambda o: (o[0], o[1][0])):
    if n < 2: continue
    print(f"  {w:5s} at ({c[0]*1000:4.0f}, {c[1]*1000:4.0f}) mm: {a:5.1f} x {b:4.1f} mm, top {('%.1f' % top) if top is not None else '  ?':>5} mm, "
          f"seen in {n} shots, real in {nr}")
