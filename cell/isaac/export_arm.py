"""Export the ER-4iA's link meshes and joint frames from Isaac's robot asset (the twin's), for a mesh self-collision
check on the Mac (lab/selfcollide.py). Run on the GPU box: ~/isaacsim/python.sh isaac/export_arm.py out.npz"""
import sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd"
add_reference_to_stage(usd_path=URL, prim_path="/World/er4ia")
for _ in range(20):
    app.update()
stage = omni.usd.get_context().get_stage()
cache = UsdGeom.XformCache()
out, names = {}, []
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith("/World/er4ia"):
        continue
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get() or [], float)
        if not len(pts):
            continue
        counts = np.array(m.GetFaceVertexCountsAttr().Get(), int)
        idx = np.array(m.GetFaceVertexIndicesAttr().Get(), int)
        tris, k = [], 0
        for c in counts:  # fan-triangulate
            for j in range(1, c - 1):
                tris.append([idx[k], idx[k + j], idx[k + j + 1]])
            k += c
        W = np.array(cache.GetLocalToWorldTransform(prim)).T  # world from mesh, at the asset's rest pose
        key = f"mesh{len(names)}"
        names.append(p)
        out[key + "_pts"] = pts
        out[key + "_tris"] = np.array(tris, int)
        out[key + "_world"] = W
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        print("JOINT", p, "axis", j.GetAxisAttr().Get(), "b0", b0, "b1", b1,
              "pos0", j.GetLocalPos0Attr().Get(), "rot0", j.GetLocalRot0Attr().Get(),
              "pos1", j.GetLocalPos1Attr().Get(), "rot1", j.GetLocalRot1Attr().Get(),
              "lim", j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get(), flush=True)
    if prim.IsA(UsdGeom.Xform) and prim.GetName().endswith("_link") or prim.GetName() in ("base_link", "base"):
        W = np.array(cache.GetLocalToWorldTransform(prim)).T
        print("LINK", p, np.round(W[:3, 3], 4).tolist(), flush=True)
np.savez(sys.argv[1], names=np.array(names), **out)
print("MESHES", len(names), names[:40], flush=True)
app.close()
