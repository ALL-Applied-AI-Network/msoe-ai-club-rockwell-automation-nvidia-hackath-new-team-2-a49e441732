"""V8's head in Rerun: the camera as a pyramid with the photo on its base, the rays it casts,
and the volumes that survive. Nothing else - what you see is exactly what the eyes know.

Every shot k appears at world/shots/<k>: a pinhole frustum placed where the wrist camera REALLY
was (observed joints, not commanded), the photo on the image plane with SAM's masks outlined,
and the cast rays from the optical center through each mask down to the table. Blobs live at
world/blobs/<id> as the voxels still possible for that shape.
"""
from __future__ import annotations

import cv2
import numpy as np
import rerun as rr

from calibration.geometry import SURFACE_Z, flange_pose


def setup() -> None:
    """The blueprint (one 3D pane) and the table. Called when a cell connects."""
    import rerun.blueprint as rrb
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(name="what the eyes know", origin="/world"),
        collapse_panels=True))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/table", rr.Boxes3D(centers=[[0.49, 0.0, SURFACE_Z - 0.002]],
                                     half_sizes=[[0.29, 0.36, 0.002]],
                                     colors=[(235, 235, 235)], fill_mode="solid"), static=True)


def clear() -> None:
    """The shot pyramid and every blob shape, gone from the viewer: a fresh head."""
    rr.log("world/shot", rr.Clear(recursive=True))
    rr.log("world/blobs", rr.Clear(recursive=True))


class View:
    def __init__(self) -> None:
        self.step = 0

    def shot(self, shot, cam: dict, grid) -> None:
        """The pyramid: camera pose, pinhole, photo (masks outlined), and the cast rays.
        ONE entity, always the latest shot - the trail of old pyramids buried the scene (scrub the
        'shot' timeline to revisit any earlier view)."""
        self.step += 1
        rr.set_time("shot", sequence=self.step)
        T = flange_pose(shot.joints) @ cam["T"]
        w, h = cam["size"]
        ent = "world/shot"
        rr.log(ent, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        ih, iw = shot.image.shape[:2]
        s = iw / w   # the pinhole must match the PHOTO's resolution, or it fills a quadrant of the plane
        rr.log(ent, rr.Pinhole(image_from_camera=[[cam["fx"] * s, 0, iw / 2], [0, cam["fx"] * s, ih / 2], [0, 0, 1]],
                               resolution=[iw, ih], image_plane_distance=0.12))
        vis = shot.image.copy()
        for m in shot.masks:
            mh, mw = m["mask"].shape
            cs, _ = cv2.findContours(m["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = (255, 120, 80) if m["cut"] else (60, 255, 80)
            cv2.drawContours(vis, [np.round(c * [vis.shape[1] / mw, vis.shape[0] / mh]).astype(np.int32)
                                   for c in cs], -1, col, 2)
        rr.log(ent, rr.Image(vis).compress(jpeg_quality=70))

        # the cast volume: rays from the optical center through the mask outline, cut at the table
        eye = T[:3, 3]
        strips = []
        for m in shot.masks:
            mh, mw = m["mask"].shape
            cs, _ = cv2.findContours(m["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                pts = c[:: max(1, len(c) // 24), 0, :]          # ~24 rays per outline
                for px in pts:
                    u, v = px[0] * w / mw, px[1] * h / mh
                    x, y = (u - w / 2) / cam["fx"], (v - h / 2) / cam["fx"]
                    d = T[:3, :3] @ np.array([x, y, 1.0])
                    if abs(d[2]) < 1e-6:
                        continue
                    t = (SURFACE_Z - eye[2]) / d[2]
                    if t <= 0:
                        continue
                    strips.append(np.stack([eye, eye + d * t]))
        if strips:
            rr.log(f"{ent}/rays", rr.LineStrips3D(strips, colors=[(90, 170, 255, 110)] * len(strips),
                                                  radii=0.0004))

    def blobs(self, blobs: list, grid) -> None:
        """Each blob as ONE solid carved shape (marching cubes over its surviving voxels): a block
        of wood being whittled, not a cloud of points - a shape motion planning can reason about."""
        from skimage.measure import marching_cubes
        rr.set_time("shot", sequence=self.step)
        rr.log("world/blobs", rr.Clear(recursive=True))
        palette = [(235, 170, 80), (120, 180, 250), (150, 220, 140), (230, 130, 160),
                   (200, 200, 120), (160, 150, 230)]
        vox = float(grid.xs[1] - grid.xs[0])
        for b in blobs:
            vol = np.pad(b.keep.reshape(grid.shape).astype(np.float32), 1)
            if vol.max() == 0:
                continue
            verts, faces, normals, _ = marching_cubes(vol, 0.5)
            verts = (verts - 1) * vox + [grid.xs[0], grid.ys[0], grid.zs[0]]
            col = palette[b.id % len(palette)]
            rr.log(f"world/blobs/{b.id}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                                    vertex_normals=normals, albedo_factor=col))
            top = verts[verts[:, 2].argmax()]
            tag = f"{b.name} #{b.id}" if getattr(b, "name", "") else f"blob #{b.id}"
            rr.log(f"world/blobs/{b.id}/tag", rr.Points3D([top + [0, 0, 0.01]], radii=0.0008,
                                                          colors=[col], labels=[tag]))
