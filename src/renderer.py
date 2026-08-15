"""PyBullet domain-randomized renderer.

Each call to render_object() resets the simulation and builds a fresh 3D scene:
object colour, floor colour, camera pose and distractor objects are all
re-sampled, then rendered. This is scene-level randomization, not a 2D
transform of a fixed image -- varying camera pose in three dimensions and
inserting genuine 3D distractors both require a geometric model of the scene.

Ref: Tobin et al. 2017, Domain Randomization for Transferring Deep Neural
Networks from Simulation to the Real World.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


class DomainRandomizedRenderer:
    def __init__(self, cfg, img_size: int | None = None):
        self.cfg = cfg
        self.img_size = img_size or cfg.image_size
        self.physics_client = None

    # ── session ────────────────────────────────────────────────────────────
    def connect(self):
        self.disconnect()
        self.physics_client = p.connect(p.DIRECT)  # headless
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        return self

    def disconnect(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _random_color():
        return [random.random(), random.random(), random.random(), 1.0]

    # ── rendering ──────────────────────────────────────────────────────────
    def render_object(self, mesh_path: Path, randomize: bool = True):
        """Render one image. Returns a PIL RGB image, or None on mesh error.

        With randomize=False the scene is fixed to the baseline appearance
        defined in the config -- this is the level-0 condition.
        """
        r = self.cfg.randomization
        b = self.cfg.baseline

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Floor
        plane_id = p.loadURDF("plane.urdf")
        floor_color = self._random_color() if randomize else b["floor_colour"]
        p.changeVisualShape(plane_id, -1, rgbaColor=floor_color)

        # Target object
        try:
            vis_id = p.createVisualShape(
                p.GEOM_MESH, fileName=str(mesh_path), meshScale=[1, 1, 1],
                rgbaColor=self._random_color() if randomize else b["object_colour"],
            )
            col_id = p.createCollisionShape(
                p.GEOM_MESH, fileName=str(mesh_path), meshScale=[1, 1, 1]
            )
            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[0, 0, 0.05],
            )
        except Exception as e:
            print(f"  mesh load error ({mesh_path.name}): {e}")
            return None

        if randomize:
            p.changeVisualShape(obj_id, -1, rgbaColor=self._random_color())

            # Distractor boxes: stop the network assuming the largest or most
            # central object is always the target (Tobin et al.).
            lo, hi = r["distractors"]
            for _ in range(random.randint(lo, hi)):
                pos = [random.uniform(-0.3, 0.3),
                       random.uniform(-0.3, 0.3),
                       random.uniform(0.02, 0.15)]
                size = random.uniform(0.02, 0.08)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size] * 3,
                                          rgbaColor=self._random_color())
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size] * 3)
                p.createMultiBody(0, col, vis, pos)

            distance = random.uniform(*r["camera_distance"])
            yaw = random.uniform(*r["camera_yaw"])
            pitch = random.uniform(*r["camera_pitch"])
        else:
            distance = b["camera_distance"]
            yaw = b["camera_yaw"]
            pitch = b["camera_pitch"]

        view_mat = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.05],
            distance=distance, yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=10.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.img_size, height=self.img_size,
            viewMatrix=view_mat, projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
        )
        arr = np.array(rgb, dtype=np.uint8).reshape(self.img_size, self.img_size, 4)
        return Image.fromarray(arr[:, :, :3])