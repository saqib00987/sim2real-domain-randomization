"""Configuration loading.

Replaces the hardcoded BASE_DIR from the notebooks. All paths in the YAML are
relative to the repository root, so this runs unchanged on Windows, Linux and
macOS. Point --config at configs/local.yaml to override.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config:
    """Thin wrapper over the YAML dict with resolved paths."""

    def __init__(self, data: dict):
        self._data = data

        p = data["paths"]
        self.ycb_meshes = self._resolve(p["ycb_meshes"])
        self.real_test = self._resolve(p["real_test"])
        self.synthetic = self._resolve(p["synthetic"])
        self.checkpoints = self._resolve(p["checkpoints"])
        self.results = self._resolve(p["results"])
        self.figures = self._resolve(p["figures"])

        self.objects: list[str] = data["objects"]
        self.class_map = {i: o for i, o in enumerate(self.objects)}
        self.levels: list[int] = data["diversity_levels"]

        d = data["dataset"]
        self.images_per_class = d["images_per_class"]
        self.image_size = d["image_size"]
        self.seed_base = d["seed_base"]
        self.obj_stride = d["obj_stride"]
        self.level_stride = d["level_stride"]

        self.randomization = data["randomization"]
        self.baseline = data["baseline"]

        m = data["model"]
        self.architecture = m["architecture"]
        self.pretrained = m["pretrained"]
        self.num_classes = m["num_classes"]

        t = data["training"]
        self.batch_size = t["batch_size"]
        self.learning_rate = t["learning_rate"]
        self.max_epochs = t["max_epochs"]
        self.patience = t["early_stopping_patience"]
        self.val_split = t["val_split"]
        self.num_workers = t["num_workers"]
        self.split_seed = t["split_seed"]

        self.gradcam = data["gradcam"]

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    def __getitem__(self, key):
        return self._data[key]

    # ── seed-cycling ───────────────────────────────────────────────────────
    def scene_seed(self, obj_idx: int, level: int, i: int) -> int:
        """Per-image seed.

            s(i) = base + obj*obj_stride + level*level_stride + (i mod level)

        The level term is what keeps levels distinct once level exceeds
        images_per_class. Without it (the V1 bug), every level >= the image
        count collapses onto the same seed range and renders identical data.
        """
        return (
            self.seed_base
            + obj_idx * self.obj_stride
            + level * self.level_stride
            + (i % level)
        )

    def level_name(self, level: int) -> str:
        return "no_randomization" if level == 0 else f"textures_{level:05d}"

    def mesh_path(self, obj_name: str) -> Path:
        return self.ycb_meshes / obj_name / "google_16k" / "textured.obj"

    def unique_scenes(self, level: int) -> int:
        """How many genuinely distinct scenes a level can produce."""
        if level == 0:
            return 1
        return min(level, self.images_per_class)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return Config(yaml.safe_load(f))


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="path to YAML config (default: configs/default.yaml)")
    return ap