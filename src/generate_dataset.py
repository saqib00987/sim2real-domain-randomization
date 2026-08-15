"""Render the synthetic training datasets.

    python src/generate_dataset.py --config configs/default.yaml
    python src/generate_dataset.py --levels 0 1000        # subset
    python src/generate_dataset.py --sample-figure        # paper Figure 1

NOTE ON RESUMING: generation skips any (level, object) folder that already
holds enough PNGs. That makes the run resumable, but it also means changing
the seed formula or the image count does nothing until the old renders are
deleted. Delete the level folder to force a regenerate.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import base_parser, load_config  # noqa: E402
from renderer import DomainRandomizedRenderer  # noqa: E402


def generate_level(cfg, renderer, obj_name: str, level: int) -> int:
    obj_idx = cfg.objects.index(obj_name)
    mesh_path = cfg.mesh_path(obj_name)
    out_dir = cfg.synthetic / cfg.level_name(level) / obj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.png")))
    if existing >= cfg.images_per_class:
        return 0

    written = 0
    for i in range(cfg.images_per_class):
        img_path = out_dir / f"{i:04d}.png"
        if img_path.exists():
            continue

        if level == 0:
            img = renderer.render_object(mesh_path, randomize=False)
        else:
            seed = cfg.scene_seed(obj_idx, level, i)
            random.seed(seed)
            np.random.seed(seed)
            img = renderer.render_object(mesh_path, randomize=True)

        if img is not None:
            img.save(str(img_path))
            written += 1
    return written


def save_sample_figure(cfg, renderer, obj_name="025_mug", n=8):
    """Paper Figure 1: the same mesh under widely different scene conditions."""
    import matplotlib.pyplot as plt

    mesh_path = cfg.mesh_path(obj_name)
    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(2.0 * cols, 4.2))
    for k, ax in enumerate(axes.flat):
        random.seed(cfg.seed_base + k)
        np.random.seed(cfg.seed_base + k)
        ax.imshow(np.array(renderer.render_object(mesh_path, randomize=True)))
        ax.axis("off")
    fig.suptitle("Synthetic training renders - visual domain randomization",
                 fontsize=10)
    plt.tight_layout()

    cfg.figures.mkdir(parents=True, exist_ok=True)
    out = cfg.figures / "sample_renders.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def main():
    ap = base_parser("Render synthetic training datasets")
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                    help="subset of diversity levels (default: all)")
    ap.add_argument("--sample-figure", action="store_true",
                    help="also write results/figures/sample_renders.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = args.levels if args.levels else cfg.levels

    total = len(levels) * len(cfg.objects) * cfg.images_per_class
    print(f"Target: {len(levels)} levels x {len(cfg.objects)} objects "
          f"x {cfg.images_per_class:,} images = {total:,} renders")
    print(f"Output: {cfg.synthetic}\n")

    t0 = time.time()
    with DomainRandomizedRenderer(cfg) as renderer:
        if args.sample_figure:
            save_sample_figure(cfg, renderer)

        for level in levels:
            name = cfg.level_name(level)
            uniq = cfg.unique_scenes(level)
            note = "" if uniq == level or level == 0 else \
                f"  (capped at {uniq} unique - images_per_class is {cfg.images_per_class})"
            print(f"{name}{note}")
            for obj in cfg.objects:
                written = generate_level(cfg, renderer, obj, level)
                status = f"{written} new" if written else "complete - skipped"
                print(f"  {obj:<24} {status}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")

    print(f"\n  {'Level':<26} {'Images':>9}  {'Unique':>7}  Status")
    for level in levels:
        name = cfg.level_name(level)
        expected = len(cfg.objects) * cfg.images_per_class
        got = sum(len(list((cfg.synthetic / name / o).glob("*.png")))
                  for o in cfg.objects
                  if (cfg.synthetic / name / o).exists())
        ok = "OK" if got == expected else f"INCOMPLETE (expected {expected:,})"
        print(f"  {name:<26} {got:>9,}  {cfg.unique_scenes(level):>7,}  {ok}")


if __name__ == "__main__":
    main()