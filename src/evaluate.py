"""Evaluate every trained model on the real YCB photographs.

    python src/evaluate.py --config configs/default.yaml
    python src/evaluate.py --scratch          # ablation checkpoints

Writes accuracy_results.csv, accuracy_curve_overall.png and
confusion_matrices.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import base_parser, load_config  # noqa: E402
from data import RealYCBDataset, build_transforms, get_device  # noqa: E402
from train import build_model  # noqa: E402

OBJ_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#6A4C93"]


def evaluate_level(cfg, level, pretrained, device, test_loader):
    suffix = "" if pretrained else "_scratch"
    ckpt_path = cfg.checkpoints / f"model_{cfg.level_name(level)}{suffix}.pth"
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg, pretrained, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbl in test_loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
            labels.extend(lbl.tolist())

    preds, labels = np.array(preds), np.array(labels)
    overall = 100.0 * (preds == labels).mean()
    per_class = {obj: 100.0 * (preds[labels == i] == i).mean()
                 for i, obj in enumerate(cfg.objects) if (labels == i).sum() > 0}

    n = len(cfg.objects)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    # A model predicting one class for everything is degenerate, not learning.
    collapsed = len(np.unique(preds)) == 1
    return {"overall": float(overall), "per_class": per_class,
            "confusion": cm, "collapsed": collapsed}


def plot_accuracy_curve(cfg, results, out_path):
    levels = [l for l in cfg.levels if l in results]
    x = range(len(levels))
    y = [results[l]["overall"] for l in levels]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, y, marker="o", linewidth=2, color="#2A9D8F", zorder=3)
    ax.axhline(100.0 / len(cfg.objects), linestyle="--", linewidth=1,
               color="#888", label=f"chance ({100.0 / len(cfg.objects):.0f}%)")

    peak = max(range(len(y)), key=lambda i: y[i])
    ax.scatter([peak], [y[peak]], s=140, facecolors="none",
               edgecolors="#E63946", linewidths=2, zorder=4)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["none" if l == 0 else f"{l:,}" for l in levels])
    ax.set_xlabel("Colour diversity level (unique scenes per class)")
    ax.set_ylabel("Top-1 accuracy on real photographs (%)")
    ax.set_title("Real-world accuracy vs. randomization diversity")
    ax.set_ylim(0, max(y) * 1.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(cfg, results, out_path):
    levels = [l for l in cfg.levels if l in results]
    short = [o.split("_", 1)[1].replace("_", " ") for o in cfg.objects]
    ncols = 3
    nrows = (len(levels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    for ax, level in zip(axes.flat, levels):
        cm = results[level]["confusion"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=7)
        title = "no randomization" if level == 0 else f"level {level:,}"
        ax.set_title(f"{title} - {results[level]['overall']:.1f}%", fontsize=9)
        for i in range(len(short)):
            for j in range(len(short)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    for ax in list(axes.flat)[len(levels):]:
        ax.axis("off")

    fig.supxlabel("Predicted class", fontsize=9)
    fig.supylabel("True class", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = base_parser("Evaluate models on real YCB photographs")
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--scratch", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = args.levels if args.levels else cfg.levels
    pretrained = cfg.pretrained and not args.scratch
    device = get_device()

    _, test_tf = build_transforms(cfg.image_size)
    test_ds = RealYCBDataset(cfg, transform=test_tf)
    print(f"Real test set: {len(test_ds):,} images")
    for i, obj in enumerate(cfg.objects):
        print(f"  {obj:<24} {sum(1 for _, l in test_ds.samples if l == i):>5}")

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    results = {}
    for level in levels:
        r = evaluate_level(cfg, level, pretrained, device, test_loader)
        if r:
            results[level] = r

    if not results:
        print("\nNo checkpoints found - run train.py first.")
        return

    short = [o.split("_", 1)[1].replace("_", " ")[:9] for o in cfg.objects]
    print(f"\n  {'Level':<20} {'Overall':>8}  " + "  ".join(f"{s:>9}" for s in short))
    for level in levels:
        if level not in results:
            continue
        r = results[level]
        cells = "  ".join(f"{r['per_class'].get(o, 0):>8.1f}%" for o in cfg.objects)
        flag = "  <- collapsed to one class" if r["collapsed"] else ""
        print(f"  {cfg.level_name(level):<20} {r['overall']:>7.1f}%  {cells}{flag}")

    cfg.results.mkdir(parents=True, exist_ok=True)
    cfg.figures.mkdir(parents=True, exist_ok=True)
    tag = "_scratch" if not pretrained else ""

    csv_path = cfg.results / f"accuracy_results{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["texture_level", "level_name", "unique_scenes",
                    "overall_accuracy", "collapsed"] + cfg.objects)
        for level in levels:
            if level not in results:
                continue
            r = results[level]
            w.writerow([level, cfg.level_name(level), cfg.unique_scenes(level),
                        round(r["overall"], 2), r["collapsed"]] +
                       [round(r["per_class"].get(o, 0), 2) for o in cfg.objects])
    print(f"\nCSV     -> {csv_path}")

    curve = cfg.figures / f"accuracy_curve_overall{tag}.png"
    plot_accuracy_curve(cfg, results, curve)
    print(f"Curve   -> {curve}")

    cms = cfg.figures / f"confusion_matrices{tag}.png"
    plot_confusion_matrices(cfg, results, cms)
    print(f"Matrices-> {cms}")


if __name__ == "__main__":
    main()