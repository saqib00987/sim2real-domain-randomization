"""Grad-CAM comparison: what the network attends to at each diversity level.

    python src/gradcam.py --config configs/default.yaml
    python src/gradcam.py --levels 0 1000

Produces a grid with one row per class: the real photograph, the baseline
model's attention, and the comparison model's attention. Heatmaps are computed
for the ground-truth class so the two columns are directly comparable even when
a model predicts wrongly.

Ref: Selvaraju et al. 2017.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import base_parser, load_config  # noqa: E402
from data import REAL_IMAGE_SUFFIXES, build_transforms, get_device  # noqa: E402
from train import build_model  # noqa: E402

CORRECT, WRONG = "#2A9D8F", "#E63946"


class GradCAM:
    """Gradient-weighted class activation mapping, hooked into layer4[-1]."""

    def __init__(self, model, device, target_layer="layer4"):
        self.model = model.eval()
        self.device = device
        self._features = None
        self._gradients = None
        target = getattr(model, target_layer)[-1]
        self._hooks = [
            target.register_forward_hook(self._fwd),
            target.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, module, inp, out):
        self._features = out.detach()

    def _bwd(self, module, gin, gout):
        self._gradients = gout[0].detach()

    def generate(self, img_tensor, class_idx=None):
        img_tensor = img_tensor.to(self.device)
        self.model.zero_grad()
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self._features).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        return cv2.resize(cam, (w, h)), class_idx

    def close(self):
        for h in self._hooks:
            h.remove()


def overlay(orig_pil, cam, size, alpha=0.45):
    base = np.array(orig_pil.resize((size, size)))
    heat = cv2.cvtColor(
        cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    return Image.fromarray((alpha * heat + (1 - alpha) * base).astype(np.uint8))


def load_model(cfg, level, device, scratch=False):
    suffix = "_scratch" if scratch else ""
    path = cfg.checkpoints / f"model_{cfg.level_name(level)}{suffix}.pth"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path} - run train.py first")
    ckpt = torch.load(path, map_location=device)
    # Architecture must match how the checkpoint was built; the weights are
    # overwritten immediately, so this only selects the right module shape.
    model = build_model(cfg, ckpt.get("pretrained", True), device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def sample_images(cfg, index):
    out = []
    for class_idx, obj in enumerate(cfg.objects):
        obj_dir = cfg.real_test / obj
        imgs = set()
        for pattern in REAL_IMAGE_SUFFIXES:
            imgs.update(obj_dir.rglob(pattern))
        imgs = sorted(p for p in imgs
                      if not {"masks", "poses"} & set(p.relative_to(obj_dir).parts))
        if imgs:
            out.append((imgs[min(index, len(imgs) - 1)], class_idx, obj))
    return out


def main():
    ap = base_parser("Grad-CAM attention comparison")
    ap.add_argument("--levels", type=int, nargs=2, default=None,
                    metavar=("BASELINE", "COMPARISON"))
    ap.add_argument("--index", type=int, default=10,
                    help="which real image per class (default: 10)")
    ap.add_argument("--scratch", action="store_true",
                    help="use the from-scratch ablation checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    baseline_level, compare_level = args.levels or cfg.gradcam["compare_levels"]
    device = get_device()
    _, test_tf = build_transforms(cfg.image_size)

    samples = sample_images(cfg, args.index)
    if not samples:
        print("No real test images found.")
        return

    model_a = load_model(cfg, baseline_level, device, args.scratch)
    model_b = load_model(cfg, compare_level, device, args.scratch)
    cam_a = GradCAM(model_a, device, cfg.gradcam["target_layer"])
    cam_b = GradCAM(model_b, device, cfg.gradcam["target_layer"])

    titles = ["Real YCB photograph",
              "no randomization" if baseline_level == 0 else f"level {baseline_level:,}",
              f"level {compare_level:,}"]

    fig, axes = plt.subplots(len(samples), 3, figsize=(10.5, 3.4 * len(samples)))
    for col, t in enumerate(titles):
        axes[0, col].set_title(t, fontsize=10, pad=10)

    for row, (img_path, true_class, obj) in enumerate(samples):
        orig = Image.open(img_path).convert("RGB")
        tensor = test_tf(orig).unsqueeze(0)
        short = obj.split("_", 1)[1].replace("_", " ")

        axes[row, 0].imshow(orig.resize((cfg.image_size, cfg.image_size)))
        axes[row, 0].set_ylabel(short, fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, (model, cam) in enumerate([(model_a, cam_a), (model_b, cam_b)], start=1):
            with torch.no_grad():
                pred = int(model(tensor.clone().to(device)).argmax(1).item())
            heat, _ = cam.generate(tensor.clone(), class_idx=true_class)
            axes[row, col].imshow(overlay(orig, heat, cfg.image_size))
            axes[row, col].axis("off")
            ok = pred == true_class
            label = "correct" if ok else \
                f"wrong -> {cfg.objects[pred].split('_', 1)[1].replace('_', ' ')}"
            axes[row, col].text(0.5, -0.04, label, transform=axes[row, col].transAxes,
                                ha="center", va="top", fontsize=8,
                                color=CORRECT if ok else WRONG)

    cam_a.close()
    cam_b.close()

    fig.suptitle("Grad-CAM: attention shifts to the object with domain randomization",
                 fontsize=11)
    plt.tight_layout()

    cfg.figures.mkdir(parents=True, exist_ok=True)
    out = cfg.figures / "gradcam_comparison.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()