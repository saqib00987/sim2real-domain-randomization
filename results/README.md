# Results: three versions, and why all three are here

Most repositories publish only the run that worked. This one publishes all three,
because the failures are checkable and the story is only credible if you can verify it.

| | V1 | V2 | V3 (headline) |
|---|---|---|---|
| Images/class | 500 | 1,000 | 5,000 |
| Total renders | 15,000 | 30,000 | 150,000 |
| Seed formula | buggy | corrected | corrected |
| Epochs | fixed 10 | fixed 10 | early stopping (patience 12, max 100) |
| Validation split | none | none | 15% stratified per class |
| LR schedule | plateau on train loss | plateau on train loss | early stopping on val loss |
| From-scratch ablation |, |, | yes |
| Peak accuracy | invalid | 65.3% @ level 1,000 | 61.3% @ level 1,000 |

## Effective diversity per version

Diversity is produced by seed-cycling, `s(i) = s₀ + obj·C_obj + level·C_lev + (i mod level)`.
The `i mod level` term means a level can never produce more unique scenes than there are
images per class. This is the crux of the whole version history:

| Nominal level | V1 unique | V2 unique | V3 unique |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 10 | 10 | 10 | 10 |
| 100 | 100 | 100 | 100 |
| 500 | 500 | 500 | 500 |
| 1,000 | **500** | 1,000 | 1,000 |
| 5,000 | **500** | **1,000** | 5,000 |

### V1: `v1_buggy/`

The original seed rule omitted the level offset. With 500 images/class, every level ≥ 500
satisfied `i mod level == i`, so levels 500, 1,000 and 5,000 drew the same seeds and
rendered **byte-identical training sets**. The run contained four distinct datasets, not
six. Nothing crashed; the accuracy differences among the top three levels were pure
training variance.

`dataset_verification.png` in this folder shows it directly: the grid samples image index
50 across levels, and because `50 mod level == 50` for every level ≥ 100, four of the
columns are the same render. The figure was built to confirm diversity and instead
documents its absence.

### V2: `v2/`

Seed formula corrected, data doubled to 1,000/class. Levels 0–1,000 are now genuinely
distinct. Level 5,000 still caps at 1,000 unique scenes, so **V2's level 1,000 and level
5,000 are the same diversity drawn from disjoint seed ranges**: different scenes, equal
count. The gap between those two conditions is therefore an estimate of scene-sampling
plus training variance at fixed diversity, and a useful sanity bound on how much of the
V3 curve's shape could be noise.

V2 has no validation split and steps its LR scheduler on training loss, so it had no
mechanism to detect overfitting. That gap is what V3 closes.

### V3: `v3/`

5,000 images/class, so every level reaches its nominal count. Early stopping against a
15% stratified per-class validation split replaces fixed-epoch training, and a
from-scratch ablation isolates the contribution of ImageNet pretraining. These are the
numbers reported in the paper.

## Verifying the V1 collision yourself

```bash
# Under the V1 rule, image 50 gets the same seed at every level >= 100
python - <<'PY'
for level in (10, 100, 500, 1000, 5000):
    print(level, 42 + (50 % level))
PY
```

Or compare the rendered files directly, if you have regenerated the V1 datasets:

```bash
md5sum synthetic_images/textures_00500/025_mug/0050.png \
       synthetic_images/textures_01000/025_mug/0050.png \
       synthetic_images/textures_05000/025_mug/0050.png
```

## Files

Each version folder holds its own data and its own figures.

```
v1_buggy/  accuracy_results.csv
           accuracy_vs_diversity.png        the invalid curve
           dataset_verification.png         the seed collision, visible
           gradcam_model0_vs_model5.png
           gradcam_per_class_best.png
           renderer_test.png                renderer smoke test

v2/        accuracy_results_v2.csv
           accuracy_curve_overall.png
           accuracy_vs_diversity_v2.png
           confusion_matrices_v2.png
           gradcam_v2_per_class_best.png

v3/        accuracy_results_v3.csv          headline numbers
           V3_SUMMARY.txt                   full training + test summary
           scratch_comparison.txt           pretraining ablation
           scratch_lr_comparison.txt        ablation retuned at lr 1e-3
           accuracy_curve_overall.png       the figure in the README
           accuracy_vs_diversity_v3.png
           confusion_norand.png             the below-chance collapse
           confusion_matrices_v3.png        all six levels
           gradcam.png
           gradcam_v3_per_class_best.png
           sample_renders.png               example synthetic training images
```

Fresh runs of `src/evaluate.py` and `src/gradcam.py` write to `results/figures/`
(set in `configs/default.yaml`), which keeps them from overwriting the committed
figures above.