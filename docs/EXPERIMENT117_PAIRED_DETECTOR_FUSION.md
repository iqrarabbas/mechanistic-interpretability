# Experiment 117: Paired, clean-calibrated corruption detection

## Question

Do the single-view internal-statistics detector (Experiment 114) and the four-view consistency detector (Experiment 116) complement each other on **the same images**, without using corruption labels to tune their combination?

## Protocol

- Frozen `google/vit-base-patch16-224`; no adapter, SAE repair, clean counterpart, or true label at inference.
- Both detectors retain their original, separately fitted clean-only photo/Sketch Ledoit-Wolf models. Their cached clean validation feature sets contain the same 500 photo and 1,000 Sketch images in the same order.
- Convert each detector's distance to its empirical clean-validation percentile. Fusion is the maximum percentile. Set each detection threshold at its own 95th clean-validation percentile. This rule is fixed before seeing Experiment 117 corruption outcomes.
- Evaluate all three rules on the same 997 SHA-256-disjoint new ImageNet-Sketch images, each clean and under all 15 ImageNet-C algorithms at severity 4. This is online corruption of Sketch images, **not official ImageNet-C evaluation**.
- Prior Sketch manifests, including those used in Experiments 113–116, are excluded by image hash. The Sketch domain and candidate fusion rule were informed by earlier experiments; this is a new-image confirmation, not a completely untouched benchmark/domain.
- Exact McNemar tests and paired image bootstrap confidence intervals are saved for each corruption in `summary.json`. The component comparison uses four ViT views for the multi-view and fused rules; it is not a conditional-compute detector.

## Results

| Detector | Clean FPR | Mean severity-4 recall | Mean AUROC |
| --- | ---: | ---: | ---: |
| Single view | 3.81% | 40.01% | 0.830 |
| Four views | 6.62% | **45.34%** | **0.848** |
| Clean-calibrated fusion | 5.42% | 42.99% | 0.845 |

The fusion does **not** improve average detection over the four-view model. It regains some Defocus Blur (64.2% to 69.0%) and Contrast (31.2% to 48.2%) recall, but loses substantial Pixelate (61.1% to 39.6%) and Frost (67.0% to 56.4%) recall. In particular, a stronger score on an image does not imply a better globally calibrated binary decision: the maximum-score fusion must use a higher clean threshold to control false positives.

All methods detect Gaussian, Shot, and Impulse Noise almost perfectly. They remain weak on Zoom Blur, Fog, and JPEG compression. This is evidence against claiming a universal, high-recall, clean-only corruption detector from these signals.

## Reproduction

Script: `scripts/experiment117_paired_detector_fusion.py`.

Results: `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment117_paired_detector_fusion/full_level4_paired_clean_calibrated_fusion_v1/summary.json`.

The run saves one compressed paired feature cache per condition and can resume without recomputing finished conditions:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. /home/dr-yougart/miniconda3/envs/vit/bin/python -u scripts/experiment117_paired_detector_fusion.py --run-name full_level4_paired_clean_calibrated_fusion_v1 --batch-size 8 --workers 2 --resume
```

## Interpretation and next decision

The paired evaluation resolves the earlier ambiguity caused by comparing Experiments 114 and 116 on different images. It demonstrates complementary strengths, but not an overall gain from this simple, clean-only max-percentile fusion. Do not tune a new fusion on this confirmation set and call the resulting accuracy independent. If pursuing a stronger detector, reserve a separate development set for score-combination learning and another untouched test set; compare at matched clean false-positive rates and report unknown-corruption recall explicitly.
