# Experiment 121: Detecting an unknown corruption family

## Goal and protocol

Detect whether one input image is corrupted, without a clean counterpart, ground-truth class, or corruption label at inference. The `google/vit-base-patch16-224` backbone is frozen. The detector uses the 152 single-view head, patch, and pixel-statistic features from Experiment 114; only a small histogram-gradient-boosted classifier is fitted.

Development uses one old ImageNet-Sketch image per class for fitting (1,000 images) and a different image per class for clean-only threshold calibration (1,000 images). Each condition uses all 1,000 training source images. The test contains **991 new images**, selected after excluding prior frozen Sketch manifests by SHA-256. The threshold for each detector is fixed at the 95th percentile of its clean validation score. All corruptions use the online `imagecorruptions` implementation at severity 4 after resizing to 224 × 224. This is **not official ImageNet-C**.

Two questions are separated:

1. **Single fixed detector (deployable):** trained on Noise and Blur only, then used unchanged for all 15 conditions. Weather, Appearance, and Digital/Geometry are unseen families.
2. **Leave-one-family-out (diagnostic):** train a different detector for each omitted family, then test on that family. This measures transfer but **is not one deployable detector**; selecting which model to use for an unknown family would itself require an oracle/router.

The clean-only Mahalanobis detector is retrained on the same clean training images and calibrated on the same clean validation images. An all-15-corruption supervised model is an upper reference for *known* corruptions, not evidence for unknown-family detection.

## Results

| Detector | Test clean false-alarm rate | Mean recall across 15 corruptions |
| --- | ---: | ---: |
| Clean-only baseline | 4.64% | 44.61% |
| **Fixed Noise+Blur detector** | **4.84%** | **69.55%** |
| Leave-one-family-out, different model per family | 4.74–5.35% depending on model | 65.90% |
| All-corruptions-known reference | 4.84% | 83.82% |

The correct apples-to-apples **unknown-family** comparison excludes the seven Noise/Blur training corruptions:

| Unseen family | Fixed detector recall | Clean-only recall | Gain |
| --- | ---: | ---: | ---: |
| Weather (3) | 54.9% | 29.8% | +25.1 pp |
| Appearance (2) | 65.0% | 45.0% | +20.0 pp |
| Digital/Geometry (3) | 16.9% | 7.1% | +9.8 pp |
| **All eight unseen algorithms** | **43.2%** | **25.1%** | **+18.1 pp** |

For the eight unseen algorithms, an exploratory paired bootstrap that resamples the 991 source-image indices gives **+18.05 pp [16.86, 19.25]** for the fixed detector versus clean-only. Corruption conditions sharing one source image are averaged *within* each resampled image, not treated as independent images. The fixed detector catches Noise/Blur near perfectly, but this is expected because those families were in its fitting data. On unknown JPEG compression it detects only 15.3%; on unknown Elastic Transform, 6.8%. Thus this is meaningful transfer, **not universal detection**.

## Interpretation

Supervised examples of some corruption mechanisms teach the frozen-ViT feature detector more transferable information than clean-only density modeling. However, generalization is uneven: digital/geometric changes remain hard, and the model has only been tested at severity 4 on the familiar Sketch *domain*. A naturally blurry or low-contrast clean image may also be statistically indistinguishable from a mildly corrupted image from one observation.

The ImageNet-Sketch domain, feature set, and broad experiment direction were informed by earlier work. Although test image hashes are new, this is not a pristine independent benchmark for claiming a final state-of-the-art method. No further tuning should use this 991-image result as validation; a new clean/corrupted domain and fresh images are needed for final confirmation.

## Artifacts

- Script: `scripts/experiment121_leave_family_out_corruption_detector.py`
- Full result: `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment121_leave_family_out_corruption_detector/full_level4_unknown_family_detector_v1/summary.json`
- Per-image paired decisions: `paired_detection_outcomes.npz` in the same result directory.
- Cached features and classifier checkpoints are in that directory; rerun with the same arguments plus `--resume` to reuse completed stages after interruption.
